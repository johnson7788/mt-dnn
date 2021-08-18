#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/8/5 3:12 下午
# @File  : filter_wrong.py
# @Author: johnson
# @Desc  : 使用多个随机数种子训练模型，然后过滤出所有预测错误的样本，供以后进行分析
import argparse
import json
import os

import pandas as pd

from experiments.myexample.mydata_prepro import do_prepro, absa_source_file, dem8_source_file, purchase_source_file
from predict import do_predict
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
mpl.rcParams['font.family'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def got_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--do_train_filter", action="store_true", help='训练模型并过滤badcase')
    parser.add_argument("-d","--seed", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",help='随机数种子,用逗号隔开，有几个种子，就运行几次')
    parser.add_argument("-k","--task", type=str, default="all", help='对哪个任务进行预测错误的筛选，默认所有')
    parser.add_argument("-w","--wrong_path", type=str, default="wrong_sample/0811", help="预测错误的样本默认保存到哪个文件夹下，错误的样本保存成pkl格式，文件名字用随机数种子命名,包含所有任务的结果")

    #分析badcase的参数
    parser.add_argument("-a","--do_analysis", action="store_true", help='分析badcase')
    parser.add_argument("-p","--analysis_path", type=str, default="wrong_sample/0811", help="分析保存预测错误的样本的文件夹，pkl格式")
    parser.add_argument("-s","--analysis_tasks", type=str, default="accuracy,samplenum,badnum,totalbad,wrongnum,export,golden_class_num,class_wrong_num", help="分析保存预测错误的样本的文件夹，pkl格式")

    args = parser.parse_args()
    return args
def train_and_filter(seed, task ,wrong_path):
    #解析参数
    seeds = seed.split(",")
    # 数字格式的随机数种子
    seeds = list(map(int, seeds))
    #对任务进行过滤
    all_tasks = ["absa","dem8","purchase"]
    if task == "all":
        tasks = all_tasks
    else:
        assert task in all_tasks, "给定任务不在我们预设的任务中，请检查"
        tasks = [task]
    # 检查保存目录
    if not os.path.exists(wrong_path):
        os.makedirs(wrong_path)
    #备份下源数据到wrong_path目录
    os.system(command=f"cp -a data_my/canonical_data/source_data {wrong_path}")
    for sd in seeds:
        wrong_sample_record = os.path.join(wrong_path, f"filter_seed_{sd}.pkl")
        records = collections.defaultdict(dict)
        records['seed'] = sd
        # 根据同一份源数据，不同的随机数种子，产生不同的训练，评估，测试数据集
        # 随机数种子不同，产生的训练评估和测试的样本也不同，这里返回它们的id
        absa_ids, dems_ids, purchase_ids, brand_ids = do_prepro(root='data_my', use_pkl=True, seed=sd)
        # 注意这里的data_id是对应的源数据的索引，是全局唯一的
        absa_train_data_id, absa_dev_data_id, absa_test_data_id = absa_ids
        dem8_train_data_id, dem8_dev_data_id, dem8_test_data_id = dems_ids
        purchase_train_data_id, purchase_dev_data_id, purchase_test_data_id = purchase_ids
        # 第二步，源数据变token
        code = os.system(command="/home/wac/johnson/anaconda3/envs/py38/bin/python prepro_std.py --model bert-base-chinese --root_dir data_my/canonical_data --task_def experiments/myexample/my_task_def.yml --do_lower_case")
        assert code == 0, "数据处理不成功，请检查"
        # 第三步，训练模型
        model_output_dir = f"checkpoints/mtdnn_seed_{sd}"
        train_options_list = {
            'data_dir': "--data_dir data_my/canonical_data/bert-base-chinese",  # 数据tokenize后的路径
            'init_checkpoint': "--init_checkpoint mt_dnn_models/bert_model_base_chinese.pt",  # base模型
            "batch_size": "--batch_size 32",
            "task_def": "--task_def experiments/myexample/my_task_def.yml",
            'output_dir': f"--output_dir {model_output_dir}",
            'log_file': f"--log_file {model_output_dir}/log.log ",
            'answer_opt': "--answer_opt 1 ",  # 可选0,1，代表是否使用SANClassifier分类头还是普通的线性分类头,1表示使用SANClassifier, 0是普通线性映射
            'optimizer': "--optimizer adamax ",
            'epochs': "--epochs 5",
            'train_datasets': "--train_datasets absa,dem8,purchase,brand",
            'test_datasets': "--test_datasets absa,dem8,purchase,brand",
            'grad_clipping': "--grad_clipping 0 ",
            'global_grad_clipping': "--global_grad_clipping 1 ",
            'learning_rate': "--learning_rate 5e-5",
        }
        train_options = " ".join(train_options_list.values())
        command = f"/home/wac/johnson/anaconda3/envs/py38/bin/python train.py {train_options}"
        code = os.system(command=command)
        assert code == 0, "训练模型失败，请检查"
        # 训练完成，使用训练完成的最后的epoch模型进行预测
        model_path = os.path.join(model_output_dir, "model_final.pt")
        tasks2id = {
            "absa": 0,
            "dem8": 1,
            "purchase": 2
        }
        records['model_path'] = model_path
        for task in tasks:
            task_record = {}
            if task == "absa":
                task_record["train_data_id"] = absa_train_data_id
                task_record["dev_data_id"] = absa_dev_data_id
                task_record["test_data_id"] = absa_test_data_id
            elif task == "dem8":
                task_record["train_data_id"] = dem8_train_data_id
                task_record["dev_data_id"] = dem8_dev_data_id
                task_record["test_data_id"] = dem8_test_data_id
            else:
                task_record["train_data_id"] = purchase_train_data_id
                task_record["dev_data_id"] = purchase_dev_data_id
                task_record["test_data_id"] = purchase_test_data_id
            task_id = tasks2id[task]
            # 对训练，测试，开发数据集都进行预测一下
            # 对于每个任务都进行测试
            datasets = ["train", "dev", "test"]
            for dataset in datasets:
                prep_input = f"data_my/canonical_data/bert-base-chinese/{task}_{dataset}.json"
                # 预测结果
                test_metrics, predict_labels, scores, gold_labels, _ = do_predict(task, task_def="experiments/myexample/my_task_def.yml", task_id=task_id, prep_input=prep_input, with_label=True, score="predict_score.txt", max_seq_len=512, batch_size_eval=32, checkpoint=model_path,
                           cuda=True, do_collection=False, collection_file=None)
                task_record[f"{dataset}_metrics"] = test_metrics
                task_record[f"{dataset}_predict_labels"] = predict_labels
                task_record[f"{dataset}_scores"] = scores
                task_record[f"{dataset}_gold_labels"] = gold_labels
            records[task] = task_record
        # 保存一次实验的seed结果
        with open(wrong_sample_record, 'w') as f:
            json.dump(records,f)
def do_analysis(analysis_path, analysis_tasks):
    """
    分析badcase
    :param analysis_tasks 绘哪个图
    :return:
    :rtype:
    """
    tasks = ["absa", "dem8", "purchase"]
    assert os.path.exists(analysis_path), f"给定的分析的路径不存在:{analysis_path}，请检查目录是否正确"
    files = os.listdir(analysis_path)
    assert "source_data" in files, "原始数据目录不在里面，请检查"
    absa_src_data = os.path.join(analysis_path, "source_data", "absa.pkl")
    dem8_src_data = os.path.join(analysis_path, "source_data", "dem8.pkl")
    purchase_src_data = os.path.join(analysis_path, "source_data", "purchase.pkl")
    assert os.path.exists(absa_src_data), "absa的原始数据文件不存在，请检查"
    assert os.path.exists(dem8_src_data), "dem8的原始数据文件不存在，请检查"
    assert os.path.exists(purchase_src_data), "purchase的原始数据文件不存在，请检查"
    # 每个随机数种子训练模型后的结果
    seed_pkl = [f for f in files if f.endswith('.pkl')]
    # 读取每个运行结果
    seeds_result = []
    for sd_file in seed_pkl:
        #读取每个记录的pkl文件
        sd_file_path = os.path.join(analysis_path, sd_file)
        with open(sd_file_path, 'rb') as f:
            #单次的运行结果
            sd_res = json.load(f)
            # 预处理下sd_res，为了以后的绘图更方便，主要统计下预测错误的样本，bad_case的基本信息
        seeds_result.append(sd_res)
    if "accuracy" in analysis_tasks:
        # 准确率的绘制
        analysis_acc(seeds_result)
    #样本训练集，开发集，测试集数量绘制
    if "samplenum" in analysis_tasks:
        analysis_sample_num(seeds_result)
    if "badnum" in analysis_tasks:
        # 只画出每次seed的错误的样本数量
        analysis_bad_sample_num(seeds_result)
    if "totalbad" in analysis_tasks:
        # 分析总的错误的样本，重复出错的和只出错一次的
        total_bad_sample_num(seeds_result)
    if "wrongnum" in analysis_tasks:
        # 所有的预测样本错误的的次数的直方图，预测错误1次的有x个，预测错误2次的有y个，预测错误3次的有z个，....
        total_bad_sample_bar(seeds_result)
    if "export" in analysis_tasks:
        export_wrong_data(absa_src_data, dem8_src_data, purchase_src_data, seeds_result)
    # 分析样本中每个类别的真实数量
    if "golden_class_num" in analysis_tasks:
        analysis_every_class_num(seeds_result)
    if "class_wrong_num" in analysis_tasks:
        # 每个任务的每个类别的正确率分析, 每个label的准确率
        analysis_every_class_wrong(seeds_result)
def analysis_every_class_num(seeds_result):
    def collect_value(sd_res,task):
        # 所有的标签，一共有多少个标签
        class_names = sd_res[task]['train_gold_labels'] + sd_res[task]['dev_gold_labels'] + sd_res[task]['test_gold_labels']
        #各个标签的百分百
        class_count = collections.Counter(class_names)
        class_name = list(class_count.keys())
        class_num = list(class_count.values())
        total_num = len(class_names)
        class_percent = [num/total_num for num in class_num]
        return class_name, class_num, class_percent
    sd_res = seeds_result[0]
    class_name, class_num, class_percent = collect_value(sd_res,"absa")
    simple_bar_plot(x=class_name, y=class_num, title="absa任务的每个类别的gold-label数量", xname="标签名称", yname="gold-label数量")
    simple_bar_plot(x=class_name, y=class_percent, title="absa任务的每个类别的gold-label数量占比", xname="标签名称", yname="gold-label占比")
    class_name, class_num, class_percent = collect_value(sd_res,"dem8")
    simple_bar_plot(x=class_name, y=class_num, title="dem8任务的每个类别的gold-label数量", xname="标签名称", yname="gold-label数量")
    simple_bar_plot(x=class_name, y=class_percent, title="dem8任务的每个类别的gold-label数量占比", xname="标签名称", yname="gold-label占比")
    class_name, class_num, class_percent = collect_value(sd_res,"purchase")
    simple_bar_plot(x=class_name, y=class_num, title="purchase任务的每个类别的gold-label数量", xname="标签名称", yname="gold-label数量")
    simple_bar_plot(x=class_name, y=class_percent, title="purchase任务的每个类别的gold-label数量占比", xname="标签名称", yname="gold-label占比")
def analysis_every_class_wrong(seeds_result):
    """
    分析每个任务的每个类别的正确率
    :param seeds_result:
    :type seeds_result:
    :return:
    :rtype:
    """
    def compaire(a, b, cnt_wrong):
        """
        :param a: 预测的标签
        :type a:
        :param b: 真实的标签
        :type b:
        :return:
        :rtype:
        """
        assert len(a) == len(b), "a和b的数量应该相等"
        for predict, gold in zip(a, b):
            if predict != gold:
                cnt_wrong[gold] +=1
        return cnt_wrong
    def collect_value(sd_res,task):
        # 所有的标签，一共有多少个标签
        class_names = sd_res[task]['train_gold_labels'] + sd_res[task]['dev_gold_labels'] + sd_res[task]['test_gold_labels']
        #各个标签的百分百
        class_count = collections.Counter(class_names)
        # 所有的标签，一共有多少个标签
        class_name = []
        #总的类别数
        class_num = []
        #预测错误的类别数
        wrong_num = []
        #预测错误的百分百
        wrong_percent = []
        # 对于不相等的label进行统计
        cnt_wrong = collections.Counter()
        cnt_wrong = compaire(sd_res[task]['train_predict_labels'], sd_res[task]['train_gold_labels'],cnt_wrong)
        cnt_wrong = compaire(sd_res[task]['dev_predict_labels'], sd_res[task]['dev_gold_labels'],cnt_wrong)
        cnt_wrong = compaire(sd_res[task]['test_predict_labels'], sd_res[task]['test_gold_labels'], cnt_wrong)
        for name, clsnum in class_count.items():
            wnum = cnt_wrong.get(name,0)
            class_name.append(name)
            class_num.append(clsnum)
            wrong_num.append(wnum)
            wrong_percent.append(wnum/clsnum)
        return class_name, class_num,wrong_num, wrong_percent
    for sd_res in seeds_result[:2]:
        # 只画前3个图
        seed = sd_res['seed']
        for task in ["absa","dem8","purchase"]:
            class_name, class_num, wrong_num, wrong_percent = collect_value(sd_res,task)
            # yvalue = list(zip(class_num, wrong_num, wrong_percent))
            simple_bar_plot(x=class_name, y=class_num, title=f"seed数为{seed}的{task}任务预测总数量", xname="类别的名称", yname="类别总样本数")
            simple_bar_plot(x=class_name, y=wrong_num, title=f"seed数为{seed}的{task}任务预测错误数量", xname="类别的名称", yname="类别错误数量")
            simple_bar_plot(x=class_name, y=wrong_percent, title=f"seed数为{seed}的{task}任务预测错误百分比", xname="类别的名称", yname="类别错误百分比")
def export_wrong_data(absa_src_data, dem8_src_data, purchase_src_data, seeds_result):
    """
    导出预测错误的样本
    :return:
    :rtype:
    """
    def compaire(predict, gold, sample_id):
        diff_id = []
        for p, g, s in zip(predict, gold, sample_id):
            if p != g:
                diff_id.append(s)
        return diff_id
    def collect_value(sd_res, task):
        # 返回预测错误的id, 错误id是对应的源数据的索引，是全局唯一的
        t_id = compaire(sd_res[task]['train_predict_labels'], sd_res[task]['train_gold_labels'],
                        sd_res[task]['train_data_id'])
        d_id = compaire(sd_res[task]['dev_predict_labels'], sd_res[task]['dev_gold_labels'],
                        sd_res[task]['train_data_id'])
        s_id = compaire(sd_res[task]['test_predict_labels'], sd_res[task]['test_gold_labels'],
                        sd_res[task]['train_data_id'])
        merge_id = t_id + d_id + s_id
        return merge_id
    absa_counter = collections.Counter()
    dem8_counter = collections.Counter()
    purchase_counter = collections.Counter()
    for sd_res in seeds_result:
        absa_bad_id = collect_value(sd_res, "absa")
        absa_counter.update(absa_bad_id)
        dem8_bad_id = collect_value(sd_res, "dem8")
        dem8_counter.update(dem8_bad_id)
        purchase_bad_id = collect_value(sd_res, "purchase")
        purchase_counter.update(purchase_bad_id)
    #按错误的次数排序id
    sorted_absa = sorted(absa_counter.items(), key=lambda x: x[1],reverse=True)
    sorted_dem8 = sorted(dem8_counter.items(), key=lambda x: x[1],reverse=True)
    sorted_purchase = sorted(purchase_counter.items(), key=lambda x: x[1],reverse=True)
    #索引原文，并导入到excel
    with open(absa_src_data, "rb") as f:
        absa_data = pickle.load(f)
    with open(dem8_src_data, "rb") as f:
        dem8_data = pickle.load(f)
    with open(purchase_src_data, "rb") as f:
        purchase_data = pickle.load(f)
    #把数据变成字典的索引，那样，方便查找
    absa_data_dict = {idx:i for idx, i in enumerate(absa_data)}
    dem8_data_dict = {idx:i for idx, i in enumerate(dem8_data)}
    purchase_data_dict = {idx:i for idx, i in enumerate(purchase_data)}
    # 保存的excel的名称
    absa_save_excel = "absa_wrong.xlsx"
    dem8_save_excel = "dem8_wrong.xlsx"
    purchase_save_excel = "purchase_wrong.xlsx"
    def saved_data(sorted_data, saved_excel, src_data, columns):
        col_data = []
        for i in sorted_data:
            id, wrong_num = i
            data = src_data[id]
            data = list(data)
            data.append(wrong_num)
            data.append(str(id))
            col_data.append(data)
            # id也加上
        df = pd.DataFrame(col_data, columns=columns)
        df.to_excel(saved_excel)
    # label是真实的标签
    saved_data(sorted_data=sorted_absa, saved_excel=absa_save_excel, src_data=absa_data_dict,  columns=['text','keyword','start','end','label','channel','wordtype','md5','wrong_num','id'])
    saved_data(sorted_data=sorted_dem8, saved_excel=dem8_save_excel, src_data=dem8_data_dict, columns=['text','keyword','start','end','label','channel','wordtype','md5','wrong_num','id'])
    saved_data(sorted_data=sorted_purchase, saved_excel=purchase_save_excel, src_data=purchase_data_dict,columns=['text','title','keyword','start','end','label','md5','wrong_num','id'])
    print(f"保存预测错误文件到{absa_save_excel}, {dem8_save_excel}, {purchase_save_excel}")
def simple_bar_plot(x, y, title, xname, yname):
    """
    普通的柱状图
    :param x: eg: [1, 2, 3, 4]
    :type x:  是对应的x轴
    :param y: eg: [1, 4, 9, 16]
    :type y:
    :param title: 绘图的标题
    :param xname: "预测错误次数"
    :type xname:
    :param yname: "样本数量"
    :type yname:
    :return:
    :rtype:
    """
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, y, width=0.3)
    ax.bar_label(rects1, padding=3)
    ax.set_title(title)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()
def total_bad_sample_bar(seeds_result):
    """
    统计几次seed中预测错误的样本的重复次数
    :param seeds_result:
    :type seeds_result:
    :return:
    :rtype:
    """
    def compaire(predict,gold,sample_id):
        diff_id = []
        for p,g,s in zip(predict,gold,sample_id):
            if p != g:
                diff_id.append(s)
        return diff_id
    def collect_value(sd_res,task):
        # 返回预测错误的id, 错误id是对应的源数据的索引，是全局唯一的
        t_id = compaire(sd_res[task]['train_predict_labels'], sd_res[task]['train_gold_labels'],sd_res[task]['train_data_id'])
        d_id = compaire(sd_res[task]['dev_predict_labels'], sd_res[task]['dev_gold_labels'],sd_res[task]['train_data_id'])
        s_id = compaire(sd_res[task]['test_predict_labels'], sd_res[task]['test_gold_labels'],sd_res[task]['train_data_id'])
        merge_id = t_id + d_id + s_id
        return merge_id
    absa_counter = collections.Counter()
    dem8_counter = collections.Counter()
    purchase_counter = collections.Counter()
    for sd_res in seeds_result:
        absa_bad_id = collect_value(sd_res,"absa")
        absa_counter.update(absa_bad_id)
        dem8_bad_id = collect_value(sd_res, "dem8")
        dem8_counter.update(dem8_bad_id)
        purchase_bad_id = collect_value(sd_res, "purchase")
        purchase_counter.update(purchase_bad_id)
    #统计和绘图
    # 错误次数出现1次的样本
    absa_wrong_count = collections.Counter([count for id, count in absa_counter.items()])
    dem8_wrong_count = collections.Counter([count for id, count in dem8_counter.items()])
    purchase_wrong_count = collections.Counter([count for id, count in purchase_counter.items()])
    x_absa = list(absa_wrong_count.keys())
    y_absa = list(absa_wrong_count.values())
    x_dem8 = list(dem8_wrong_count.keys())
    y_dem8 = list(dem8_wrong_count.values())
    x_purchase = list(purchase_wrong_count.keys())
    y_purchase = list(purchase_wrong_count.values())
    #错误次数大于1次的样本数量
    absa_more1 = [v for k,v in absa_wrong_count.items() if k >1]
    dem8_more1 = [v for k,v in dem8_wrong_count.items() if k >1]
    purchase_more1 = [v for k,v in purchase_wrong_count.items() if k >1]
    x1 = ["1次", "n次"]
    y_absa1 = [sum(y_absa)-sum(absa_more1),sum(absa_more1)]
    y_dem81 = [sum(y_dem8)-sum(dem8_more1),sum(dem8_more1)]
    y_purchase1 = [sum(y_purchase)-sum(purchase_more1),sum(purchase_more1)]
    simple_bar_plot(x1, y_absa1, title="情感任务absa的预测错误的样本的频次1次和多次 ", xname="错误频次", yname="样本数量")
    simple_bar_plot(x1, y_dem81, title="属性判断dem8的预测错误的样本的频次1次和多次 ", xname="错误频次", yname="样本数量")
    simple_bar_plot(x1, y_purchase1, title="购买意向purchase的预测错误的样本的频次1次和多次 ", xname="错误频次", yname="样本数量")
    simple_bar_plot(x_absa, y_absa, title="情感任务absa的预测错误的样本的频次", xname="错误频次", yname="样本数量")
    simple_bar_plot(x_dem8, y_dem8, title="属性判断dem8的预测错误的样本的频次", xname="错误频次", yname="样本数量")
    simple_bar_plot(x_purchase, y_purchase, title="购买意向purchase的预测错误的样本的频次", xname="错误频次", yname="样本数量")
def total_bad_sample_num(seeds_result):
    """
    几次seed预测后，总的预测错误的样本数量，总的出错数量，一个是重复出错的样本的数量，一个是几次seed后只有一次的出错的数量，会去重
    :param seeds_result:
    :type seeds_result:
    :return:
    :rtype:
    """
    plot_x = ["1"]  #没用到
    absa_plot_acc_data = []
    dem8_plot_acc_data = []
    purchase_plot_acc_data = []
    def compaire(predict,gold,sample_id):
        diff_id = []
        for p,g,s in zip(predict,gold,sample_id):
            if p != g:
                diff_id.append(s)
        return diff_id
    def collect_value(sd_res,task):
        # 返回预测错误的id, 错误id是对应的源数据的索引，是全局唯一的
        t_id = compaire(sd_res[task]['train_predict_labels'], sd_res[task]['train_gold_labels'],sd_res[task]['train_data_id'])
        d_id = compaire(sd_res[task]['dev_predict_labels'], sd_res[task]['dev_gold_labels'],sd_res[task]['train_data_id'])
        s_id = compaire(sd_res[task]['test_predict_labels'], sd_res[task]['test_gold_labels'],sd_res[task]['train_data_id'])
        merge_id = t_id + d_id + s_id
        return merge_id
    absa_counter = collections.Counter()
    dem8_counter = collections.Counter()
    purchase_counter = collections.Counter()
    for sd_res in seeds_result:
        absa_bad_id = collect_value(sd_res,"absa")
        absa_counter.update(absa_bad_id)
        dem8_bad_id = collect_value(sd_res, "dem8")
        dem8_counter.update(dem8_bad_id)
        purchase_bad_id = collect_value(sd_res, "purchase")
        purchase_counter.update(purchase_bad_id)
    #统计和绘图
    # 错误次数出现1次的样本
    a1 = {x: count for x, count in absa_counter.items() if count == 1}
    d1 = {x: count for x, count in dem8_counter.items() if count == 1}
    p1 = {x: count for x, count in purchase_counter.items() if count == 1}
    #预测错误多于一次的
    ma = len(absa_counter) - len(a1)
    md = len(dem8_counter) - len(d1)
    mp = len(purchase_counter) - len(p1)
    absa_plot_acc_data.append([len(absa_counter),len(a1),ma])
    dem8_plot_acc_data.append([len(dem8_counter), len(d1), md])
    purchase_plot_acc_data.append([len(purchase_counter), len(p1), mp])
    plot_bar(title="所有seed情感任务absa的汇总预测错误",yname="样本数",seeds=plot_x, yvalue=absa_plot_acc_data,xname="汇总预测错误",bar_group_labels=["总错误数","错误一次数","错误n次数"])
    plot_bar(title="所有seed属性判断dem8的汇总预测错误",yname="样本数",seeds=plot_x, yvalue=dem8_plot_acc_data,xname="汇总预测错误",bar_group_labels=["总错误数","错误一次数","错误n次数"])
    plot_bar(title="所有seed购买意向purchase的汇总预测错误",yname="样本数",seeds=plot_x, yvalue=purchase_plot_acc_data,xname="汇总预测错误",bar_group_labels=["总错误数","错误一次数","错误n次数"])
def analysis_bad_sample_num(seeds_result):
    """
    读取每个seed种子的结果，绘制错误的样本的数量
    :param seeds_result:
    :type seeds_result:
    :return:
    :rtype:
    """
    def compaire(a, b):
        assert len(a) == len(b), "a和b的数量应该相等"
        same_sample = [i for i, j in zip(a, b) if i == j]
        diff_num = len(a) - len(same_sample)
        return diff_num, len(same_sample)
    plot_seeds = []
    absa_plot_acc_data = []
    dem8_plot_acc_data = []
    purchase_plot_acc_data = []
    def collect_value(sd_res,task):
        a, _ = compaire(sd_res[task]['train_predict_labels'], sd_res[task]['train_gold_labels'])
        b, _ = compaire(sd_res[task]['dev_predict_labels'], sd_res[task]['dev_gold_labels'])
        c, _ = compaire(sd_res[task]['test_predict_labels'], sd_res[task]['test_gold_labels'])
        return a,b,c
    for sd_res in seeds_result:
        seed = sd_res['seed']
        plot_seeds.append(seed)
        a,b,c = collect_value(sd_res,"absa")
        absa_plot_acc_data.append([a,b,c])
        # dem8的准确率收集
        a, b, c = collect_value(sd_res, "dem8")
        dem8_plot_acc_data.append([a, b, c])
        # purchase
        a, b, c = collect_value(sd_res, "purchase")
        purchase_plot_acc_data.append([a, b, c])
    plot_bar(title="情感任务absa的预测错误样本数",yname="样本数",seeds=plot_seeds, yvalue=absa_plot_acc_data)
    plot_bar(title="属性判断dem8的预测错误样本数",yname="样本数",seeds=plot_seeds, yvalue=dem8_plot_acc_data)
    plot_bar(title="购买意向purchase的预测错误样本数",yname="样本数",seeds=plot_seeds, yvalue=purchase_plot_acc_data)
def analysis_sample_num(seeds_result):
    """
    读取每个seed种子的结果，绘制样本数量，样本数量基本是一样的
    :param seeds_result:
    :type seeds_result:
    :return:
    :rtype:
    """
    plot_seeds = []
    absa_plot_acc_data = []
    dem8_plot_acc_data = []
    purchase_plot_acc_data = []
    for sd_res in seeds_result:
        seed = sd_res['seed']
        plot_seeds.append(seed)
        absa_train_acc = len(sd_res['absa']['train_data_id'])
        absa_dev_acc = len(sd_res['absa']['dev_data_id'])
        absa_test_acc = len(sd_res['absa']['test_data_id'])
        absa_plot_acc_data.append([absa_train_acc,absa_dev_acc,absa_test_acc])
        # dem8的准确率收集
        dem8_train_acc = len(sd_res['dem8']['train_data_id'])
        dem8_dev_acc = len(sd_res['dem8']['dev_data_id'])
        dem8_test_acc = len(sd_res['dem8']['test_data_id'])
        dem8_plot_acc_data.append([dem8_train_acc,dem8_dev_acc,dem8_test_acc])
        # purchase
        purchase_train_acc = len(sd_res['purchase']['train_data_id'])
        purchase_dev_acc = len(sd_res['purchase']['dev_data_id'])
        purchase_test_acc = len(sd_res['purchase']['test_data_id'])
        purchase_plot_acc_data.append([purchase_train_acc, purchase_dev_acc, purchase_test_acc])
    plot_bar(title="情感任务absa的总样本数",yname="样本数",seeds=plot_seeds, yvalue=absa_plot_acc_data)
    plot_bar(title="属性判断dem8的总样本数",yname="样本数",seeds=plot_seeds, yvalue=dem8_plot_acc_data)
    plot_bar(title="购买意向purchase的总样本数",yname="样本数",seeds=plot_seeds, yvalue=purchase_plot_acc_data)
def analysis_acc(seeds_result):
    """
    读取每个seed种子的结果，绘图准确率
    :param seeds_result:
    :type seeds_result:
    :return:
    :rtype:
    """
    plot_seeds = []
    absa_plot_acc_data = []
    dem8_plot_acc_data = []
    purchase_plot_acc_data = []
    for sd_res in seeds_result:
        seed = sd_res['seed']
        plot_seeds.append(seed)
        absa_train_acc = sd_res['absa']['train_metrics']['ACC']
        absa_dev_acc = sd_res['absa']['dev_metrics']['ACC']
        absa_test_acc = sd_res['absa']['test_metrics']['ACC']
        absa_plot_acc_data.append([absa_train_acc,absa_dev_acc,absa_test_acc])
        # dem8的准确率收集
        dem8_train_acc = sd_res['dem8']['train_metrics']['ACC']
        dem8_dev_acc = sd_res['dem8']['dev_metrics']['ACC']
        dem8_test_acc = sd_res['dem8']['test_metrics']['ACC']
        dem8_plot_acc_data.append([dem8_train_acc,dem8_dev_acc,dem8_test_acc])
        # purchase
        purchase_train_acc = sd_res['purchase']['train_metrics']['ACC']
        purchase_dev_acc = sd_res['purchase']['dev_metrics']['ACC']
        purchase_test_acc = sd_res['purchase']['test_metrics']['ACC']
        purchase_plot_acc_data.append([purchase_train_acc, purchase_dev_acc, purchase_test_acc])
    #平均的准确率
    def average_acc(acc_data):
        # 加一个平均值，在末尾
        avg_train = sum([i[0] for i in acc_data])/len(acc_data)
        avg_dev = sum([i[1] for i in acc_data])/len(acc_data)
        avg_test = sum([i[2] for i in acc_data])/len(acc_data)
        acc_data.append([avg_train,avg_dev,avg_test])
        return acc_data
    absa_plot_acc_data = average_acc(absa_plot_acc_data)
    dem8_plot_acc_data = average_acc(dem8_plot_acc_data)
    purchase_plot_acc_data = average_acc(purchase_plot_acc_data)
    # 999代表平均值
    plot_seeds.append(999)
    plot_bar(title="情感任务absa的准确率",yname="准确率",seeds=plot_seeds, yvalue=absa_plot_acc_data, ylimit=[0, 100])
    plot_bar(title="属性判断dem8的准确率",yname="准确率",seeds=plot_seeds, yvalue=dem8_plot_acc_data,ylimit=[0, 100])
    plot_bar(title="购买意向purchase的准确率",yname="准确率",seeds=plot_seeds, yvalue=purchase_plot_acc_data,ylimit=[0, 100])
    print(f"情感任务absa的准确率: {absa_plot_acc_data}")
    print(f"属性判断dem8的准确率: {dem8_plot_acc_data}")
    print(f"购买意向purchase的准确率: {purchase_plot_acc_data}")
def plot_bar(title,yname,seeds, yvalue, ylimit=None,xname="随机数种子",bar_group_labels=["训练集","开发集","测试集"]):
    """
    绘制准确率的柱状图
    :param title:  绘图显示的标题
    :type title:
    :param seeds: 随机数种子的列表，是标签而已
    :type seeds:
    :param yvalue: 纵坐标的值，例如: 准确率的列表，嵌套的列表，每个列表是【训练集，开发集，测试集】结果
    :type yvalue:
    :param ylimit: y轴的大小
    :param xname: x轴的名字
    :param bar_group_labels: 一组中，每个bar代表的名字，对应yvalue的值
    :return:
    :rtype:
    """
    ## matplotlib 3.4.2版本以上支持
    # 横坐标
    # 给柱状图分配位置和宽度
    x = np.arange(len(seeds))  # the label locations
    width = 0.6/3  # Bar的宽度
    yvalue = np.array(yvalue).T

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, yvalue[0], width, label=bar_group_labels[0])
    rects2 = ax.bar(x + width, yvalue[1], width, label=bar_group_labels[1])
    rects3 = ax.bar(x, yvalue[2], width, label=bar_group_labels[2])
    # 设置y坐标轴长度
    if ylimit:
        ax.set_ylim(ylimit)

    # 横坐标和纵坐标的设置
    ax.set_ylabel(yname)
    ax.set_title(title)
    ax.set_xlabel(xname)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    # 显示legend，即说明,哪个bar的颜色是哪个
    ax.legend()

    # 给每个bar上面加显示的数值， padding是距离bar多高的位置显示数字
    ax.bar_label(rects1, padding=3,color='blue')
    ax.bar_label(rects2, padding=15,color='orange')
    ax.bar_label(rects3, padding=30,color='green')
    # 紧凑显示，显示的图更大一些
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = got_args()
    if args.do_train_filter:
        train_and_filter(seed=args.seed, task=args.task ,wrong_path=args.wrong_path)
    else:
        #分析
        plot_tasks = args.analysis_tasks.split(',')
        do_analysis(analysis_path=args.analysis_path, analysis_tasks=plot_tasks)