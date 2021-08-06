#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/8/5 3:12 下午
# @File  : filter_wrong.py
# @Author: johnson
# @Desc  : 使用多个随机数种子训练模型，然后过滤出所有预测错误的样本，供以后进行分析
import argparse
import json
import os
from experiments.myexample.mydata_prepro import do_prepro, absa_source_file, dem8_source_file, purchase_source_file
from predict import do_predict
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def got_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--do_train_filter", action="store_true", help='训练模型并过滤badcase')
    parser.add_argument("--seed", type=str, default="1,2",help='随机数种子,用逗号隔开，有几个种子，就运行几次')
    parser.add_argument("--task", type=str, default="all", help='对哪个任务进行预测错误的筛选，默认所有')
    parser.add_argument("--wrong_path", type=str, default="wrong_sample/0806", help="预测错误的样本默认保存到哪个文件夹下，错误的样本保存成pkl格式，文件名字用随机数种子命名,包含所有任务的结果")

    #分析badcase的参数
    parser.add_argument("-a","--do_analysis", action="store_true", help='分析badcase')
    parser.add_argument("--analysis_path", type=str, default="wrong_sample/0806", help="分析保存预测错误的样本的文件夹，pkl格式")


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
        absa_ids, dems_ids, purchase_ids = do_prepro(root='data_my', use_pkl=True, seed=sd)
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
            'train_datasets': "--train_datasets absa,dem8,purchase",
            'test_datasets': "--test_datasets absa,dem8,purchase",
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

def do_analysis(analysis_path):
    """
    分析badcase
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
        seeds_result.append(sd_res)
    #准确率的绘制
    analysis_acc(seeds_result)
    analysis_sample_num(seeds_result)

def analysis_sample_num(seeds_result):
    """
    读取每个seed种子的结果，绘制样本数量，样本数量基本是一样的
    :param seeds_result:
    :type seeds_result:
    :return:
    :rtype:
    """
    plot_seeds = [1,2]
    absa_plot_acc_data = []
    dem8_plot_acc_data = []
    purchase_plot_acc_data = []
    for sd_res in seeds_result:
        # plot_seeds.append()
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
    plot_bar(title="情感任务absa的样本数",yname="样本数",seeds=plot_seeds, yvalue=absa_plot_acc_data)
    plot_bar(title="属性判断dem8的样本数",yname="样本数",seeds=plot_seeds, yvalue=dem8_plot_acc_data)
    plot_bar(title="购买意向purchase的样本数",yname="样本数",seeds=plot_seeds, yvalue=purchase_plot_acc_data)

def analysis_acc(seeds_result):
    """
    读取每个seed种子的结果，绘图准确率
    :param seeds_result:
    :type seeds_result:
    :return:
    :rtype:
    """
    plot_seeds = [1,2]
    absa_plot_acc_data = []
    dem8_plot_acc_data = []
    purchase_plot_acc_data = []
    for sd_res in seeds_result:
        # plot_seeds.append()
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
    plot_bar(title="情感任务absa的准确率",yname="准确率",seeds=plot_seeds, yvalue=absa_plot_acc_data, ylimit=[0, 100])
    plot_bar(title="属性判断dem8的准确率",yname="准确率",seeds=plot_seeds, yvalue=dem8_plot_acc_data,ylimit=[0, 100])
    plot_bar(title="购买意向purchase的准确率",yname="准确率",seeds=plot_seeds, yvalue=purchase_plot_acc_data,ylimit=[0, 100])

def plot_bar(title,yname,seeds, yvalue, ylimit=None):
    """
    绘制准确率的柱状图
    :param title:  绘图显示的标题
    :type title:
    :param seeds: 随机数种子的列表
    :type seeds:
    :param yvalue: 纵坐标的值，例如: 准确率的列表，嵌套的列表，每个列表是【训练集，开发集，测试集】结果
    :type yvalue:
    :param ylimit: y轴的大小
    :return:
    :rtype:
    """
    mpl.rcParams['font.family'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    ## matplotlib 3.4.2版本以上支持
    # 横坐标
    # 给柱状图分配位置和宽度
    x = np.arange(len(seeds))  # the label locations
    width = 0.6/3  # Bar的宽度
    yvalue = np.array(yvalue).T

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, yvalue[0], width, label='训练集')
    rects2 = ax.bar(x + width, yvalue[1], width, label='开发集')
    rects3 = ax.bar(x, yvalue[2], width, label='测试集')
    # 设置y坐标轴长度
    if ylimit:
        ax.set_ylim(ylimit)

    # 横坐标和纵坐标的设置
    ax.set_ylabel(yname)
    ax.set_title(title)
    ax.set_xlabel('随机数种子')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    # 显示legend，即说明,哪个bar的颜色是哪个
    ax.legend()

    # 给每个bar上面加显示的数值， padding是距离bar多高的位置显示数字
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    # 紧凑显示，显示的图更大一些
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = got_args()
    if args.do_train_filter:
        train_and_filter(seed=args.seed, task=args.task ,wrong_path=args.wrong_path)
    else:
        #分析
        do_analysis(analysis_path=args.analysis_path)