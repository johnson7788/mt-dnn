#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/8/5 3:12 下午
# @File  : filter_wrong.py
# @Author: johnson
# @Desc  : 使用多个随机数种子训练模型，然后过滤出所有预测错误的样本，供以后进行分析
import argparse
import os
from experiments.myexample.mydata_prepro import do_prepro, absa_source_file, dem8_source_file, purchase_source_file


def got_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default="1,2",help='随机数种子,用逗号隔开，有几个种子，就运行几次')
    parser.add_argument("--task", type=str, default="all", help='对哪个任务进行预测错误的筛选，默认所有')
    parser.add_argument("--wrong_path", type=str, default="wrong_sample/0805", help="预测错误的样本默认保存到哪个文件夹下，错误的样本保存成pkl格式，文件名字用任务名+随机数种子命名")
    args = parser.parse_args()
    return args

def train_and_filter(args):
    #解析参数
    seed = args.seed
    seeds = seed.split(",")
    # 数字格式的随机数种子
    seeds = list(map(int, seeds))
    #对任务进行过滤
    all_tasks = ["absa","dem8","purchase"]
    if args.task == "all":
        tasks = all_tasks
    else:
        assert args.task in all_tasks, "给定任务不在我们预设的任务中，请检查"
        tasks = [args.task]
    # 检查保存目录
    wrong_path = args.wrong_path
    if not os.path.exists(wrong_path):
        os.makedirs(wrong_path)
    for sd in seeds:
        # 根据同一份源数据，不同的随机数种子，产生不同的训练，评估，测试数据集
        # 随机数种子不同，产生的训练评估和测试的样本也不同，这里返回它们的id
        absa_ids, dems_ids, purchase_ids = do_prepro(root='data_my', use_pkl=True, seed=sd)
        absa_train_data_id, absa_dev_data_id, absa_test_data_id = absa_ids
        dem8_train_data_id, dem8_dev_data_id, dem8_test_data_id = dems_ids
        purchase_train_data_id, purchase_dev_data_id, purchase_test_data_id = purchase_ids
        # 第二步，源数据变token
        os.system(command="python prepro_std.py --model bert-base-chinese --root_dir data_my/canonical_data --task_def experiments/myexample/my_task_def.yml --do_lower_case")
        # 第三步，训练模型
        model_output_dir = f"checkpoints/mtdnn_seed{sd}"
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
        command = f"python train.py {train_options}"
        os.system(command=command)
        # 训练完成，使用训练完成的最后的epoch模型进行预测
        model_path = os.path.join(model_output_dir, "model_final.pt")


if __name__ == '__main__':
    args = got_args()
    train_and_filter(args)