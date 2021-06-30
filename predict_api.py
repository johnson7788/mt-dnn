#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2020/06/28 4:56 下午
# @File  : api.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :
######################################################
# 改造成一个flask api，
# 包括预测接口api
# /api/predict
######################################################

import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from data_utils.task_def import TaskType
from experiments.exp_def import TaskDefs, EncoderModelType
from torch.utils.data import Dataset, DataLoader, BatchSampler
from mt_dnn.batcher import SingleTaskDataset, Collater
from mt_dnn.model import MTDNNModel
from data_utils.metrics import calc_metrics
from mt_dnn.inference import eval_model
import logging.config

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger("Main")

from flask import Flask, request, jsonify, abort
app = Flask(__name__)

class TorchMTDNNModel(object):
    def __init__(self, verbose=False):
        """
        预测的模型
        :param verbose:
        :type verbose:
        """
        # 任务的配置文件
        self.task_deffile = 'experiments/myexample/my_task_def.yml'
        self.task_defs = None  #解析配置文件后的结果
        # absa 情感分析， dem8是8个维度的判断
        self.task_names = ['absa', 'dem8']
        # 保存每个task需要的一些必要的信息
        self.tasks_info = {}
        # 最大序列长度
        self.max_seq_len = 512
        # 最大的batch_size
        self.predict_batch_size = 64
        # 训练好的模型的存放位置
        self.checkpoint = 'trained_model/absa_dem8.pt'
        # 是否放GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.cuda = True
        else:
            self.device = torch.device("cpu")
            self.cuda = False
        self.load_task()
        self.load_model()
    def load_model(self):
        """
        加载模型
        :return:
        :rtype:
        """
        assert os.path.exists(self.checkpoint), "模型文件不存在"
        if self.cuda:
            self.state_dict = torch.load(self.checkpoint)
        else:
            self.state_dict = torch.load(self.checkpoint, map_location="cpu")
        # 模型的配置参数, # 配置和模型保存到一起了
        self.config = self.state_dict['config']
        # task_id的获取
        for task_name in self.task_names:
            train_datasets = self.config['train_datasets']
            task_id = train_datasets.index(task_name)
            self.tasks_info[task_name]['task_id'] = task_id
        self.config["cuda"] = self.cuda
        ## temp fix
        self.config['fp16'] = False
        self.config['answer_opt'] = 0
        self.config['adv_train'] = False
        del self.state_dict['optimizer']
        # 初始化模型
        self.model = MTDNNModel(self.config, device=self.device, state_dict=self.state_dict)
        # encoder的类型 EncoderModelType.BERT
        encoder_type = self.config.get('encoder_type', EncoderModelType.BERT)
        # collator 函数
        self.collater = Collater(is_train=False, encoder_type=encoder_type)

    def load_task(self):
        """
        读取任务的配置文件
        :return:
        :rtype:
        """
        task_defs = TaskDefs(self.task_deffile)
        self.task_defs = task_defs
        for task_name in self.task_names:
            task_def = task_defs.get_task_def(task_name)
            # eg: DataFormat.PremiseAndOneHypothesis
            data_type = task_defs._data_type_map[task_name]
            # eg: TaskType.Classification
            task_type = task_defs._task_type_map[task_name]
            # eg: Metric.ACC
            metric_meta = task_defs._metric_meta_map[task_name]
            self.tasks_info[task_name] = {
                "task_def": task_def,
                "data_type": data_type,
                "task_type": task_type,
                "metric_meta": metric_meta
            }

    def load_data(self, task_name, data_path='data_my/canonical_data/bert-base-chinese/absa_test.json'):
        # load data， 加载数据集
        test_data_set = SingleTaskDataset(path=data_path, is_train=False, maxlen=self.max_seq_len, task_id=self.tasks_info[task_name]['task_id'], task_def=self.tasks_info[task_name]['task_def'])
        test_data = DataLoader(test_data_set, batch_size=self.predict_batch_size, collate_fn=self.collater.collate_fn,pin_memory=self.cuda)
        return test_data

    def eval(self, task_name):
        test_data = self.load_data(task_name)
        with torch.no_grad():
            # test_metrics eg: acc结果，准确率结果
            # test_predictions: 预测的结果， scores预测的置信度， golds是我们标注的结果，标注的label， test_ids样本id, 打印metrics
            test_metrics, test_predictions, scores, golds, test_ids = eval_model(model=self.model, data=test_data,
                                                                                 metric_meta=self.tasks_info[task_name]['metric_meta'],
                                                                                 device=self.device,
                                                                                 with_label=True)
            results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids, 'scores': scores}
            print(f"测试的数据总量是{len(test_ids)}, 测试的结果是{test_metrics}")


@app.route("/api/absa_predict", methods=['POST'])
def absa_predict():
    """
    情感的预测接收POST请求，获取data参数, data信息包含aspect关键在在句子中的位置信息，方便我们截取，我们截取aspect关键字的前后一定的字符作为输入
    例如关键字前后的25个字作为sentenceA，aspect关键字作为sentenceB，输入模型
    Args:
        test_data: 需要预测的数据，是一个文字列表, [(content,aspect,start_idx, end_idx),...,]
        如果传过来的数据没有索引，那么需要自己去查找索引 [(content,aspect),...,]
    Returns: 返回格式是 [(predicted_label, predict_score),...]
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    results = model.eval(test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/dem8_predict", methods=['POST'])
def dem8_predict():
    """
    8个维度的预测，接收POST请求，获取data参数, data信息包含aspect关键在在句子中的位置信息，方便我们截取，我们截取aspect关键字的前后一定的字符作为输入
    例如关键字前后的25个字作为sentenceA，aspect关键字作为sentenceB，输入模型
    Args:
        test_data: 需要预测的数据，是一个文字列表, [(content,aspect,start_idx, end_idx),...,]
        如果传过来的数据没有索引，那么需要自己去查找索引 [(content,aspect),...,]
    Returns: 返回格式是 [(predicted_label, predict_score),...]
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    results = model.eval(test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

if __name__ == "__main__":
    model = TorchMTDNNModel()
    app.run(host='0.0.0.0', port=5018, debug=True, threaded=True)