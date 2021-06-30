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
from mt_dnn.batcher import Collater
from experiments.mlm.mlm_utils import create_instances_from_document
from mt_dnn.model import MTDNNModel
import tasks
import random
from data_utils.metrics import calc_metrics
import numpy as np
from mt_dnn.inference import eval_model
from data_utils.task_def import TaskType, DataFormat
from data_utils.log_wrapper import create_logger
from experiments.exp_def import TaskDefs
from experiments.squad import squad_utils
from transformers import AutoTokenizer

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

class SinglePredictDataset(Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 is_train=False,
                 maxlen=512,
                 factor=1.0,
                 task_id=0,
                 task_def=None,
                 do_lower_case=True,
                 masked_lm_prob=0.15,
                 seed=13,
                 short_seq_prob=0.1,
                 max_seq_length=512,
                 max_predictions_per_seq=80,
                 printable=True):
        data = self.build_data(data=data, tokenizer=tokenizer, data_format=task_def.data_type, lab_dict=task_def.label_vocab)
        data, tokenizer = self.load(data, is_train, maxlen, factor, task_def, printable=printable)
        self._data = data
        self._tokenizer = tokenizer
        self._task_id = task_id
        self._task_def = task_def
        # init vocab words
        self._vocab_words = None if tokenizer is None else list(self._tokenizer.vocab.keys())
        self._masked_lm_prob = masked_lm_prob
        self._seed = seed
        self._short_seq_prob = short_seq_prob
        self._max_seq_length = max_seq_length
        self._max_predictions_per_seq = max_predictions_per_seq
        self._rng = random.Random(seed)
        self.maxlen = maxlen

    def get_task_id(self):
        return self._task_id

    def load(self, path, is_train=True, maxlen=512, factor=1.0, task_def=None, printable=True):
        task_type = task_def.task_type
        assert task_type is not None
        with open(path, 'r', encoding='utf-8') as reader:
            data = []
            cnt = 0
            for line in reader:
                sample = json.loads(line)
                sample['factor'] = factor
                cnt += 1
                if is_train:
                    task_obj = tasks.get_task_obj(task_def)
                    if task_obj is not None and not task_obj.input_is_valid_sample(sample, maxlen):
                        continue
                    if (task_type == TaskType.Ranking) and (len(sample['token_id'][0]) > maxlen or len(sample['token_id'][1]) > maxlen):
                        continue
                    if (task_type != TaskType.Ranking) and (len(sample['token_id']) > maxlen):
                        continue
                data.append(sample)
            if printable:
                print('Loaded {} samples out of {}'.format(len(data), cnt))
        return data, None

    def feature_extractor(self, tokenizer, text_a, text_b=None, max_length=512, do_padding=False):
        inputs = tokenizer(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding=do_padding
        )
        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"] if "token_type_ids" in inputs else [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = inputs["attention_mask"]
        if do_padding:
            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                                max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                                max_length)
        return input_ids, attention_mask, token_type_ids

    def build_data(self, data, tokenizer, data_format=DataFormat.PremiseOnly,
                   max_seq_len=512, lab_dict=None, do_padding=False, truncation=True):
        def build_data_premise_only(
                data, max_seq_len=512, tokenizer=None):
            """Build data of single sentence tasks
            """
            feature_datas = []
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                label = sample['label']
                input_ids, input_mask, type_ids = self.feature_extractor(tokenizer, premise, max_length=max_seq_len)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids,
                    'type_id': type_ids,
                    'attention_mask': input_mask}
                feature_datas.append(features)
            return feature_datas

        def build_data_premise_and_one_hypo(
                data, dump_path, max_seq_len=512, tokenizer=None):
            """Build data of sentence pair tasks
            """
            feature_datas = []
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                hypothesis = sample['hypothesis']
                label = sample['label']
                input_ids, input_mask, type_ids = self.feature_extractor(tokenizer, premise, text_b=hypothesis,
                                                                    max_length=max_seq_len)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids,
                    'type_id': type_ids,
                    'attention_mask': input_mask}
                feature_datas.append(features)
            return feature_datas

        def build_data_premise_and_multi_hypo(
                data, max_seq_len=512, tokenizer=None):
            """Build QNLI as a pair-wise ranking task
            """
            feature_datas = []
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                hypothesis_list = sample['hypothesis']
                label = sample['label']
                input_ids_list = []
                type_ids_list = []
                attention_mask_list = []
                for hypothesis in hypothesis_list:
                    input_ids, input_mask, type_ids = self.feature_extractor(tokenizer,
                                                                        premise, hypothesis, max_length=max_seq_len)
                    input_ids_list.append(input_ids)
                    type_ids_list.append(type_ids)
                    attention_mask_list.append(input_mask)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids_list,
                    'type_id': type_ids_list,
                    'ruid': sample['ruid'],
                    'olabel': sample['olabel'],
                    'attention_mask': attention_mask_list}
                feature_datas.append(features)
            return feature_datas

        def build_data_sequence(data, dump_path, max_seq_len=512, tokenizer=None, label_mapper=None):
            feature_datas = []
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                tokens = []
                labels = []
                for i, word in enumerate(premise):
                    subwords = tokenizer.tokenize(word)
                    tokens.extend(subwords)
                    for j in range(len(subwords)):
                        if j == 0:
                            labels.append(sample['label'][i])
                        else:
                            labels.append(label_mapper['X'])
                if len(premise) > max_seq_len - 2:
                    tokens = tokens[:max_seq_len - 2]
                    labels = labels[:max_seq_len - 2]

                label = [label_mapper['CLS']] + labels + [label_mapper['SEP']]
                input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
                assert len(label) == len(input_ids)
                type_ids = [0] * len(input_ids)
                features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
                feature_datas.append(features)
            return feature_datas

        def build_data_mrc(data, dump_path, max_seq_len=512, tokenizer=None, label_mapper=None, is_training=True):
            unique_id = 1000000000  # TODO: this is from BERT, needed to remove it...
            feature_datas = []
            for example_index, sample in enumerate(data):
                ids = sample['uid']
                doc = sample['premise']
                query = sample['hypothesis']
                label = sample['label']
                doc_tokens, cw_map = squad_utils.token_doc(doc)
                answer_start, answer_end, answer, is_impossible = squad_utils.parse_squad_label(label)
                answer_start_adjusted, answer_end_adjusted = squad_utils.recompute_span(answer, answer_start,
                                                                                        cw_map)
                is_valid = squad_utils.is_valid_answer(doc_tokens, answer_start_adjusted, answer_end_adjusted,
                                                       answer)
                if not is_valid: continue
                """
                TODO --xiaodl: support RoBERTa
                """
                feature_list = squad_utils.mrc_feature(tokenizer,
                                                       unique_id,
                                                       example_index,
                                                       query,
                                                       doc_tokens,
                                                       answer_start_adjusted,
                                                       answer_end_adjusted,
                                                       is_impossible,
                                                       max_seq_len,
                                                       512,
                                                       180,
                                                       answer_text=answer,
                                                       is_training=True)
                unique_id += len(feature_list)
                for feature in feature_list:
                    so = json.dumps({'uid': ids,
                                     'token_id': feature.input_ids,
                                     'mask': feature.input_mask,
                                     'type_id': feature.segment_ids,
                                     'example_index': feature.example_index,
                                     'doc_span_index': feature.doc_span_index,
                                     'tokens': feature.tokens,
                                     'token_to_orig_map': feature.token_to_orig_map,
                                     'token_is_max_context': feature.token_is_max_context,
                                     'start_position': feature.start_position,
                                     'end_position': feature.end_position,
                                     'label': feature.is_impossible,
                                     'doc': doc,
                                     'doc_offset': feature.doc_offset,
                                     'answer': [answer]})
                    feature_datas.append(so)
            return feature_datas

        if data_format == DataFormat.PremiseOnly:
            build_data_premise_only(
                data,
                max_seq_len,
                tokenizer)
        elif data_format == DataFormat.PremiseAndOneHypothesis:
            feature_data = build_data_premise_and_one_hypo(
                data, max_seq_len, tokenizer)
        elif data_format == DataFormat.PremiseAndMultiHypothesis:
            feature_data = build_data_premise_and_multi_hypo(
                data, max_seq_len, tokenizer)
        elif data_format == DataFormat.Seqence:
            feature_data = build_data_sequence(data, max_seq_len, tokenizer, lab_dict)
        elif data_format == DataFormat.MRC:
            feature_data = build_data_mrc(data, max_seq_len, tokenizer)
        else:
            raise ValueError(data_format)
        return feature_data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self._task_def.task_type == TaskType.MaskLM:
            # create a MLM instance
            instances = create_instances_from_document(self._data,
                                                       idx,
                                                       self._max_seq_length,
                                                       self._short_seq_prob,
                                                       self._masked_lm_prob,
                                                       self._max_predictions_per_seq,
                                                       self._vocab_words,
                                                       self._rng)
            instance_ids = list(range(0, len(instances)))
            choice = np.random.choice(instance_ids, 1)[0]
            instance = instances[choice]
            labels = self._tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
            position = instance.masked_lm_positions
            labels = [lab if idx in position else -1 for idx, lab in enumerate(labels)]
            sample = {'token_id': self._tokenizer.convert_tokens_to_ids(instance.tokens),
                      'type_id': instance.segment_ids,
                      'nsp_lab': 1 if instance.is_random_next else 0,
                      'position': instance.masked_lm_positions,
                      'label': labels,
                      'uid': idx}
            return {"task": {"task_id": self._task_id, "task_def": self._task_def},
                    "sample": sample}
        else:
            return {"task": {"task_id": self._task_id, "task_def": self._task_def},
                    "sample": self._data[idx]}

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
        self.tokenize_model = 'bert-base-chinese'
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
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenize_model)
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

    def predict(self, task_name, data):
        assert task_name in self.task_names, "指定的task不在我们的预设task内，所以不支持这个task"
        test_data_set = SinglePredictDataset(data, tokenizer=self.tokenizer, maxlen=self.max_seq_len, task_id=self.tasks_info[task_name]['task_id'], task_def=self.tasks_info[task_name]['task_def'])
        test_data = DataLoader(test_data_set, batch_size=self.predict_batch_size, collate_fn=self.collater.collate_fn,pin_memory=self.cuda)
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
    results = model.predict(task_name='absa', data=test_data)
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
    results = model.predict(task_name='dem8', data=test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

if __name__ == "__main__":
    model = TorchMTDNNModel()
    app.run(host='0.0.0.0', port=5018, debug=True, threaded=True)