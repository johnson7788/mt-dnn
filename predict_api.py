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
import json
import os
import re
import torch
from experiments.exp_def import TaskDefs, EncoderModelType
from torch.utils.data import Dataset, DataLoader, BatchSampler
from mt_dnn.batcher import Collater
from experiments.mlm.mlm_utils import create_instances_from_document
from mt_dnn.model import MTDNNModel
import random
from data_utils.metrics import calc_metrics
import numpy as np
from data_utils.task_def import TaskType, DataFormat
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
                 max_predictions_per_seq=80
                 ):
        data = self.build_data(data=data, tokenizer=tokenizer, data_format=task_def.data_type, lab_dict=task_def.label_vocab)
        data = self.add_factor(data)
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

    def add_factor(self, data, factor=1.0):
        new_data = []
        for sample in data:
            sample['factor'] = factor
            new_data.append(sample)
        return new_data

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
                data, max_seq_len=512, tokenizer=None):
            """Build data of sentence pair tasks
            """
            feature_datas = []
            for idx, sample in enumerate(data):
                premise = sample[0]
                hypothesis = sample[1]
                label = 0
                input_ids, input_mask, type_ids = self.feature_extractor(tokenizer, premise, text_b=hypothesis,
                                                                    max_length=max_seq_len)
                features = {
                    'uid': idx,
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
                    'uid': idx,
                    'label': label,
                    'token_id': input_ids_list,
                    'type_id': type_ids_list,
                    'ruid': sample['ruid'],
                    'olabel': sample['olabel'],
                    'attention_mask': attention_mask_list}
                feature_datas.append(features)
            return feature_datas

        def build_data_sequence(data, max_seq_len=512, tokenizer=None, label_mapper=None):
            feature_datas = []
            for idx, sample in enumerate(data):
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
                features = {'uid': idx, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
                feature_datas.append(features)
            return feature_datas

        def build_data_mrc(data, max_seq_len=512, tokenizer=None, label_mapper=None, is_training=True):
            unique_id = 1000000000  # TODO: this is from BERT, needed to remove it...
            feature_datas = []
            for example_index, sample in enumerate(data):
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
                for f_idx, feature in enumerate(feature_list):
                    so = json.dumps({'uid': f"{example_index}_{f_idx}",
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
            feature_datas = build_data_premise_only(
                data,
                max_seq_len,
                tokenizer)
        elif data_format == DataFormat.PremiseAndOneHypothesis:
            feature_datas = build_data_premise_and_one_hypo(
                data, max_seq_len, tokenizer)
        elif data_format == DataFormat.PremiseAndMultiHypothesis:
            feature_datas = build_data_premise_and_multi_hypo(
                data, max_seq_len, tokenizer)
        elif data_format == DataFormat.Seqence:
            feature_datas = build_data_sequence(data, max_seq_len, tokenizer, lab_dict)
        elif data_format == DataFormat.MRC:
            feature_datas = build_data_mrc(data, max_seq_len, tokenizer)
        else:
            raise ValueError(data_format)
        return feature_datas

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
        # absa 情感分析， dem8是8个维度的判断, purchase 购买意向
        self.task_names = ['absa', 'dem8', 'purchase']
        # 保存每个task需要的一些必要的信息
        self.tasks_info = {}
        # 最大序列长度
        self.max_seq_len = 512
        # 最大的batch_size
        self.predict_batch_size = 64
        self.tokenize_model = 'bert-base-chinese'
        # 训练好的模型的存放位置
        self.checkpoint = 'trained_model/absa_dem8.pt'
        self.type_to_prefix = {
            "成分": '成分:',
            "功效": '功效:',
            "香味":"香味:",
            "包装": "包装:",
            "肤感": "肤感:",
            "促销":"促销:",
            "服务":"服务:",
            "价格":"价格:",
            "component": '成分:',
            "effect": '功效:',
            "fragrance": "香味:",
            "pack": "包装:",
            "skin": "肤感:",
            "promotion": "促销:",
            "service": "服务:",
            "price": "价格:",
        }
        self.left_max_seq_len = 60
        self.aspect_max_seq_len = 30
        self.right_max_seq_len = 60
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
                "metric_meta": metric_meta,
                "id2tok": task_def.label_vocab.ind2tok,
            }
    def truncate(self,input_text, max_len, trun_post='post'):
        """
        实施截断数据
        :param input_text:
        :param max_len:   eg: 15
        :param trun_post: 截取方向，向前还是向后截取，
                        "pre"：截取前面的， "post"：截取后面的
        :return:
        """
        if max_len is not None and len(input_text) > max_len:
            if trun_post == "post":
                return input_text[-max_len:]
            else:
                return input_text[:max_len]
        else:
            return input_text
    def aspect_truncate(self, content, aspect, aspect_start, aspect_end, left_max_seq_len, aspect_max_seq_len, right_max_seq_len):
        """
        截断函数
        :param content:
        :param aspect:
        :param aspect_start:
        :param aspect_end:
        :return:
        """
        text_left = content[:aspect_start]
        text_right = content[aspect_end:]
        text_left = self.truncate(text_left, left_max_seq_len)
        aspect = self.truncate(aspect, aspect_max_seq_len)
        text_right = self.truncate(text_right, right_max_seq_len, trun_post="pre")
        new_content = text_left + aspect + text_right
        return new_content
    def aspect_base_truncate(self, data, prefix_data=None, search_first=True):
        """aspect的类型的任务的truncate
        对数据做truncate
        :param data:针对不同类型的数据进行不同的截断
        :param search_first: 是否只搜索第一个匹配的关键字
        :return:返回列表，是截断后的文本，aspect
        所以如果一个句子中有多个aspect关键字，那么就会产生多个截断的文本+关键字，组成的列表，会产生多个预测结果
        """
        contents = []
        #保存关键字的索引，[(start_idx, end_idx)...]
        locations = []
        # 搜索到的关键字的数量
        keywords_index = [0] * len(data)
        for idx, one_data in enumerate(data):
            if isinstance(one_data,str):
                # 句子级的情感，没有aspect
                contents.append((one_data, ""))
                locations.append((0, 0))
            elif len(one_data) == 2 or len(one_data) == 3:
                #不带aspect关键字的位置信息，自己查找位置
                content, aspect = one_data[0], one_data[1]
                iter = re.finditer(aspect, content)
                for m in iter:
                    aspect_start, aspect_end = m.span()
                    new_content = self.aspect_truncate(content, aspect, aspect_start, aspect_end, left_max_seq_len=self.left_max_seq_len, aspect_max_seq_len=self.aspect_max_seq_len, right_max_seq_len=self.right_max_seq_len)
                    if prefix_data:
                        prefix = prefix_data[idx]
                        new_content = prefix + new_content
                    contents.append((new_content, aspect))
                    locations.append((aspect_start,aspect_end))
                    if search_first:
                        #只取第一个关键字的数据
                        break
                    else:
                        keywords_index[idx] += 1
            elif len(one_data) == 4:
                # 不带label时，长度是4，
                content, aspect, aspect_start, aspect_end = one_data
                new_content = self.aspect_truncate(content, aspect, aspect_start,aspect_end, left_max_seq_len=self.left_max_seq_len, aspect_max_seq_len=self.aspect_max_seq_len, right_max_seq_len=self.right_max_seq_len)
                if prefix_data:
                    prefix = prefix_data[idx]
                    new_content = prefix + new_content
                contents.append((new_content, aspect))
                locations.append((aspect_start, aspect_end))
            elif len(one_data) == 5:
                content, aspect, attr_type, aspect_start, aspect_end = one_data
                new_content = self.aspect_truncate(content, aspect, aspect_start, aspect_end, left_max_seq_len=self.left_max_seq_len, aspect_max_seq_len=self.aspect_max_seq_len, right_max_seq_len=self.right_max_seq_len)
                if prefix_data:
                    prefix = prefix_data[idx]
                    new_content = prefix + new_content
                contents.append((new_content, aspect))
                locations.append((aspect_start, aspect_end))
            else:
                raise Exception(f"这条数据异常: {one_data},数据长度或者为1, 2, 4，或者为5")
        if search_first:
            return contents, locations
        else:
            return contents, locations, keywords_index
    def purchase_text_truncate(self, data, search_first=True):
        """购买意向的数据进行截断处理
        对数据做truncate
        :param data:针对不同类型的数据进行不同的截断
        :param search_first: 是否只搜索第一个匹配的关键字
        :return:返回列表，是截断后的文本，aspect
        所以如果一个句子中有多个aspect关键字，那么就会产生多个截断的文本+关键字，组成的列表，会产生多个预测结果
        """
        contents = []
        #保存关键字的索引，[(start_idx, end_idx)...]
        locations = []
        # 搜索到的关键字的数量
        keywords_index = [0] * len(data)
        for idx, one_data in enumerate(data):
            if len(one_data) == 3:
                #不带aspect关键字的位置信息，自己查找位置
                content, title, aspect = one_data[0], one_data[1], one_data[2]
                title_content = title + content
                title_content = title_content.lower()
                iter = re.finditer(aspect, title_content)
                for m in iter:
                    aspect_start, aspect_end = m.span()
                    new_content = self.aspect_truncate(title_content, aspect, aspect_start, aspect_end, left_max_seq_len=self.left_max_seq_len, aspect_max_seq_len=self.aspect_max_seq_len, right_max_seq_len=self.right_max_seq_len)
                    contents.append((new_content, aspect))
                    locations.append((aspect_start,aspect_end))
                    if search_first:
                        #只取第一个关键字的数据
                        break
                    else:
                        keywords_index[idx] += 1
            elif len(one_data) == 5:
                content, title, aspect, aspect_start, aspect_end = one_data
                # 拼接title的内容
                title_content = title + content
                title_content = title_content.lower()
                aspect_start = aspect_start + len(title)
                aspect_end = aspect_end + len(title)
                new_content = self.aspect_truncate(title_content, aspect, aspect_start, aspect_end, left_max_seq_len=self.left_max_seq_len, aspect_max_seq_len=self.aspect_max_seq_len, right_max_seq_len=self.right_max_seq_len)
                contents.append((new_content, aspect))
                locations.append((aspect_start, aspect_end))
            else:
                raise Exception(f"这条数据异常: {one_data},数据长度或者为1, 2, 4，或者为5")
        if search_first:
            return contents, locations
        else:
            return contents, locations, keywords_index
    def dem8_truncate(self, data, prefix_data):
        """
        多个关键字的aspect的trunacate方法
        :param data:针对不同类型的数据进行不同的截断
        :param search_first: 是否只搜索第一个匹配的关键字
        :return:返回列表，是截断后的文本，aspect
        所以如果一个句子中有多个aspect关键字，那么就会产生多个截断的文本+关键字，组成的列表，会产生多个预测结果
        """
        contents = []
        #保存关键字的索引，[(start_idx, end_idx)...]
        locations = []
        # 每个句子的搜索到的关键字的数量
        aspects_index = {}
        for idx, one_data in enumerate(data):
            #不带aspect关键字的位置信息，自己查找位置,aspects 是一个列表，是多个关键字
            content, aspects = one_data[0], one_data[1]
            aspects_index[idx] = len(aspects) * [0]
            for aidx, aspect in enumerate(aspects):
                iter = re.finditer(aspect, content)
                for m in iter:
                    aspect_start, aspect_end = m.span()
                    new_content = self.aspect_truncate(content, aspect, aspect_start, aspect_end)
                    if prefix_data:
                        prefix = prefix_data[idx]
                        new_content = prefix + new_content
                    contents.append((new_content, aspect))
                    locations.append((aspect_start,aspect_end))
                    aspects_index[idx][aidx] += 1
        return contents, locations, aspects_index
    def get_dem8_prefix(self, data):
        """
        获取8个维度的分类任务的前缀, 每个data的类型可能不同，那么前缀也可能不同
        :param data:
        :type data:
        :return: prefix的data组成的列表
        :rtype: list
        """
        prefix_data = []
        for d in data:
            type_name = d[2]
            prefix_name = self.type_to_prefix[type_name]
            prefix_data.append(prefix_name)
        return prefix_data
    def predict_batch(self, task_name, data, with_label=False, aspect_base=True, full_score=False):
        """
        预测数据
        :param task_name:
        :type task_name:
        :param data:
        :type data:
        :param with_label:  如果数据带了标签，那么打印metric
        :type with_label:
        :param aspect_base:  是否是aspect_base的任务, 返回aspect的位置
        :type aspect_base:
        :param full_score: bool, 返回的score是按照最大概率返回，还是返回一个列表，返回预测的结果的那个所有score
        :return:
        :rtype:
        """
        assert task_name in self.task_names, "指定的task不在我们的预设task内，所以不支持这个task"
        if aspect_base:
            if task_name == 'absa':
                truncate_data, locations = self.aspect_base_truncate(data)
            elif task_name == 'dem8':
                # dem8和purchase都有prefix，
                prefix_data = self.get_dem8_prefix(data)
                truncate_data, locations = self.aspect_base_truncate(data,prefix_data=prefix_data)
            else:
                # purchase是把title作为prefix
                truncate_data, locations = self.purchase_text_truncate(data)
            test_data_set = SinglePredictDataset(truncate_data, tokenizer=self.tokenizer, maxlen=self.max_seq_len, task_id=self.tasks_info[task_name]['task_id'], task_def=self.tasks_info[task_name]['task_def'])
        else:
            test_data_set = SinglePredictDataset(data, tokenizer=self.tokenizer, maxlen=self.max_seq_len, task_id=self.tasks_info[task_name]['task_id'], task_def=self.tasks_info[task_name]['task_def'])
        test_data = DataLoader(test_data_set, batch_size=self.predict_batch_size, collate_fn=self.collater.collate_fn,pin_memory=self.cuda)
        with torch.no_grad():
            # test_metrics eg: acc结果，准确率结果
            # test_predictions: 预测的结果， scores预测的置信度， golds是我们标注的结果，标注的label， test_ids样本id, 打印metrics
            predictions = []
            golds = []
            scores = []
            ids = []
            for (batch_info, batch_data) in test_data:
                batch_info, batch_data = Collater.patch_data(self.device, batch_info, batch_data)
                score, pred, gold = self.model.predict(batch_info, batch_data, full_score)
                predictions.extend(pred)
                golds.extend(gold)
                scores.extend(score)
                ids.extend(batch_info['uids'])
            if with_label:
                metrics = calc_metrics(self.tasks_info[task_name]['metric_meta'], golds, predictions, scores, label_mapper=None)
            id2tok = self.tasks_info[task_name]['id2tok']
            predict_labels = [id2tok[p] for p in predictions]
            print(f"预测结果{predictions}, 预测的标签是 {predict_labels}")
        if aspect_base:
            results = list(zip(predict_labels, scores, data, locations))
        else:
            results = list(zip(predict_labels, scores, data))
        return results
    def predict_dem8(self, data):
        for idx, one_data in enumerate(data):
            if len(one_data) != 3:
                return f"错误: 第{idx}条数据的结构不对，结构必须是[[句子，aspect关键字，类型]，...]"
            word_type = one_data[2]
            if not self.type_to_prefix.get(word_type):
                return f"错误: 第{idx}条数据给了不支持的判断的单词类型,{word_type}"
        prefix_data = self.get_dem8_prefix(data)
        truncate_data, locations, aspects_index = self.dem8_truncate(data, prefix_data=prefix_data)
        test_data_set = SinglePredictDataset(truncate_data, tokenizer=self.tokenizer, maxlen=self.max_seq_len, task_id=self.tasks_info['dem8']['task_id'], task_def=self.tasks_info['dem8']['task_def'])
        test_data = DataLoader(test_data_set, batch_size=self.predict_batch_size, collate_fn=self.collater.collate_fn,pin_memory=self.cuda)
        with torch.no_grad():
            # test_metrics eg: acc结果，准确率结果
            # test_predictions: 预测的结果， scores预测的置信度， golds是我们标注的结果，标注的label， test_ids样本id, 打印metrics
            predictions = []
            golds = []
            scores = []
            for (batch_info, batch_data) in test_data:
                batch_info, batch_data = Collater.patch_data(self.device, batch_info, batch_data)
                score, pred, gold = self.model.predict(batch_info, batch_data)
                predictions.extend(pred)
                golds.extend(gold)
                scores.extend(score)
            id2tok = self.tasks_info['dem8']['id2tok']
            predict_labels = [id2tok[p] for p in predictions]
            print(f"预测结果{predictions}, 预测的标签是 {predict_labels}")
        result = []
        #开始索引的位置
        start_idx = 0
        for sentence_idx, key_nums in aspects_index.items():
            data_one = []
            for k_idx, key_num in enumerate(key_nums):
                end_idx = start_idx + key_num
                keyword_data = {
                    'keyword': data[sentence_idx][1][k_idx],
                    'type': data[sentence_idx][2],
                    'locations': locations[start_idx:end_idx],
                    'labels': predict_labels[start_idx:end_idx],
                }
                start_idx = end_idx
                data_one.append(keyword_data)
            result.append(data_one)
        return result

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
    results = model.predict_batch(task_name='absa', data=test_data)
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
    results = model.predict_batch(task_name='dem8', data=test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/purchase_predict", methods=['POST'])
def purchase_predict():
    """
    购买意向分类
    例如关键字前后的25个字作为sentenceA，aspect关键字作为sentenceB，输入模型
    Args:
        接受数据是 [(content,title, aspect),...,]
        或者：
        test_data: 需要预测的数据，是一个文字列表, [(content,title, aspect,start_idx, end_idx),...,]
        如果传过来的数据没有索引，那么需要自己去查找索引 [(content,aspect),...,]
    Returns: 返回格式是 [(predicted_label, predict_score),...]
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    logger.info(f"要进行的任务是购买意向判断")
    results = model.predict_batch(task_name='purchase', data=test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/dem8", methods=['POST'])
def dem8():
    """
    8个维度的预测，接收POST请求，获取data参数, data信息包含aspect关键在在句子中的位置信息，方便我们截取，我们截取aspect关键字的前后一定的字符作为输入
    例如关键字前后的25个字作为sentenceA，aspect关键字作为sentenceB，输入模型
    Args:
        test_data: 需要预测的数据，是一个文字列表, [['使用感受：补水效果好 皮肤一种舒服的状态适合肤质：油性干性的都适合补水，熬夜的效果修复更好吸收效果：吸收快 刚好合适保湿效果：持久不油腻其他特色：包装精致性价比高 品牌值得信赖一直用一叶子的产品喜欢 点赞',['保湿', '补水', '熬夜'], '成分'], ['产品质感：质感非常好 质地很清爽 不拔干 适合肤质：我是敏感肌 用了都没什么问题的 应该是都可以用的卸妆力度：很好用的 这都数不清是第几瓶了 无线回购的好产品 洁净效果：卸妆的同时 还可以补水噢 容量大小：超级大的一瓶 可以用好久其他特色：物美价廉的产品 真的是很好用',['质感', '质地', '补水', '卸妆'],'功效']]
    Returns: 返回格式是[[{'keyword': '保湿','type':'成分',locations: [(5,7), (30,32)], labels:['是','否']},{'keyword':'补水',... }],[...]]
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    results = model.predict_dem8(data=test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

if __name__ == "__main__":
    model = TorchMTDNNModel()
    app.run(host='0.0.0.0', port=3326, debug=False, threaded=False)