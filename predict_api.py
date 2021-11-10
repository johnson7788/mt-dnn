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
import pandas as pd
import torch
from experiments.exp_def import TaskDefs, EncoderModelType
from torch.utils.data import Dataset, DataLoader, BatchSampler
from mt_dnn.batcher import Collater
from experiments.mlm.mlm_utils import create_instances_from_document
from mt_dnn.model import MTDNNModel
import random
import collections
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
        data = self.build_data(data=data, tokenizer=tokenizer, data_format=task_def.data_type, lab_dict=task_def.label_vocab, max_seq_len=max_seq_length)
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

    def build_data_premise_only(self,
            data, max_seq_len=512, tokenizer=None):
        """Build data of single sentence tasks
        """
        feature_datas = []
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = sample['premise']
            if tokenizer.do_lower_case:
                premise = premise.lower()
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

    def build_data_premise_and_one_hypo(self,
            data, max_seq_len=512, tokenizer=None):
        """Build data of sentence pair tasks
        """
        feature_datas = []
        for idx, sample in enumerate(data):
            premise = sample[0]
            hypothesis = sample[1]
            if tokenizer.do_lower_case:
                premise = premise.lower()
                hypothesis = hypothesis.lower()
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

    def build_data_premise_and_multi_hypo(self,
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

    def build_data_sequence(self,data, max_seq_len=512, tokenizer=None, label_mapper=None):
        feature_datas = []
        for idx, sample in enumerate(data):
            premise = sample['premise']
            if tokenizer.do_lower_case:
                premise = premise.lower()
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

    def build_data_mrc(self,data, max_seq_len=512, tokenizer=None, label_mapper=None, is_training=True):
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

    def build_data_relation(self,data, tokenizer, max_seq_len=512):
        """
        创建关系判断的数据集
        :param data:
        :type data:
        :param dump_path: 导出路径
        :type dump_path:
        :param max_seq_len: 最大序列长度
        :type max_seq_len:
        :param tokenizer:
        :type tokenizer:
        :return:
        :rtype:
        """
        feature_datas = []
        for idx, item in enumerate(data):
            # 一条包含text，头部实体，尾部实体，和实体位置的json的字符床
            input_ids, input_mask, type_ids = self.relation_feature_extractor(tokenizer, item,
                                                                              max_length=max_seq_len,
                                                                              do_padding=False)
            features = {
                'uid': idx,
                'label': 0,   # 假的label
                'token_id': input_ids,
                'type_id': type_ids,
                'attention_mask': input_mask}
            feature_datas.append(features)
        return feature_datas
    def build_data(self, data, tokenizer, data_format=DataFormat.PremiseOnly,
                   max_seq_len=512, lab_dict=None, do_padding=False, truncation=True):
        if data_format == DataFormat.PremiseOnly:
            feature_datas = self.build_data_premise_only(
                data,
                max_seq_len,
                tokenizer)
        elif data_format == DataFormat.PremiseAndOneHypothesis:
            feature_datas = self.build_data_premise_and_one_hypo(
                data, max_seq_len, tokenizer)
        elif data_format == DataFormat.PremiseAndMultiHypothesis:
            feature_datas = self.build_data_premise_and_multi_hypo(
                data, max_seq_len, tokenizer)
        elif data_format == DataFormat.Sequence:
            feature_datas = self.build_data_sequence(data, max_seq_len, tokenizer, lab_dict)
        elif data_format == DataFormat.MRC:
            feature_datas = self.build_data_mrc(data, max_seq_len, tokenizer)
        elif data_format == DataFormat.RELATION:
            feature_datas = self.build_data_relation(data, tokenizer)
        else:
            raise ValueError(data_format)
        return feature_datas

    def relation_feature_extractor(self, tokenizer, item, max_length=512, do_padding=False):
        """
        关系判断的数据tokenize
        :return:
        :rtype:
        """
        # Sentence -> token
        sentence = item['text']
        pos_head = item['brand']['pos']
        pos_tail = item['attribute']['pos']
        if tokenizer.do_lower_case:
            sentence = sentence.lower()

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False

        sent0 = tokenizer.tokenize(sentence[:pos_min[0]])
        ent0 = tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
        sent1 = tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
        ent1 = tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
        sent2 = tokenizer.tokenize(sentence[pos_max[1]:])
        ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
        ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']
        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        indexed_tokens = tokenizer.convert_tokens_to_ids(re_tokens)
        assert len(indexed_tokens) <= 512, "注意，长度过大，大于了最大长度512，请检查数据"
        avai_len = len(indexed_tokens)
        # Padding
        if do_padding:
            while len(indexed_tokens) < max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:max_length]
        # Attention mask
        att_mask = [0] * len(indexed_tokens)
        att_mask[:avai_len] = [1] * avai_len
        token_type_ids = [0] * len(indexed_tokens)
        return indexed_tokens, att_mask, token_type_ids
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
        # absa 情感分析， dem8是8个维度的判断, purchase 购买意向, brand品牌功效关系判断
        self.task_names = ['absa', 'dem8', 'purchase','brand','wholesentiment', 'pinpainer']
        # 保存每个task需要的一些必要的信息
        self.tasks_info = {}
        # 最大序列长度
        self.max_seq_len = 500
        # 最大的batch_size
        self.predict_batch_size = 64
        self.tokenize_model = 'bert-base-chinese'
        self.do_lower_case = False
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
            # 有的任务没训练，就没在train_datasets中
            if task_name in train_datasets:
                task_id = train_datasets.index(task_name)
                self.tasks_info[task_name]['task_id'] = task_id
        self.config["cuda"] = self.cuda
        ## temp fix
        self.config['fp16'] = False
        self.config['answer_opt'] = 0
        self.config['adv_train'] = False
        del self.state_dict['optimizer']
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenize_model, do_lower_case=self.do_lower_case)
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
                print(one_data)
                iter = re.finditer(re.escape(aspect), content)
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
                else:
                    # 没有搜到任何关键字，那么打印注意信息
                    print(f"在content： {content}中，未搜到aspect:{aspect}, 返回一个00默认值")
                    contents.append((content, aspect))
                    locations.append((0, 0))
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
                if pd.isna(content):
                    content = "Empty"
                if pd.isna(title):
                    title_content = content
                else:
                    title_content = str(title) + str(content)
                aspect = aspect.lower()
                title_content = title_content.lower()
                iter = re.finditer(re.escape(aspect), title_content)
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
                else:
                    print(f"注意，未通过关键词匹配到数据{one_data}")
                    new_content = title_content[:100]
                    contents.append((new_content, aspect))
                    locations.append((0, 0))
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
                    new_content = self.aspect_truncate(content, aspect, aspect_start, aspect_end, left_max_seq_len=self.left_max_seq_len, aspect_max_seq_len=self.aspect_max_seq_len, right_max_seq_len=self.right_max_seq_len)
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
    def predict_batch(self, task_name, data, with_label=False, aspect_base=True, full_score=False, search_first=True, softmax=True, both_softmax_logits=False):
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
        :param search_first: bool, 对于没有给定关键字位置的句子，自动搜索到的第一个关键字作为结果，不继续搜索, 为了收到的数据和返回的数据的条目相同
        :return:
        :rtype:
        """
        assert task_name in self.task_names, "指定的task不在我们的预设task内，所以不支持这个task"
        # 如果给的数据没有位置信息，并且要搜索所有的keywords，那么keywords_index应该是有值的
        keywords_index = None
        if aspect_base:
            if task_name == 'absa':
                if search_first:
                    truncate_data, locations = self.aspect_base_truncate(data,search_first=True)
                else:
                    truncate_data, locations, keywords_index = self.aspect_base_truncate(data,search_first=False)
            elif task_name == 'dem8':
                # dem8和purchase都有prefix，
                prefix_data = self.get_dem8_prefix(data)
                if search_first:
                    truncate_data, locations = self.aspect_base_truncate(data,prefix_data=prefix_data,search_first=search_first)
                else:
                    truncate_data, locations, keywords_index = self.aspect_base_truncate(data,search_first=False)
            else:
                # purchase是把title作为prefix
                truncate_data, locations = self.purchase_text_truncate(data)
                assert len(truncate_data) == len(data), "数据查找关键字后的条数变得不匹配，请校对"
            test_data_set = SinglePredictDataset(truncate_data, tokenizer=self.tokenizer, max_seq_length=self.max_seq_len, task_id=self.tasks_info[task_name]['task_id'], task_def=self.tasks_info[task_name]['task_def'])
        else:
            test_data_set = SinglePredictDataset(data, tokenizer=self.tokenizer, max_seq_length=self.max_seq_len, task_id=self.tasks_info[task_name]['task_id'], task_def=self.tasks_info[task_name]['task_def'])
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
                score, pred, gold = self.model.predict(batch_info, batch_data, full_score, softmax, both_softmax_logits)
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
            if keywords_index:
                # 如果搜索了多个关键字，那么结果都是嵌套列表的形式返回
                results = []
                tmp_res = list(zip(predict_labels, scores,locations))
                for idx, key_counts in enumerate(keywords_index):
                    one_result = []
                    #源数据的data的索引
                    src_one_data = data[idx]
                    for key_count in range(key_counts):
                        keyword_result = tmp_res.pop(0)
                        keyword_result = list(keyword_result)
                        # 把源数据插入进去
                        keyword_result.insert(2,src_one_data)
                        one_result.append(keyword_result)
                    results.append(one_result)
            else:
                results = list(zip(predict_labels, scores, data, locations))
        else:
            results = list(zip(predict_labels, scores, data))
        return results
    def predict_dem8_kn_s1(self, data):
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
    def predict_absa_dem8_k1_s1(self,data, full_score=False, softmax=True, both_softmax_logits=False):
        """
        预测属性之后预测情感, 只给一个keyword，只搜索一个keyword
        :param data:
        :type data:
        :param both_softmax_logits:  返回的结果 [logits, score], predict
        :return:
        :rtype:
        """
        dem_result = self.predict_batch(task_name='dem8', data=data, search_first=True)
        #用于预测的数据收集
        as_data = []
        #用于索引是不预测了情感
        as_index = []
        #预测为不是某个属性的数据
        demnot_data = []
        for dr in dem_result:
            # 一条情感的数据
            dem_label, _, src_data, locations = dr
            content, keyword, attr_type = src_data
            start, end = locations
            #是和否，加到新的列表，用于预测后返回
            as_index.append(dem_label)
            if dem_label == "是":
                as_d = [content, keyword, start, end]
                as_data.append(as_d)
            else:
                demnot_data.append([content,keyword])
        # 汇总预测结果
        as_result = []
        if as_data:
            # 如果有需要预测情感的数据，那么预测情感的结果
            as_predict = self.predict_batch(task_name='absa', data=as_data,full_score=full_score,softmax=softmax,both_softmax_logits=both_softmax_logits)
            for as_idx in as_index:
                if as_idx == "是":
                    one_predict = as_predict.pop(0)
                    label, score, src_data, locations = one_predict
                    # label, score, content, keyword,locations
                    predict = [label,score, src_data[0], src_data[1], locations]
                else:
                    one_src = demnot_data.pop(0)
                    content, keyword = one_src
                    # label, score, content, keyword,locations
                    predict = [0, 0, content, keyword, 0]
                as_result.append(predict)
        return as_result
    def predict_absa_dem8_k1_sn(self,data, full_score=False):
        """
        预测属性之后预测情感, 只给一个keyword，搜索所有keyword，对所有keyword进行属性和情感判断
        :param data:
        :type data:
        :return:
        :rtype:
        """
        dem_result = self.predict_batch(task_name='dem8', data=data, search_first=False)
        #用于预测的数据收集
        as_data = []
        #用于索引是不预测了情感
        as_index = []
        #预测为不是某个属性的数据
        demnot_data = []
        # keyword_count, 每个content的keyword的个数
        keyword_counts = [len(dr) for dr in dem_result]
        for dr in dem_result:
            for mr in dr:
                # 一条情感的数据
                dem_label, _, src_data, locations = mr
                content, keyword, attr_type = src_data
                start, end = locations
                #是和否，加到新的列表，用于预测后返回
                as_index.append(dem_label)
                if dem_label == "是":
                    as_d = [content, keyword, start, end]
                    as_data.append(as_d)
                else:
                    demnot_data.append([content,keyword])
        # 汇总预测结果
        as_result = []
        if as_data:
            # 如果有需要预测情感的数据，那么预测情感的结果
            as_predict = self.predict_batch(task_name='absa', data=as_data, full_score=full_score)
            for as_idx in as_index:
                if as_idx == "是":
                    one_predict = as_predict.pop(0)
                    label, score, src_data, locations = one_predict
                    # label, score, content, keyword,locations
                    predict = [label, score, src_data[0], src_data[1], locations]
                else:
                    one_src = demnot_data.pop(0)
                    content, keyword = one_src
                    # label, score, content, keyword,locations
                    predict = [0, 0, content, keyword, 0]
                as_result.append(predict)
        #返回个数与data源数据相同
        full_result = []
        if as_result:
            for key_idx, key_cnt in enumerate(keyword_counts):
                one_result = []
                for cnt in range(key_cnt):
                    key_result = as_result.pop(0)
                    one_result.append(key_result)
                full_result.append(one_result)
        final_result = []
        if full_result:
            # 如果整个句子只有一个关键字，那么保留最终结果，即使没有情感，
            # 如果有多个关键字，有的有情感，有的没有情感，那么只保留有情感的
            # 如果多个关键字都没有情感， 那么保留最后一个就可以了
            for res in full_result:
                if len(res) == 1:
                    # 如果唯一，那么加上结果
                    final_result.append(res)
                else:
                    # 判断关键字，这里关键字都是相同的，只需判断label是否为0，去掉label为0的
                    # 标签为0的那个结果
                    empty_res = None
                    # 将加入到结果中
                    added_res = []
                    for every_res in res:
                        if every_res[0] == 0:
                            empty_res = every_res
                        else:
                            added_res.append(every_res)
                    # 如果最终added_res是有内容的，直接加入到结果中
                    if added_res:
                        final_result.append(added_res)
                    else:
                        # 如果没有内容，就把最后一个空的结果加入到结果中
                        added_res.append(empty_res)
                        final_result.append(added_res)
        return final_result
    def truncate_relation(self, data, max_seq_len=150):
        """
        只对text的长度进行截取，根据
        :param data: 源数据
        :type data:
        :param max_seq_len: 最大序列长度
        :type max_seq_len:
        :return:
        :rtype:
        """
        # 把最大长度减去20，作为实体词的长度的备用
        max_length = max_seq_len - 20
        truncate_data = []
        length_counter = collections.Counter()
        for one in data:
            text = one['text']
            if len(text) > max_seq_len:
                length_counter['超过最大长度'] += 1
                # 开始截断
                h_entity = one['brand']['name']
                t_entity = one['attribute']['name']
                h_length = len(one['brand']['name'])
                t_length = len(one['attribute']['name'])
                h_start = one['brand']['pos'][0]
                h_end = one['brand']['pos'][1]
                t_start = one['attribute']['pos'][0]
                t_end = one['attribute']['pos'][1]
                # 先判断2个实体词之间的距离是否大于max_seq_len,如果大于，那么就2个实体词的2层分别保留一段位置，否则就从2个实体词的2层剪断
                if h_start < t_start:
                    # 实体词h在前，t在后
                    if t_end - h_start > max_length:
                        # 实体词的2册都进行截取,  形式是,被截断的示例是: xx|xxx entity1 xxx|xx   +  xxx|xx entity2 xxxx|x, 其中|表示被截断的标记
                        half_length = max_length / 2
                        # 第一个实体前后的句子开始和结束位置
                        l1_start = h_start - int(half_length / 2)
                        if l1_start < 0:
                            l1_start = 0
                        l1_end = h_end + int(half_length / 2)
                        l2_start = t_start - int(half_length / 2)
                        l2_end = t_end + int(half_length / 2)
                        newtext = text[l1_start:l1_end] + text[l2_start:l2_end]
                        h_start = h_start - l1_start
                        h_end = h_start + h_length
                        # 第二个实体位置新的开始
                        t_start = t_start - l1_start - (l2_start - l1_end)
                        t_end = t_start + t_length
                        assert newtext[h_start:h_end] == h_entity, "截断后的实体位置信息不对"
                        assert newtext[t_start:t_end] == t_entity, "截断后的实体位置信息不对"
                        # assert len(newtext) <= max_seq_len, f"最大长度截断后过长{len(newtext)}"
                    else:
                        # 在2侧分别剪断, 计算下2侧分别可以保存的长度, 形式是: xx|xxx entity1 xxxxx entity2 xxx|xx, |表示被截断
                        can_keep_length = max_length - (t_end - h_start)
                        # 实体1左侧可以保留的长度
                        left_keep = int(can_keep_length / 2)
                        right_keep = can_keep_length - left_keep
                        # 句子的索引位置
                        left_start = h_start - left_keep
                        if left_start < 0:
                            left_start = 0
                        right_end = t_end + right_keep
                        # 截取后的文本长度
                        newtext = text[left_start:right_end]
                        h_start = h_start - left_start
                        h_end = h_start + h_length
                        t_start = t_start - left_start
                        t_end = t_start + t_length
                        assert newtext[h_start:h_end] == h_entity, "截断后的实体位置信息不对"
                        assert newtext[t_start:t_end] == t_entity, "截断后的实体位置信息不对"
                        # assert len(newtext) <= max_seq_len, f"最大长度截断后过长{len(newtext)}"
                else:
                    # 实体词h在后，t在前, 尚未修改, xx|xxx entity2 xxx|xx   +  xxx|xx entity1 xxxx|x, 其中|表示被截断的标记
                    if h_end - t_start > max_length:
                        half_length = max_length / 2
                        # 第一个实体前后的句子开始和结束位置
                        l1_start = t_start - int(half_length / 2)
                        if l1_start < 0:
                            l1_start = 0
                        l1_end = t_end + int(half_length / 2)
                        l2_start = h_start - int(half_length / 2)
                        l2_end = h_end + int(half_length / 2)
                        newtext = text[l1_start:l1_end] + text[l2_start:l2_end]
                        h_start = h_start - l1_start - (l2_start - l1_end)
                        h_end = h_start + h_length
                        t_start = t_start - l1_start
                        t_end = t_start + t_length
                        assert newtext[h_start:h_end] == h_entity, "截断后的实体位置信息不对"
                        assert newtext[t_start:t_end] == t_entity, "截断后的实体位置信息不对"
                        # assert len(newtext) <= max_seq_len, f"最大长度截断后过长{len(newtext)}"
                    else:
                        # 在2层分别剪断, 计算下2层分别可以保存的长度, xx|xxx entity2 xxxxx entity1 xxx|xx, |表示被截断
                        can_keep_length = max_length - (h_start - t_end)
                        # 实体1左侧可以保留的长度
                        left_keep = int(can_keep_length / 2)
                        right_keep = can_keep_length - left_keep
                        # 句子的索引位置
                        left_start = t_start - left_keep
                        if left_start < 0:
                            left_start = 0
                        right_end = h_end + right_keep
                        # 截取后的文本长度
                        newtext = text[left_start:right_end]
                        h_start = h_start - left_start
                        h_end = h_start + h_length
                        t_start = t_start - left_start
                        if t_start < 0:
                            t_start = 0
                        t_end = t_start + t_length
                        assert newtext[h_start:h_end] == h_entity, "截断后的实体位置信息不对"
                        assert newtext[t_start:t_end] == t_entity, "截断后的实体位置信息不对"
                        # assert len(newtext) <= max_seq_len, f"最大长度截断后过长{len(newtext)}"
                one['text'] = newtext
                one['brand']['pos'][0] = h_start
                one['brand']['pos'][1] = h_end
                one['attribute']['pos'][0] = t_start
                one['attribute']['pos'][1] = t_end
            else:
                length_counter['未超最大长度'] += 1
            truncate_data.append(one)
        print(f"超过和未超过最大长度{max_seq_len}的统计结果{length_counter}, 超过最大长度后将动态根据2个实体所在的位置对句子进行截断")
        return truncate_data
    def predict_brand(self, data):
        """
        预测品牌和属性的关系
        :param data:
        :type data:
        :return:
        :rtype:
        """
        truncate_data = self.truncate_relation(data)
        test_data_set = SinglePredictDataset(truncate_data, tokenizer=self.tokenizer, maxlen=self.max_seq_len,
                                             task_id=self.tasks_info['brand']['task_id'],
                                             task_def=self.tasks_info['brand']['task_def'])
        test_data = DataLoader(test_data_set, batch_size=self.predict_batch_size, collate_fn=self.collater.collate_fn,
                               pin_memory=self.cuda)
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
            id2tok = self.tasks_info['brand']['id2tok']
            predict_labels = [id2tok[p] for p in predictions]
            print(f"预测结果{predictions}, 预测的标签是 {predict_labels}")
        result = list(zip(predict_labels,scores))
        # 开始索引的位置
        return result
    def predict_purchase_single(self, data):
        convert_data = []
        #变成[content, title, keyword]的格式
        aspect_list = data['aspect_list']
        content = data['text']
        for keyword in aspect_list:
            one_data = [content, '', keyword]
            convert_data.append(one_data)
        # 预测
        result = self.predict_batch(task_name='purchase', data=convert_data)
        # 结果转换，返回字典格式
        final_res = {}
        for keyword, res in zip(aspect_list, result):
            final_res[keyword] = [res]
        return final_res
    def predict_purchase_batch(self, data):
        convert_data = []
        # 计数下每条数据中的aspect的个数
        aspect_count = [0] * len(data)
        for idx, one in enumerate(data):
            # 由于data中的aspect_list 有多个，需要进行拆分
            #变成[content, title, keyword]的格式
            aspect_list = one['aspect_list']
            content = one['content']
            title = one.get('title', '')
            for keyword in aspect_list:
                aspect_count[idx] += 1
                one_data = [content, title, keyword]
                convert_data.append(one_data)
        # 预测
        result = self.predict_batch(task_name='purchase', data=convert_data)
        # 结果转换，返回字典格式
        final_res = []
        # 索引result的结果
        start_idx = 0
        for cnt in aspect_count:
            end_idx = start_idx + cnt
            every_result = result[start_idx:end_idx]
            start_idx = end_idx
            final_res.append(every_result)
        return final_res
    def predict_pinpainer(self, data, max_seq_len=500, task_name='pinpainer'):
        """
        品牌的ner识别, 接收来自label-studio的数据, 只返回是品牌的那些词
        :param data:[[text, keywords_text]]
        :type data:
        :return: 嵌套列表 预测的返回的结果，keyword，对应的标签，一个概率值，位置信息
                        one_result = [keyword, label, '0.5', start, end]
        :rtype:
        """
        results = []
        if isinstance(data[0], str):
            # 数据中每个都是一个text格式的话
            text_data = [{"premise":d,"label":[0] * len(d)} for d in data]
        else:
            # 元祖格式，每个元祖包含2条数据，用于label-studio
            text_data = [{"premise":d[0],"label":[0] * len(d[0])} for d in data]
        test_data_set = SinglePredictDataset(text_data, tokenizer=self.tokenizer, max_seq_length=max_seq_len, task_id=self.tasks_info[task_name]['task_id'], task_def=self.tasks_info[task_name]['task_def'])
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
                score, pred, gold = self.model.predict(batch_info, batch_data)
                predictions.extend(pred)
                golds.extend(gold)
                scores.append(score)  #score还有一些问题，暂时不用了
                ids.extend(batch_info['uids'])
        id2tok = self.tasks_info[task_name]['id2tok']
        predict_labels = [[id2tok[tokp] for tokp in p] for p in predictions ]
        # 对预测的每个token的label进行筛选
        # print(f"预测结果{predictions}, 预测的标签是 {predict_labels}")
        data_tokens = []
        tok2txt_locations = []
        # text 转变成tokens
        for idx, sample in enumerate(data):
            if isinstance(data[0], str):
                premise = sample
            else:
                premise = sample[0]
            tokens = []
            token2text_loc = {}
            tokens_loc = 0
            for idx, word in enumerate(premise):
                # word是每个单词, subwords是子词
                subwords = self.tokenizer.tokenize(word)
                tokens.extend(subwords)
                # text的位置对应着起始和结束位置
                for i in range(tokens_loc,tokens_loc + len(subwords)):
                    # token的位置映射到原txt的位置
                    token2text_loc[tokens_loc] = idx
                tokens_loc = tokens_loc + len(subwords)
            # 确保和正确的tokenizer之前的长度保持一致
            tokens = tokens[:max_seq_len - 2]
            #所有tokens
            data_tokens.append(tokens)
            tok2txt_locations.append(token2text_loc)   #token和原文本长度的对应关系
        for plabel, tokens, t2tloc, sdata in zip(predict_labels, data_tokens, tok2txt_locations, data):
            if isinstance(data[0], str):
                text = sdata
            else:
                text = sdata[0]
            # 去掉第一个和最后一个token的预测结果，即去掉CLS和SEP
            token_label = plabel[1:-1]
            # 不相等也是有可能的，因为进行了截断或填充
            # assert len(text) == len(token_label), "预测的token的label长度和text的长度不等"
            pinpai_words = ""
            #保存这个品牌词的位置信息
            pinpai_words_idx = []
            # 保存这个句子所有的品牌词和品牌词的起始位置, 这是对应的tokenize后的内容，我们要找到原text中的内容，这里可能有UNK的出现
            p_words = []
            # 这是token的位置
            p_words_start_end = []
            # 对应到原文txt之后的位置
            text_words_start_end = []
            # 对应原txt之后的品牌词的内容，这里肯定不会出现UNK了
            text_pinpai_words = []
            assert len(token_label) == len(tokens), f"tokenizer后的tokens长度和预测后的label的长度不一致，请检查{len(token_label)}, {len(tokens)}:{tokens}"
            for idx in range(len(token_label)):
                # 单词的位置应该是tokenize后的结果，
                word = tokens[idx]
                if token_label[idx] == "B-PIN":
                    if pinpai_words:
                        #说明上一个词也是品牌词，说明这是连续的2个品牌词，那么这个信息存一下
                        p_words.append(pinpai_words)
                        token_start = pinpai_words_idx[0]
                        token_end = pinpai_words_idx[-1]+1
                        text_start = t2tloc[token_start]
                        text_end = t2tloc[token_end]
                        p_words_start_end.append([token_start,token_end])
                        text_words_start_end.append([text_start,text_end])
                        text_pinpai_word = text[text_start:text_end]
                        text_pinpai_words.append(text_pinpai_word)
                        # 重置
                        pinpai_words = ""
                        pinpai_words_idx = []
                    #发现品牌词的开头单词
                    pinpai_words += word
                    pinpai_words_idx.append(idx)
                elif token_label[idx] == "I-PIN":
                    # 如果没有预测到B-PIN开头，直接是I-PIN，那么这个也算一个单词
                    pinpai_words += word
                    pinpai_words_idx.append(idx)
                else:
                    if pinpai_words:
                        # 说明一个品牌词结束了，改保存了
                        p_words.append(pinpai_words)
                        token_start = pinpai_words_idx[0]
                        token_end = pinpai_words_idx[-1]
                        text_start = t2tloc[token_start]
                        text_end = t2tloc[token_end]
                        p_words_start_end.append([token_start,token_end])
                        text_words_start_end.append([text_start,text_end])
                        #因为截取的是片段，所以左开右闭，需要+1
                        text_pinpai_word = text[text_start:text_end+1]
                        text_pinpai_words.append(text_pinpai_word)
                        # 重置
                        pinpai_words = ""
                        pinpai_words_idx = []
            if pinpai_words:
                # 末尾可能的是品牌词的情况
                p_words.append(pinpai_words)
                token_start = pinpai_words_idx[0]
                token_end = pinpai_words_idx[-1]
                text_start = t2tloc[token_start]
                text_end = t2tloc[token_end]
                p_words_start_end.append([token_start, token_end])
                text_words_start_end.append([text_start, text_end])
                # 因为截取的是片段，所以左开右闭，需要+1
                text_pinpai_word = text[text_start:text_end+1]
                text_pinpai_words.append(text_pinpai_word)
            # 根据p_words_start_end（识别到的品牌词的token位置信息） 和t2tloc（token到text的位置映射）映射品牌词到原text中，修改p_words_start_end, 找出对应原文的正确的位置信息
            # 对每条数据的预测结果进行整理，返回label-studio需要的格式
            # one_result = [keyword, label, '0.5', start, end]
            result = []
            for pword, pstart_end in zip(text_pinpai_words, text_words_start_end):
                one_result = [pword, "品牌", '0.5', pstart_end[0], pstart_end[1]]
                result.append(one_result)
            results.append(result)
        return results
    def predict_wholesentiment(self, data, max_seq_len=150):
        """
        预测整体情感
        :param data:
        :type data:
        :param max_seq_len: 最大序列长度
        :return:
        :rtype:
        """
        truncate_data = []
        for idx, d in enumerate(data):
            if isinstance(d, str):
                content = d[:max_seq_len]
            elif isinstance(d, list):
                # 只取列表的第一个元素
                content = d[0][:max_seq_len]
            else:
                logger.warning(f"不支持的数据格式: {d}")
                content = "hello world"
            one = {"uid": idx, "premise": content, "label":"整体积极"}
            truncate_data.append(one)
        test_data_set = SinglePredictDataset(truncate_data, tokenizer=self.tokenizer, maxlen=self.max_seq_len,
                                             task_id=self.tasks_info['wholesentiment']['task_id'],
                                             task_def=self.tasks_info['wholesentiment']['task_def'])
        test_data = DataLoader(test_data_set, batch_size=self.predict_batch_size, collate_fn=self.collater.collate_fn,
                               pin_memory=self.cuda)
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
            id2tok = self.tasks_info['wholesentiment']['id2tok']
            predict_labels = [id2tok[p] for p in predictions]
            print(f"预测结果{predictions}, 预测的标签是 {predict_labels}")
        result = list(zip(predict_labels,scores))
        # 开始索引的位置
        return result


def verify_data(data, task):
    """
    校验用户传入的数据是否符合要求，不符合要求就会被返回错误信息, 如果符合要求，返回None, 否则返回检查的错误信息
    :param data: 用户数据
    :type data:
    :param task: 任务的名称，不同任务的数据不同
    :return:
    :rtype:
    """
    if data is None:
        return "传入的数据为空，请检查数据是否正确"
    if data == []:
        # 如果传入的数据是[]，直接返回[]
        return []
    if task == "brand":
        # 校验品牌功效的关系数据
        if not isinstance(data, list):
            return "传入的数据格式不对，应该是列表格式"
        #检查每个列表中的每个数据
        for idx,d in enumerate(data):
            # eg: {"text": " malin goetz清洁面膜。净化清洁 补水。温和不刺激 敏感肌都可用。柏瑞特dr.brandt清洁面膜。深层清洁 抗氧化 排浊 紧致皮肤。伊菲丹超级面膜。急救修护 补水 紧致抗老 提亮肤色。菲洛嘉十全大补面膜。补水保湿 细腻毛孔 提亮肤色。法尔曼幸福面膜。补水 修护 抗老 唤肤。奥伦纳素冰白面膜。深层补水 细腻毛孔 提亮肤色 舒缓修复。@美妆薯  @美妆情报局", "brand": {"name": "菲洛嘉十全大补面膜", "pos": [98, 107]}, "attribute": {"name": "提亮", "pos": [162, 164]}}
            if d.get('text') is None:
                return f"第{idx}条数据没有text字段"
            if d.get('brand') is None:
                return f"第{idx}条数据没有brand字段"
            if d.get('attribute') is None:
                return f"第{idx}条数据没有attribute字段"
            #校验brand和attribute中是否存在name和pos字段
            brand = d['brand']
            text = d['text']
            if brand.get('name') is None:
                return f"第{idx}条的brand项数据没有name字段"
            if brand.get('pos') is None:
                return f"第{idx}条的brand项数据没有pos字段"
            #校验name要存在于text中，并且pos的位置是对的
            brand_name = brand['name']
            brand_pos = brand['pos']
            brand_pos_start = brand_pos[0]
            brand_pos_end = brand_pos[1]
            pos_text = text[brand_pos_start:brand_pos_end]
            if pos_text != brand_name:
                return f"第{idx}条的brand name项数据对应的pos位置在原文中的位置不匹配,,应该是{brand_name},但是原文是{pos_text}"
            attribute = d['attribute']
            if attribute.get('name') is None:
                return f"第{idx}条的attribute项数据没有name字段"
            if attribute.get('pos') is None:
                return f"第{idx}条的attribute项数据没有pos字段"
            #校验name要存在于text中，并且pos的位置是对的
            attribute_name = attribute['name']
            attribute_pos = attribute['pos']
            attribute_pos_start = attribute_pos[0]
            attribute_pos_end = attribute_pos[1]
            pos_text = text[attribute_pos_start:attribute_pos_end]
            if pos_text != attribute_name:
                return f"第{idx}条的attribute name项数据对应的pos位置在原文中的位置不匹配,应该是{attribute_name},但是原文是{pos_text}"
    elif task == "purchase":
        #校验购买意向， 数据格式应该是类似
        # [[text,title,keyword],...]
        for idx, d in enumerate(data):
            if len(d) != 3:
                return f"第{idx}条数据传入的数据长度不对，请检查"
            # title可以为空，但是keyword必须要在text和title的组合内容中
            content = d[0] + d[1]
            keyword = d[2]
            if keyword not in content:
                #尝试变成小写
                print(f"第{idx}条数据传入的数据的关键字可能是不在文本或标题中，变成小写后重试")
                if keyword.lower() not in content.lower():
                    return f"第{idx}条数据传入的数据的关键字不在文本和标题中，请检查"
            if keyword == "":
                return f"第{idx}条数据传入的数据的关键字为空，请检查"
    elif task == "purchase_single":
        #校验购买意向， 数据格式应该是类似
        #         data = {
        #             "aspect_list" : aspect_list,
        #             "text": text,
        #             "channel": channel   #不是必须，没啥用
        #         }
        if not isinstance(data, dict):
            return "传入的数据不是字典格式，请检查"
        if not data.get('aspect_list'):
            return "传入的数据字典中没有aspect_list字段"
        if not data.get('text'):
            return "传入的数据字典中没有text字段"
    elif task == "purchase_batch":
        #校验购买意向， 数据格式应该是类似
        #   [{'aspect_list':xx, 'content':xxx, 'title':xxx},...]
        if not isinstance(data, list):
            return "传入的数据不是列表格式，请检查"
        for idx, one in enumerate(data):
            if not isinstance(one, dict):
                return "传入的单条的第{idx}条数据不是字典格式，请检查"
            if not one.get('aspect_list'):
                return "传入的的第{idx}条数据字典中没有aspect_list字段"
            if not one.get('content'):
                return f"传入的数据的第{idx}条字典中没有content字段"
            aspect_list = one.get('aspect_list')
            if not isinstance(aspect_list, list):
                return "传入的数据的第{idx}条aspect_list不是列表格式，请检查"
            if aspect_list == []:
                return "传入的数据的第{idx}条aspect_list是空的，请检查"
    elif task == 'dem8_predict':
        if len(data[0]) == 3:
            #数据是(content,aspect,属性)
            pass
        elif len(data[0]) ==5:
            #数据是 (content,aspect,属性，start,end)
            pass
        else:
            msg = "传入的数据的长度格式不符合要求，要求传入的nest list是2或4的长度"
            return msg
    elif task == 'wholesentiment':
        for idx, d in enumerate(data):
            if not isinstance(d, list) and not isinstance(d, str):
                msg = "传入的数据格式不符合要求，必须是包含字符串的列表或嵌套列表"
                return msg



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

@app.route("/api/absa_predict_sentence", methods=['POST'])
def absa_predict_sentence():
    """
    # 使用aspect base的数据做句子情感的判断，准确性不高
    情感的预测接收POST请求，获取data参数, data信息包含aspect关键在在句子中的位置信息，方便我们截取，我们截取aspect关键字的前后一定的字符作为输入
    例如关键字前后的25个字作为sentenceA，aspect关键字作为sentenceB，输入模型
    Args:
        test_data: 需要预测的数据，是一个文字列表, [content,...,]
    Returns: 返回格式是 [(predicted_label, predict_score, src_data),...]
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    results = model.predict_batch(task_name='absa', data=test_data,aspect_base=False)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/absa_predict_fullscore", methods=['POST'])
def absa_predict_fullscore():
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
    results = model.predict_batch(task_name='absa', data=test_data, full_score=True)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/absa_dem8_predict", methods=['POST'])
def absa_dem8_predict():
    """
    判断情感之前，判断属性，如果属性正确，才进行情感判断，否则不进行情感判断
    例如关键字前后的25个字作为sentenceA，aspect关键字作为sentenceB，输入模型
    Args:
        test_data: 需要预测的数据，是一个单个词的句子, [('持妆不能输雅诗兰黛上妆即定妆雅诗兰黛DW粉底是我的心头好持妆遮瑕磨皮粉底液测评', '遮瑕', '成分')]
        如果传过来的数据没有索引，那么需要自己去查找索引 [(content,aspect),...,]
    Returns: 返回格式是 [{'label': 0, 'score': 0, 'content': 0, 'keyword': 0, 'locations': 0}, {'label': '积极', 'score': 0.9996993541717529, 'content': '活动有赠品比较划算，之前买过快用完了，一支可以分两次使用，早上抗氧化必备VC', 'keyword': '抗氧化', 'locations': (31, 34)}
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    search_first = jsonres.get('search_first', False)
    full_score = jsonres.get('full_score', False)
    softmax = jsonres.get('softmax', True)
    # both_softmax_logits, list(zip(score,logits)), predict
    both_softmax_logits = jsonres.get('both_softmax_logits', False)
    if search_first:
        results = model.predict_absa_dem8_k1_s1(data=test_data, full_score=full_score, softmax=softmax, both_softmax_logits=both_softmax_logits)
    else:
        results = model.predict_absa_dem8_k1_sn(data=test_data, full_score=full_score)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/dem8_predict", methods=['POST'])
def dem8_predict():
    """
    8个维度的预测，接收POST请求，获取data参数, data信息包含aspect关键在在句子中的位置信息，方便我们截取，我们截取aspect关键字的前后一定的字符作为输入
    例如关键字前后的25个字作为sentenceA，aspect关键字作为sentenceB，输入模型
    Args:
        test_data: 需要预测的数据，是一个文字列表, [(content,aspect,属性),...,] 或者(content,aspect,属性，start,end)
        如果传过来的数据没有索引，那么需要自己去查找索引 [(content,aspect),...,]
    Returns: 返回格式是 [(predicted_label, predict_score),...]
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    verify_msg = verify_data(test_data, task='dem8_predict')
    if verify_msg is not None:
        return jsonify(verify_msg), 210
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
    verify_msg = verify_data(test_data, task='purchase')
    if verify_msg is not None:
        return jsonify(verify_msg), 210
    results = model.predict_batch(task_name='purchase', data=test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/purchase_predict_single", methods=['POST'])
def purchase_predict_single():
    """
    购买意向分类， 用于单条数据，供opinion项目调用
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
    verify_msg = verify_data(test_data, task='purchase_single')
    if verify_msg is not None:
        return jsonify(verify_msg), 210
    results = model.predict_purchase_single(data=test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/purchase_predict_batch", methods=['POST'])
def purchase_predict_batch():
    """
    购买意向分类， 主要是opnition的excel的接口进行调用
    例如关键字前后的25个字作为sentenceA，aspect关键字作为sentenceB，输入模型
    Args:
        接受数据是 [{'aspect_list':xx, 'content':xxx, 'title':xxx},...]
        或者：
        test_data: 需要预测的数据，是一个文字列表, [(content,title, aspect,start_idx, end_idx),...,]
        如果传过来的数据没有索引，那么需要自己去查找索引 [(content,aspect),...,]
    Returns: 返回格式是 [(predicted_label, predict_score),...]
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    # 是否只返回预测结果的格式， 返回一个列表，否则返回一个代score的列表
    logger.info(f"要进行的任务是购买意向判断")
    verify_msg = verify_data(test_data, task='purchase_batch')
    if verify_msg is not None:
        return jsonify(verify_msg), 210
    results = model.predict_purchase_batch(data=test_data)
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
    results = model.predict_dem8_kn_s1(data=test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/brand_predict", methods=['POST'])
def brand_predict():
    """
    品牌功效的关系判断
    Args:
        test_data: 需要预测的数据，是一个文字列表, [
{"text": " malin goetz清洁面膜。净化清洁 补水。温和不刺激 敏感肌都可用。柏瑞特dr.brandt清洁面膜。深层清洁 抗氧化 排浊 紧致皮肤。伊菲丹超级面膜。急救修护 补水 紧致抗老 提亮肤色。菲洛嘉十全大补面膜。补水保湿 细腻毛孔 提亮肤色。法尔曼幸福面膜。补水 修护 抗老 唤肤。奥伦纳素冰白面膜。深层补水 细腻毛孔 提亮肤色 舒缓修复。@美妆薯  @美妆情报局", "brand": {"name": "菲洛嘉十全大补面膜", "pos": [98, 107]}, "attribute": {"name": "提亮", "pos": [162, 164]}}
{"text": "修丽可有一说一真的好用啊  买爆r  买爆r  买爆r 。修丽可植萃亮妍精华露。色修又称色修精华，植物精萃，舒缓亮妍，白话一点就是，修护红敏，红血丝 祛痘印，肤色不匀称等。。修丽可cf精华。抗氧化防止皱纹生成，保护肌肤免受空气污染，抗衰指数5 同时有效的美白，淡化黑色素。。高端医美修丽可紫米精华。一瓶就含有10 玻色因，硬核抗老，饱满丰盈，抗衰紧致真的很心动，成分很安全敏感肌也可放心用哦！。修丽可b5保湿精华。兼具保湿和修护的两大功效，在给予水份锁住水份的同时，又能修护平日因刺激带来皮肤屏障损伤，很适合干敏肌的宝宝们 。修丽可发光瓶。3 传明酸   1 曲酸   5 烟酰胺   5 磺酸  去黄提亮 淡斑淡痘印 搭配同系列色修精华  高效淡化痘印的同时美白肌肤有效改善顽固黄褐斑。", "brand": {"name": "修丽可cf", "pos": [87, 92]}, "attribute": {"name": "抗衰", "pos": [116, 118]}}
]
    Returns: 返回格式是 [(predicted_label, predict_score),...]
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    verify_msg = verify_data(test_data, task='brand')
    if verify_msg is not None:
        return jsonify(verify_msg), 210
    results = model.predict_brand(data=test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/label_studio_pinpai_predict", methods=['POST'])
def pinpainer_labelstudio_predict():
    """
    用于label studio的品牌的预测, aspects词是可能是多个，是用逗号隔开
    Args:
        test_data: 需要预测的数据，是一个文字列表, [(content,aspects),...,]
    Returns: 返回格式是[one_result,one_result2,one_result3]
     嵌套列表 预测的返回的结果，keyword，对应的标签，一个概率值，位置信息
                    one_result = [keyword, label, '0.5', start, end]
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    results = model.predict_pinpainer(data=test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/pinpai_predict", methods=['POST'])
def pinpainer_predict():
    """
    用于label studio的品牌的预测, 文本内容是用逗号隔开
    Args:
        test_data: 需要预测的数据，是一个文字列表, [content,...,]
    Returns: 返回格式是[one_result,one_result2,one_result3]
     嵌套列表 预测的返回的结果，keyword，对应的标签，一个概率值，位置信息
                    one_result = [keyword, label, '0.5', start, end]
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    results = model.predict_pinpainer(data=test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/wholesentiment_predict", methods=['POST'])
def wholesentiment_predict():
    """
    整体情感分类任务
    Args:
        接受数据是 [content,...,]
    Returns: 返回格式是 [(predicted_label, predict_score),...]
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    logger.info(f"要进行的任务是整体情感判断")
    verify_msg = verify_data(test_data, task='wholesentiment')
    if verify_msg is not None:
        return jsonify(verify_msg), 210
    results = model.predict_wholesentiment(data=test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

if __name__ == "__main__":
    model = TorchMTDNNModel()
    app.run(host='0.0.0.0', port=3326, debug=False, threaded=False)