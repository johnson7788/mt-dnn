# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import yaml
import os
import numpy as np
import argparse
import json
import sys
from data_utils import load_data
from data_utils.task_def import TaskType, DataFormat
from data_utils.log_wrapper import create_logger
from experiments.exp_def import TaskDefs
from experiments.squad import squad_utils
from transformers import AutoTokenizer
import torch


DEBUG_MODE = False
MAX_SEQ_LEN = 512
DOC_STRIDE = 180
MAX_QUERY_LEN = 64
MRC_MAX_SEQ_LEN = 384

logger = create_logger(
    __name__,
    to_disk=True,
    log_file='mt_dnn_data_proc_{}.log'.format(MAX_SEQ_LEN))

def feature_extractor(tokenizer, text_a, text_b=None, max_length=512, do_padding=False):
    if tokenizer.do_lower_case:
        text_a = text_a.lower()
        if text_b:
            text_b = text_b.lower()
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
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
    return input_ids, attention_mask, token_type_ids

def relation_feature_extractor(tokenizer, item, max_length=512, do_padding=False):
    """
    关系判断的数据tokenize
    :return:
    :rtype:
    """
    # Sentence -> token
    sentence = item['text']
    pos_head = item['h']['pos']
    pos_tail = item['t']['pos']

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

def build_data(data, dump_path, tokenizer, data_format=DataFormat.PremiseOnly,
               max_seq_len=MAX_SEQ_LEN, lab_dict=None, do_padding=False, truncation=True):
    """

    :param data: [{'uid': '0', 'label': 0, 'premise': '合盖子的时候有种塑料感', 'hypothesis': '塑料'}, {}]
    :type data:
    :param dump_path:
    :type dump_path:
    :param tokenizer:
    :type tokenizer:
    :param data_format:
    :type data_format:
    :param max_seq_len:
    :type max_seq_len:
    :param lab_dict: <data_utils.vocab.Vocabulary object at 0x7f4736cd2100>
        INIT_LEN = {int} 4
        ind2tok = {dict: 3} {0: '消极', 1: '中性', 2: '积极'}
        neat = {bool} True
        tok2ind = {dict: 3} {'消极': 0, '中性': 1, '积极': 2}
    :type lab_dict:
    :param do_padding:
    :type do_padding:
    :param truncation:
    :type truncation:
    :return:
    :rtype:
    """
    def build_data_premise_only(
            data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None):
        """Build data of single sentence tasks
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                label = sample['label']
                input_ids, input_mask, type_ids = feature_extractor(tokenizer, premise, max_length=max_seq_len)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids,
                    'type_id': type_ids,
                    'attention_mask': input_mask}
                writer.write('{}\n'.format(json.dumps(features)))

    def build_data_premise_and_one_hypo(
            data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None):
        """Build data of sentence pair tasks
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                hypothesis = sample['hypothesis']
                label = sample['label']
                input_ids, input_mask, type_ids = feature_extractor(tokenizer, premise, text_b=hypothesis, max_length=max_seq_len)
                features = {
                    'uid': ids,
                    'label': label,
                    'token_id': input_ids,
                    'type_id': type_ids,
                    'attention_mask': input_mask}
                writer.write('{}\n'.format(json.dumps(features)))

    def build_data_premise_and_multi_hypo(
            data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None):
        """Build QNLI as a pair-wise ranking task
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                hypothesis_list = sample['hypothesis']
                label = sample['label']
                input_ids_list = []
                type_ids_list = []
                attention_mask_list = []
                for hypothesis in hypothesis_list:
                    input_ids, input_mask, type_ids = feature_extractor(tokenizer,
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
                writer.write('{}\n'.format(json.dumps(features)))

    def build_data_sequence(data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, label_mapper=None):
        with open(dump_path, 'w', encoding='utf-8') as writer:
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
                if len(premise) >  max_seq_len - 2:
                    tokens = tokens[:max_seq_len - 2]
                    labels = labels[:max_seq_len - 2]

                label = [label_mapper['CLS']] + labels + [label_mapper['SEP']]
                input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
                assert len(label) == len(input_ids)
                type_ids = [0] * len(input_ids)
                features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
                writer.write('{}\n'.format(json.dumps(features)))

    def build_data_mrc(data, dump_path, max_seq_len=MRC_MAX_SEQ_LEN, tokenizer=None, label_mapper=None, is_training=True):
        with open(dump_path, 'w', encoding='utf-8') as writer:
            unique_id = 1000000000 # TODO: this is from BERT, needed to remove it...
            for example_index, sample in enumerate(data):
                ids = sample['uid']
                doc = sample['premise']
                query = sample['hypothesis']
                label = sample['label']
                doc_tokens, cw_map = squad_utils.token_doc(doc)
                answer_start, answer_end, answer, is_impossible = squad_utils.parse_squad_label(label)
                answer_start_adjusted, answer_end_adjusted = squad_utils.recompute_span(answer, answer_start, cw_map)
                is_valid = squad_utils.is_valid_answer(doc_tokens, answer_start_adjusted, answer_end_adjusted, answer)
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
                                        MAX_QUERY_LEN,
                                        DOC_STRIDE,
                                        answer_text=answer,
                                        is_training=True)
                unique_id += len(feature_list)
                for feature in feature_list:
                    so = json.dumps({'uid': ids,
                                'token_id' : feature.input_ids,
                                'mask': feature.input_mask,
                                'type_id': feature.segment_ids,
                                'example_index': feature.example_index,
                                'doc_span_index':feature.doc_span_index,
                                'tokens': feature.tokens,
                                'token_to_orig_map': feature.token_to_orig_map,
                                'token_is_max_context': feature.token_is_max_context,
                                'start_position': feature.start_position,
                                'end_position': feature.end_position,
                                'label': feature.is_impossible,
                                'doc': doc,
                                'doc_offset': feature.doc_offset,
                                'answer': [answer]})
                    writer.write('{}\n'.format(so))


    if data_format == DataFormat.PremiseOnly:
        build_data_premise_only(
            data,
            dump_path,
            max_seq_len,
            tokenizer)
    elif data_format == DataFormat.PremiseAndOneHypothesis:
        build_data_premise_and_one_hypo(
            data, dump_path, max_seq_len, tokenizer)
    elif data_format == DataFormat.PremiseAndMultiHypothesis:
        build_data_premise_and_multi_hypo(
            data, dump_path, max_seq_len, tokenizer)
    elif data_format == DataFormat.Sequence:
        build_data_sequence(data, dump_path, max_seq_len, tokenizer, lab_dict)
    elif data_format == DataFormat.MRC:
        build_data_mrc(data, dump_path, max_seq_len, tokenizer)
    elif data_format == DataFormat.RELATION:
        build_data_relation(data, dump_path, max_seq_len, tokenizer)
    else:
        raise ValueError(data_format)

def build_data_relation(data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None):
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
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            #一条包含text，头部实体，尾部实体，和实体位置的json的字符床
            data = sample['data']
            item = json.loads(data)
            label = sample['label']
            input_ids, input_mask, type_ids = relation_feature_extractor(tokenizer, item, max_length=max_seq_len, do_padding=False)
            features = {
                'uid': ids,
                'label': label,
                'token_id': input_ids,
                'type_id': type_ids,
                'attention_mask': input_mask}
            writer.write('{}\n'.format(json.dumps(features)))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocessing GLUE/SNLI/SciTail dataset.')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                        help='support all BERT and ROBERTA family supported by HuggingFace Transformers')
    parser.add_argument('--do_lower_case', action='store_true', help='是否变成小写')
    parser.add_argument('--do_padding', action='store_true')
    parser.add_argument('--dataset', type=str, default='all',help='默认处理哪个数据集，all代表所有, 或者部分数据集，例如absa,dem8这样用逗号分割的参数')
    parser.add_argument('--root_dir', type=str, default='data/canonical_data',help='规范后的数据的位置，处理后的输出目录也在这个目录下，根据处理的模型的名字命名')
    parser.add_argument('--task_def', type=str, default="experiments/glue/glue_task_def.yml")

    args = parser.parse_args()
    return args


def main(args):
    # hyper param
    root = args.root_dir
    assert os.path.exists(root), f"路径{root}不存在"

    tokenizer = AutoTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)

    mt_dnn_root = os.path.join(root, args.model)
    if not os.path.isdir(mt_dnn_root):
        os.makedirs(mt_dnn_root)

    task_defs = TaskDefs(args.task_def)
    datasets = args.dataset.split(',')
    for task in task_defs.get_task_names():
        task_def = task_defs.get_task_def(task)
        if 'all' in datasets or task in datasets:
            logger.info("开始tokenize任务: %s" % task)
            for split_name in task_def.split_names:
                file_path = os.path.join(root, "%s_%s.tsv" % (task, split_name))
                if not os.path.exists(file_path):
                    logger.warning(f"文件{file_path}不存在，请检查")
                    sys.exit(1)
                rows = load_data(file_path, task_def)
                dump_path = os.path.join(mt_dnn_root, "%s_%s.json" % (task, split_name))
                logger.info(f"保存文件到{dump_path}")
                build_data(
                    rows,
                    dump_path,
                    tokenizer,
                    task_def.data_type,
                    lab_dict=task_def.label_vocab)

if __name__ == '__main__':
    args = parse_args()
    main(args)
