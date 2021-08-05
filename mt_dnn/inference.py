# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
from data_utils.metrics import calc_metrics
from mt_dnn.batcher import Collater
from data_utils.task_def import TaskType
import torch
from tqdm import tqdm

def extract_encoding(model, data, use_cuda=True):
    if use_cuda:
        model.cuda()
    sequence_outputs = []
    max_seq_len = 0
    for idx, (batch_info, batch_data) in enumerate(data):
        batch_info, batch_data = Collater.patch_data(use_cuda, batch_info, batch_data)
        sequence_output = model.encode(batch_info, batch_data)
        sequence_outputs.append(sequence_output)
        max_seq_len = max(max_seq_len, sequence_output.shape[1])
    
    new_sequence_outputs = []
    for sequence_output in sequence_outputs:
        new_sequence_output = torch.zeros(sequence_output.shape[0], max_seq_len, sequence_output.shape[2])
        new_sequence_output[:, :sequence_output.shape[1], :] = sequence_output
        new_sequence_outputs.append(new_sequence_output)

    return torch.cat(new_sequence_outputs)

def eval_model(model, data, metric_meta, device, with_label=True, label_mapper=None, task_type=TaskType.Classification):
    """
    存储所有数据的预测结果
    :param model:
    :type model:
    :param data:
    :type data:
    :param metric_meta:
    :type metric_meta:
    :param device:
    :type device:
    :param with_label:
    :type with_label:
    :param label_mapper:
    :type label_mapper:
    :param task_type:
    :type task_type:
    :return: metrics: {'ACC': 68.92341842397336},  predictions: list, 预测的id
            scores: list,预测的概率,  golds: gold 标签的id, ids: 预测数据的索引
    :rtype:
    """
    predictions = []
    golds = []
    scores = []
    ids = []
    metrics = {}
    # 按顺序读取一个批次的数据
    for (batch_info, batch_data) in data:
        batch_info, batch_data = Collater.patch_data(device, batch_info, batch_data)
        # score是预测的置信度，pred是预测的类别id，gold是golden-standard
        score, pred, gold = model.predict(batch_info, batch_data)
        predictions.extend(pred)
        golds.extend(gold)
        scores.extend(score)
        ids.extend(batch_info['uids'])

    if task_type == TaskType.Span:
        from experiments.squad import squad_utils
        golds = squad_utils.merge_answers(ids, golds)
        predictions, scores = squad_utils.select_answers(ids, predictions, scores)
    if with_label:
        metrics = calc_metrics(metric_meta, golds, predictions, scores, label_mapper)
    return metrics, predictions, scores, golds, ids
