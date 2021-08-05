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
import pickle

def dump(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def got_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_def", type=str, default="experiments/glue/glue_task_def.yml",help='任务的配置文件')
    parser.add_argument("--task", type=str, help='预测哪个任务, 你的任务配置文件中的任务的名字')
    parser.add_argument("--task_id", type=int, help="训练时的任务id，根据训练时指定的--train_datasets参数的顺序，默认从0开始")
    parser.add_argument("--prep_input", type=str, help='测试集的数据的文件的路径', default='data_my/canonical_data/bert-base-chinese/absa_test.json')
    parser.add_argument("--with_label", action="store_true", help='打印metrics')
    parser.add_argument("--score", type=str, default='predict_score.txt',help="scores的保存路径, 每个样本的predict的分数，即最大的概率")
    parser.add_argument('--max_seq_len', type=int, default=512, help='最大序列长度')
    parser.add_argument('--batch_size_eval', type=int, default=32, help='评估的batch_size大小')
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(), help='是否使用GPU')
    parser.add_argument("--checkpoint", default='mt_dnn_models/bert_model_base_uncased.pt', type=str,help='你训练好的模型，使用哪个模型进行预测')
    parser.add_argument('--do_collection', action="store_true", help='收集预测错误的数据，保存的特定的文件中，由collection_file指定')
    parser.add_argument("--collection_file", type=str, help='如果使用收集预测错误的数据的选项，那么需要指定把错误的数据保存到哪里，保存成pickle的格式,这里是给定一个文件的名字包含路径', default=None)
    args = parser.parse_args()
    return args

def do_predict(task, task_def, task_id, prep_input, with_label, score, max_seq_len, batch_size_eval, checkpoint, cuda, do_collection, collection_file):
    # load task info, eg: absa
    task_defs = TaskDefs(task_def)
    assert task in task_defs._task_type_map, "确保任务在任务的配置文件中"
    assert task in task_defs._data_type_map, "确保任务在任务的配置文件中"
    assert task in task_defs._metric_meta_map, "确保任务在任务的配置文件中"
    # eg: absa
    prefix = task.split('_')[0]
    task_def = task_defs.get_task_def(prefix)
    # eg: DataFormat.PremiseAndOneHypothesis
    data_type = task_defs._data_type_map[task]
    # eg: TaskType.Classification
    task_type = task_defs._task_type_map[task]
    # eg: Metric.ACC
    metric_meta = task_defs._metric_meta_map[task]
    # load model, eg: 'trained_model/absa_dem8.pt'
    checkpoint_path = checkpoint
    # 确保checkpoint存在
    assert os.path.exists(checkpoint_path)

    # 是否放GPU
    if cuda and torch.cuda.is_available():
        state_dict = torch.load(checkpoint_path)
        device = torch.device("cuda")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        device = torch.device("cpu")
    # 配置和模型保存到一起了
    config = state_dict['config']
    config["cuda"] = cuda
    #每个任务对应一个配置, 必须给定，否则任务不加载
    task_def_dem8 = task_defs.get_task_def('dem8')
    task_def_absa = task_defs.get_task_def('absa')
    task_def_purchase = task_defs.get_task_def('purchase')
    task_def_list = [task_def_absa,task_def_dem8,task_def_purchase]
    #加载到配置中
    config['task_def_list'] = task_def_list
    ## temp fix
    config['fp16'] = False
    config['answer_opt'] = 0
    config['adv_train'] = False
    del state_dict['optimizer']
    # 初始化模型
    model = MTDNNModel(config, device=device, state_dict=state_dict)
    # encoder的类型 EncoderModelType.BERT
    encoder_type = config.get('encoder_type', EncoderModelType.BERT)
    # load data， 加载数据集
    test_data_set = SingleTaskDataset(path=prep_input, is_train=False, maxlen=max_seq_len, task_id=task_id, task_def=task_def)
    collater = Collater(is_train=False, encoder_type=encoder_type)
    test_data = DataLoader(test_data_set, batch_size=batch_size_eval, collate_fn=collater.collate_fn, pin_memory=cuda)


    with torch.no_grad():
        # test_metrics eg: acc结果，准确率结果
        # test_predictions: 预测的结果， scores预测的置信度， golds是我们标注的结果，标注的label， test_ids样本id
        test_metrics, test_predictions, scores, golds, test_ids = eval_model(model=model, data=test_data,
                                                                             metric_meta=metric_meta,
                                                                             device=device, with_label=with_label)
        results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids, 'scores': scores}
        dump(path=score, data=results)
        if with_label:
            print(f"测试的数据总量是{len(test_ids)}, 测试的结果是{test_metrics}")
    # 把预测的id和gold的id变成标签名字
    id2label = task_def_list[task_id].label_vocab.ind2tok
    predict_labels = [id2label[p] for p in test_predictions]
    gold_labels = [id2label[p] for p in golds]
    if do_collection:
        # 预测数据的位置 prep_input 对应的源文件位置, json是处理过后的数据，tsv是源数据
        json_file = os.path.basename(prep_input)
        tsv_file = os.path.join("data_my/canonical_data", json_file)
        tsv_file = tsv_file[:-4] + 'tsv'
        assert os.path.exists(tsv_file), "tsv源文件不存在，请检查"
        collect_wrongs(save_file=collection_file, predictions=predict_labels, goldens=gold_labels, src_file=tsv_file)
    return test_metrics, predict_labels, scores, gold_labels, test_ids
def collect_wrongs(save_file,predictions,goldens,src_file):
    """
    把预测错误的过滤出来，找到源数据，然后保存到save_file中
    :param save_file: 要保存的cache文件
    :type save_file:
    :param predictions: 预测结果的标签名
    :type predictions:
    :param goldens: 真实结果标签
    :type goldens:
    :param src_file: 源文件的保存位置，用于提取错误的源数据
    :return:
    :rtype:
    """
    print(f"要对源文件{src_file}的预测错误的样本进行筛选，保存到{save_file}中")
    with open(src_file, 'r') as f:
        lines = f.readlines()
    assert len(lines) == len(predictions) == len(goldens), "预测的标签和真实的文件行数不匹配，请检查"
    wrong_predicts = []
    for p, g, l in zip(predictions,goldens,lines):
        if p != g:
            #预测结果不一样，
            data = {
                "predict": p,
                "label": g,
                "srctext": l,
            }
            wrong_predicts.append(data)
    print(f"总样本数量{len(predictions)},预测错误的样本数量{len(wrong_predicts)}")
    pickle.dump(wrong_predicts, open(save_file, "wb"))
if __name__ == '__main__':
    args = got_args()
    do_predict(task=args.task, task_def=args.task_def, task_id=args.task_id, prep_input=args.prep_input, with_label=args.with_label, score=args.score, max_seq_len=args.max_seq_len, batch_size_eval=args.batch_size_eval, checkpoint=args.checkpoint, cuda=args.cuda, do_collection=args.do_collection, collection_file=args.collection_file)