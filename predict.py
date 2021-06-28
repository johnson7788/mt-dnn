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

def dump(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

parser = argparse.ArgumentParser()
parser.add_argument("--task_def", type=str, default="experiments/glue/glue_task_def.yml",help='任务的配置文件')
parser.add_argument("--task", type=str, help='预测哪个任务, 你的任务配置文件中的任务的名字')
parser.add_argument("--task_id", type=int, help="训练时的任务id，根据训练时指定的--train_datasets参数的顺序，默认从0开始")

parser.add_argument("--prep_input", type=str, help='测试集的数据的文件的路径', default='data_my/canonical_data/bert-base-chinese/absa_test.json')
parser.add_argument("--with_label", action="store_true", help='打印metrics')
parser.add_argument("--score", type=str, default='predict_score.txt',help="scores的保存路径, 每个样本的predict的分数，即最大的概率")

parser.add_argument('--max_seq_len', type=int, default=512, help='最大序列长度')
parser.add_argument('--batch_size_eval', type=int, default=8, help='评估的batch_size大小')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(), help='是否使用GPU')
parser.add_argument("--checkpoint", default='mt_dnn_models/bert_model_base_uncased.pt', type=str,help='你训练好的模型，使用哪个模型进行预测')

args = parser.parse_args()

# load task info, eg: absa
task = args.task
task_defs = TaskDefs(args.task_def)
assert args.task in task_defs._task_type_map, "确保任务在任务的配置文件中"
assert args.task in task_defs._data_type_map, "确保任务在任务的配置文件中"
assert args.task in task_defs._metric_meta_map, "确保任务在任务的配置文件中"
# eg: absa
prefix = task.split('_')[0]
task_def = task_defs.get_task_def(prefix)
# eg: DataFormat.PremiseAndOneHypothesis
data_type = task_defs._data_type_map[args.task]
# eg: TaskType.Classification
task_type = task_defs._task_type_map[args.task]
# eg: Metric.ACC
metric_meta = task_defs._metric_meta_map[args.task]
# load model, eg: 'trained_model/absa_dem8.pt'
checkpoint_path = args.checkpoint
# 确保checkpoint存在
assert os.path.exists(checkpoint_path)

# 是否放GPU
if args.cuda and torch.cuda.is_available():
    state_dict = torch.load(checkpoint_path)
    device = torch.device("cuda")
else:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    device = torch.device("cpu")
# 配置和模型保存到一起了
config = state_dict['config']
config["cuda"] = args.cuda
task_def = task_defs.get_task_def(prefix)
task_def_list = [task_def]
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
test_data_set = SingleTaskDataset(path=args.prep_input, is_train=False, maxlen=args.max_seq_len, task_id=args.task_id, task_def=task_def)
collater = Collater(is_train=False, encoder_type=encoder_type)
test_data = DataLoader(test_data_set, batch_size=args.batch_size_eval, collate_fn=collater.collate_fn, pin_memory=args.cuda)


with torch.no_grad():
    test_metrics, test_predictions, scores, golds, test_ids = eval_model(model=model, data=test_data,
                                                                         metric_meta=metric_meta,
                                                                         device=device, with_label=args.with_label)

    results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids, 'scores': scores}
    dump(path=args.score, data=results)
    if args.with_label:
        print(f"测试的数据总量是{len(test_ids)}, 测试的结果是{test_metrics}")