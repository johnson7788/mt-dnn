# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import argparse
import json
import os
import random
from datetime import datetime
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
from pretrained_models import *
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
from experiments.exp_def import TaskDefs
from mt_dnn.inference import eval_model, extract_encoding
from data_utils.log_wrapper import create_logger
from data_utils.task_def import EncoderModelType
from data_utils.utils import set_environment
from mt_dnn.batcher import SingleTaskDataset, MultiTaskDataset, Collater, MultiTaskBatchSampler, DistMultiTaskBatchSampler, DistSingleTaskBatchSampler
from mt_dnn.batcher import DistTaskDataset
from mt_dnn.model import MTDNNModel


def model_config(parser):
    parser.add_argument('--update_bert_opt', default=0, type=int, help='是否更新固定预训练的bert模型参数，大于0表示固定')
    parser.add_argument('--multi_gpu_on', action='store_true',help='默认False，是否使用多GPU')
    parser.add_argument('--mem_cum_type', type=str, default='simple',
                        help='bilinear/simple/default')
    parser.add_argument('--answer_num_turn', type=int, default=5,help='论文中的超参数K，K步推理')
    parser.add_argument('--answer_mem_drop_p', type=float, default=0.1)
    parser.add_argument('--answer_att_hidden_size', type=int, default=128)
    parser.add_argument('--answer_att_type', type=str, default='bilinear', help='bilinear/simple/default')
    parser.add_argument('--answer_rnn_type', type=str, default='gru', help='SAN逐步推理模块使用的结构是，rnn/gru/lstm')
    parser.add_argument('--answer_sum_att_type', type=str, default='bilinear', help='bilinear/simple/default')
    parser.add_argument('--answer_merge_opt', type=int, default=1)
    parser.add_argument('--answer_mem_type', type=int, default=1)
    parser.add_argument('--max_answer_len', type=int, default=10)
    parser.add_argument('--answer_dropout_p', type=float, default=0.1)
    parser.add_argument('--answer_weight_norm_on', action='store_true')
    parser.add_argument('--dump_state_on', action='store_true')
    parser.add_argument('--answer_opt', type=int, default=1, help='可选0,1，代表是否使用SANClassifier分类头还是普通的线性分类头,1表示使用SANClassifier, 0是普通线性映射')
    parser.add_argument('--pooler_actf', type=str, default='tanh',
                        help='tanh/relu/gelu, 构建输出头的时的激活函数的选择')
    parser.add_argument('--mtl_opt', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--mix_opt', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--init_ratio', type=float, default=1)
    parser.add_argument('--encoder_type', type=int, default=EncoderModelType.BERT)
    parser.add_argument('--num_hidden_layers', type=int, default=-1, help='-1表示不修改模型的隐藏层参数，使用默认值，否则修改')

    # BERT pre-training
    parser.add_argument('--bert_model_type', type=str, default='bert-base-uncased',help='使用的预训练模型')
    parser.add_argument('--do_lower_case', action='store_true',help='是否小写')
    parser.add_argument('--masked_lm_prob', type=float, default=0.15)
    parser.add_argument('--short_seq_prob', type=float, default=0.2)
    parser.add_argument('--max_predictions_per_seq', type=int, default=128)

    # bin samples
    parser.add_argument('--bin_on', action='store_true')
    parser.add_argument('--bin_size', type=int, default=64)
    parser.add_argument('--bin_grow_ratio', type=int, default=0.5)

    # 分布式 training
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--world_size", type=int, default=1, help="For distributed training: world size")
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="6600")
    parser.add_argument("--backend", type=str, default="nccl")
    return parser


def data_config(parser):
    parser.add_argument('--log_file', default='mt-dnn-train.log', help='path for log file.')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--tensorboard_logdir', default='tensorboard_logdir')
    parser.add_argument("--init_checkpoint", default='mt_dnn_models/bert_model_base_uncased.pt', type=str, help='使用哪个模型初始模型参数，请注意，选择正确的中英文模型')
    parser.add_argument('--data_dir', default='data/canonical_data/bert_uncased_lower',help='tokenize后的数据的地址')
    parser.add_argument('--data_sort_on', action='store_true')
    parser.add_argument('--name', default='farmer')
    parser.add_argument('--task_def', type=str, default="experiments/glue/glue_task_def.yml",help="使用的task任务定义的文件，默认是glue的task进行训练")
    parser.add_argument('--train_datasets', default='mnli',help='训练的多个任务的数据集，用逗号,分隔，如果多个数据集存在')
    parser.add_argument('--test_datasets', default='mnli_matched,mnli_mismatched',help='测试的多个任务的数据集，用逗号,分隔，如果多个数据集存在，根据任务名前缀自动匹配，例如mnli的前半部分mnli_')
    parser.add_argument('--glue_format_on', action='store_true')
    parser.add_argument('--mkd-opt', type=int, default=0, 
                        help=">0表示开启知识蒸馏, requires 'softlabel' column in input data")
    parser.add_argument('--do_padding', action='store_true')
    return parser


def train_config(parser):
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='是否使用GPU')
    parser.add_argument('--log_per_updates', type=int, default=500)
    parser.add_argument('--save_per_updates', type=int, default=10000,help='结合save_per_updates_on一起使用，表示每多少step，进行模型评估和保存')
    parser.add_argument('--save_per_updates_on', action='store_true',help='每一步都保存模型，保存频繁,每步都评估 ')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8, help='训练的batch_size')
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd, adadelta, adam， 使用的优化器')
    parser.add_argument('--grad_clipping', type=float, default=0)
    parser.add_argument('--global_grad_clipping', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--warmup_schedule', type=str, default='warmup_linear')
    parser.add_argument('--adam_eps', type=float, default=1e-6)

    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.1,help='构建输出头时Pooler的dropout设置')
    parser.add_argument('--dropout_w', type=float, default=0.000)
    parser.add_argument('--bert_dropout_p', type=float, default=0.1)

    # loading
    parser.add_argument("--model_ckpt", default='checkpoints/model_0.pt', type=str, help='继续训练模型时的已存在模型')
    parser.add_argument("--resume", action='store_true',help='继续训练模型，结合参数--model_ckpt一起使用')

    # scheduler
    parser.add_argument('--have_lr_scheduler', dest='have_lr_scheduler', action='store_false')
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    #parser.add_argument('--feature_based_on', action='store_true')
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--output_dir', default='checkpoint')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--grad_accumulation_step', type=int, default=1)

    #fp 16
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # adv training
    parser.add_argument('--adv_train', action='store_true')
    # the current release only includes smart perturbation
    parser.add_argument('--adv_opt', default=0, type=int)
    parser.add_argument('--adv_norm_level', default=0, type=int)
    parser.add_argument('--adv_p_norm', default='inf', type=str)
    parser.add_argument('--adv_alpha', default=1, type=float)
    parser.add_argument('--adv_k', default=1, type=int)
    parser.add_argument('--adv_step_size', default=1e-5, type=float)
    parser.add_argument('--adv_noise_var', default=1e-5, type=float)
    parser.add_argument('--adv_epsilon', default=1e-6, type=float)
    parser.add_argument('--encode_mode', action='store_true', help='只把测试数据用模型编码一下，然后保存到checkpoint目录，没啥用')
    parser.add_argument('--debug', action='store_true', help="print debug info")
    return parser

# 各种参数
parser = argparse.ArgumentParser()
parser = data_config(parser)
parser = model_config(parser)
parser = train_config(parser)

args = parser.parse_args()

output_dir = args.output_dir
data_dir = args.data_dir
args.train_datasets = args.train_datasets.split(',')
args.test_datasets = args.test_datasets.split(',')

os.makedirs(output_dir, exist_ok=True)
output_dir = os.path.abspath(output_dir)

set_environment(args.seed, args.cuda)
log_path = args.log_file
logger = create_logger(__name__, to_disk=True, log_file=log_path)

task_defs = TaskDefs(args.task_def)
encoder_type = args.encoder_type

def dump(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def evaluation(model, datasets, data_list, task_defs, output_dir='checkpoints', epoch=0, n_updates=-1, with_label=False, tensorboard=None, glue_format_on=False, test_on=False, device=None, logger=None):
    # eval on rank 1
    print_message(logger, "开始评估")
    test_prefix = "Test" if test_on else "Dev"
    if n_updates > 0:
        updates_str = "updates"
    else:
        updates_str = "epoch"
    updates = model.updates if n_updates > 0 else epoch
    for idx, dataset in enumerate(datasets):
        prefix = dataset.split('_')[0]
        task_def = task_defs.get_task_def(prefix)
        label_dict = task_def.label_vocab
        test_data = data_list[idx]
        if test_data is not None:
            with torch.no_grad():
                test_metrics, test_predictions, test_scores, test_golds, test_ids= eval_model(model,
                                                                                test_data,
                                                                                metric_meta=task_def.metric_meta,
                                                                                device=device,
                                                                                with_label=with_label,
                                                                                label_mapper=label_dict,
                                                                                task_type=task_def.task_type)
            for key, val in test_metrics.items():
                if tensorboard:
                    tensorboard.add_scalar('{}/{}/{}'.format(test_prefix, dataset, key), val, global_step=updates)
                if isinstance(val, str):
                    print_message(logger, '任务是 {0} -- {1} {2} -- {3} {4}: {5}'.format(dataset, updates_str, updates, test_prefix, key, val), level=1)
                elif isinstance(val, float):
                    print_message(logger, '任务是 {0} -- {1} {2} -- {3} {4}: {5:.3f}'.format(dataset, updates_str, updates, test_prefix, key, val), level=1)
                else:
                    test_metrics[key] = str(val)
                    print_message(logger, 'Task {0} -- {1} {2} -- {3} {4}: \n{5}'.format(dataset, updates_str, updates, test_prefix, key, val), level=1)

            if args.local_rank in [-1, 0]:
                score_file = os.path.join(output_dir, '{}_{}_scores_{}_{}.json'.format(dataset, test_prefix.lower(), updates_str, updates))
                results = {'metrics': test_metrics, 'predictions': test_predictions, 'uids': test_ids, 'scores': test_scores}
                dump(score_file, results)
                if glue_format_on:
                    from experiments.glue.glue_utils import submit
                    official_score_file = os.path.join(output_dir, '{}_{}_scores_{}.tsv'.format(dataset, test_prefix.lower(), updates_str))
                    submit(official_score_file, results, label_dict)
def initialize_distributed(args):
    """Initialize torch.distributed."""
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    if os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'):
        # We are using (OpenMPI) mpirun for launching distributed data parallel processes
        local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
        local_size = int(os.getenv('OMPI_COMM_WORLD_LOCAL_SIZE'))
        args.local_rank = local_rank
        args.rank = nodeid * local_size + local_rank
        args.world_size = num_nodes * local_size
    #args.batch_size = args.batch_size * args.world_size

    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    device = torch.device('cuda', args.local_rank)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6600')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)
    return device

def print_message(logger, message, level=0):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            do_logging = True
        else:
            do_logging = False
    else:
        do_logging = True
    if do_logging:
        if level == 1:
            logger.warning(message)
        else:
            logger.info(message)

def main():
    # set up dist
    if args.local_rank > -1:
        device = initialize_distributed(args)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # opt还是args，只不过是字典格式
    opt = vars(args)
    # update data dir
    opt['data_dir'] = data_dir
    batch_size = args.batch_size
    print_message(logger, '开始MT-DNN训练')
    #return
    tasks = {}
    task_def_list = []
    dropout_list = []
    # 不是分布式，那么就打印
    printable = args.local_rank in [-1, 0]
    train_datasets = []
    # 初始化每个任务的数据集
    for dataset in args.train_datasets:
        prefix = dataset.split('_')[0]
        if prefix in tasks:
            continue
        task_id = len(tasks)
        tasks[prefix] = task_id
        #训练的基本数据信息，例如用哪个损失，任务类型，任务标签等
        task_def = task_defs.get_task_def(prefix)
        #把任务的名字也加进去
        task_def['name'] = prefix
        task_def_list.append(task_def)
        assert len(task_def.label_vocab.ind2tok) == task_def.n_class, "配置中的类别数量和标签数量不相等，请检查"
        train_path = os.path.join(data_dir, '{}_train.json'.format(dataset))
        print_message(logger, '加载训练任务 {}，训练任务的顺序id是： {}'.format(train_path, task_id))
        # 训练的数据的json文件， train_path = 'data_my/canonical_data/bert-base-chinese/absa_train.json'
        train_data_set = SingleTaskDataset(path=train_path, is_train=True, maxlen=args.max_seq_len, task_id=task_id, task_def=task_def, printable=printable)
        train_datasets.append(train_data_set)
    #Collater函数
    train_collater = Collater(dropout_w=args.dropout_w, encoder_type=encoder_type, soft_label=args.mkd_opt > 0, max_seq_len=args.max_seq_len, do_padding=args.do_padding)
    #把数据放到一起
    multi_task_train_dataset = MultiTaskDataset(train_datasets)
    if args.local_rank != -1:
        multi_task_batch_sampler = DistMultiTaskBatchSampler(train_datasets, args.batch_size, args.mix_opt, args.ratio, rank=args.local_rank, world_size=args.world_size)
    else:
        # 一个batch的数据集采用器
        multi_task_batch_sampler = MultiTaskBatchSampler(train_datasets, args.batch_size, args.mix_opt, args.ratio, bin_on=args.bin_on, bin_size=args.bin_size, bin_grow_ratio=args.bin_grow_ratio)
    # Dataloader格式
    multi_task_train_data = DataLoader(multi_task_train_dataset, batch_sampler=multi_task_batch_sampler, collate_fn=train_collater.collate_fn, pin_memory=args.cuda)
    # len(task_def_list)，里面包含几个task，长度就是几
    opt['task_def_list'] = task_def_list
    # 测试数据，同理
    dev_data_list = []
    test_data_list = []
    test_collater = Collater(is_train=False, encoder_type=encoder_type, max_seq_len=args.max_seq_len, do_padding=args.do_padding)
    for dataset in args.test_datasets:
        prefix = dataset.split('_')[0]
        task_def = task_defs.get_task_def(prefix)
        task_id = tasks[prefix]
        task_type = task_def.task_type
        data_type = task_def.data_type

        dev_path = os.path.join(data_dir, '{}_dev.json'.format(dataset))
        dev_data = None
        if os.path.exists(dev_path):
            dev_data_set = SingleTaskDataset(dev_path, False, maxlen=args.max_seq_len, task_id=task_id, task_def=task_def, printable=printable)
            if args.local_rank != -1:
                dev_data_set = DistTaskDataset(dev_data_set, task_id)
                single_task_batch_sampler = DistSingleTaskBatchSampler(dev_data_set, args.batch_size_eval, rank=args.local_rank, world_size=args.world_size)
                dev_data = DataLoader(dev_data_set, batch_sampler=single_task_batch_sampler, collate_fn=test_collater.collate_fn, pin_memory=args.cuda)
            else:
                dev_data = DataLoader(dev_data_set, batch_size=args.batch_size_eval, collate_fn=test_collater.collate_fn, pin_memory=args.cuda)
        dev_data_list.append(dev_data)

        test_path = os.path.join(data_dir, '{}_test.json'.format(dataset))
        test_data = None
        if os.path.exists(test_path):
            test_data_set = SingleTaskDataset(test_path, False, maxlen=args.max_seq_len, task_id=task_id, task_def=task_def, printable=printable)
            if args.local_rank != -1:
                test_data_set = DistTaskDataset(test_data_set, task_id)
                single_task_batch_sampler = DistSingleTaskBatchSampler(test_data_set, args.batch_size_eval, rank=args.local_rank, world_size=args.world_size)
                test_data = DataLoader(test_data_set, batch_sampler=single_task_batch_sampler, collate_fn=test_collater.collate_fn, pin_memory=args.cuda)
            else:
                test_data = DataLoader(test_data_set, batch_size=args.batch_size_eval, collate_fn=test_collater.collate_fn, pin_memory=args.cuda)
        test_data_list.append(test_data)
    # 打印默认参数
    print_message(logger, '#' * 20)
    print_message(logger, opt)
    print_message(logger, '#' * 20)

    # 需要除以grad accumulation，来计算一共需要多少个batch step
    num_all_batches = args.epochs * len(multi_task_train_data) // args.grad_accumulation_step
    print_message(logger, '############# Gradient Accumulation 信息 #############')
    print_message(logger, '原有训练的step数是: {}'.format(args.epochs * len(multi_task_train_data)))
    print_message(logger, '梯度度累积参数 grad_accumulation 为: {}'.format(args.grad_accumulation_step))
    print_message(logger, '经过梯度累积后的训练step数是: {}'.format(num_all_batches))
    print_message(logger, '############# Gradient Accumulation 信息 #############')
    #使用哪个模型初始化参数
    init_model = args.init_checkpoint
    state_dict = None
    # 加载模型参数，可选bert和roberta
    if os.path.exists(init_model):
        if encoder_type == EncoderModelType.BERT or \
            encoder_type == EncoderModelType.DEBERTA or \
            encoder_type == EncoderModelType.ELECTRA:
            state_dict = torch.load(init_model, map_location=device)
            config = state_dict['config']
        elif encoder_type == EncoderModelType.ROBERTA or encoder_type == EncoderModelType.XLM:
            model_path = '{}/model.pt'.format(init_model)
            state_dict = torch.load(model_path, map_location=device)
            arch = state_dict['args'].arch
            arch = arch.replace('_', '-')
            if encoder_type == EncoderModelType.XLM:
                arch = "xlm-{}".format(arch)
            # convert model arch
            from data_utils.roberta_utils import update_roberta_keys
            from data_utils.roberta_utils import patch_name_dict
            state = update_roberta_keys(state_dict['model'], nlayer=state_dict['args'].encoder_layers)
            state = patch_name_dict(state)
            literal_encoder_type = EncoderModelType(opt['encoder_type']).name.lower()
            config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_encoder_type]
            config = config_class.from_pretrained(arch).to_dict()
            state_dict = {'state': state}
    else:
        if opt['encoder_type'] not in EncoderModelType._value2member_map_:
            raise ValueError("encoder_type is out of pre-defined types")
        literal_encoder_type = EncoderModelType(opt['encoder_type']).name.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_encoder_type]
        config = config_class.from_pretrained(init_model).to_dict()
    # config是预训练模型的参数，设置一下，dropout默认0.1
    config['attention_probs_dropout_prob'] = args.bert_dropout_p
    config['hidden_dropout_prob'] = args.bert_dropout_p
    # 是否开启多GPU
    config['multi_gpu_on'] = opt["multi_gpu_on"]
    # 如果大于0，说明模型的修改隐藏层参数
    if args.num_hidden_layers > 0:
        config['num_hidden_layers'] = args.num_hidden_layers
    #更新下opt，用于保存所有参数
    opt.update(config)
    #MTDNN模型初始化
    model = MTDNNModel(opt, device=device, state_dict=state_dict, num_train_step=num_all_batches)
    # 是否是继续训练模型
    if args.resume and args.model_ckpt:
        print_message(logger, '选择了继续训练模型，并且模型{}也存在'.format(args.model_ckpt))
        model.load(args.model_ckpt)

    #### model meta str
    headline = '############# 打印 MT-DNN 模型的结果信息 #############'
    ### print network
    print_message(logger, '\n{}\n{}\n'.format(headline, model.network))

    #保存配置信息
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as writer:
        writer.write('{}\n'.format(json.dumps(opt)))
        writer.write('\n{}\n{}\n'.format(headline, model.network))
    print_message(logger, f"保存参数信息到{config_file}中")
    print_message(logger, "总的参数量是: {}".format(model.total_param))

    # tensorboard, 配置tensorboard
    tensorboard = None
    if args.tensorboard:
        args.tensorboard_logdir = os.path.join(args.output_dir, args.tensorboard_logdir)
        tensorboard = SummaryWriter(log_dir=args.tensorboard_logdir)
    #只编码测试数据并保存
    if args.encode_mode:
        for idx, dataset in enumerate(args.test_datasets):
            prefix = dataset.split('_')[0]
            test_data = test_data_list[idx]
            with torch.no_grad():
                encoding = extract_encoding(model, test_data, use_cuda=args.cuda)
            torch.save(encoding, os.path.join(output_dir, '{}_encoding.pt'.format(dataset)))
        return
    # 开始训练
    for epoch in range(0, args.epochs):
        print_message(logger, '开始训练Epoch: {}'.format(epoch), level=1)
        start = datetime.now()
        # batch_meta, 一个批次数据的元信息，就是基本信息， batch_data是一个批次的数据， colllater函数已经在enumerate时调用了，batch_data是mt_dnn下的batcher.py函数collate_fn返回的结果
        for i, (batch_meta, batch_data) in enumerate(multi_task_train_data):
            # batch_data包含的数据 token_ids, type_ids, masks, premise_masks(前提mask), hypothesis_masks(假设mask)，label， 前提和假设的mask只有在问答时有用，decoder_opt ==1 的时候是问答
            # 使用Collater的patch_data函数对一个批次的数据进一步处理，例如放到GPU上
            batch_meta, batch_data = Collater.patch_data(device, batch_meta, batch_data)
            task_id = batch_meta['task_id']
            task_name = batch_meta['task_def']['name']
            #模型训练
            model.update(batch_meta, batch_data)
            # 打印一些信息
            if (model.updates) % (args.log_per_updates) == 0 or model.updates == 1:
                ramaining_time = str((datetime.now() - start) / (i + 1) * (len(multi_task_train_data) - i - 1)).split('.')[0]
                if args.adv_train and args.debug:
                    debug_info = ' adv loss[%.5f] emb val[%.8f] eff_perturb[%.8f] ' % (
                        model.adv_loss.avg,
                        model.emb_val.avg,
                        model.eff_perturb.avg
                    )
                else:
                    debug_info = ' '
                print_message(logger, '任务ID:{0:1}，任务:{1}，训练了第[{2:6}]步， 训练损失为：[{3:.5f}]{4}，预计还需时间：[{5}]'.format(task_id,task_name,
                                                                                                    model.updates,
                                                                                                    model.train_loss.avg,
                                                                                                    debug_info,
                                                                                                    ramaining_time))
                if args.tensorboard:
                    tensorboard.add_scalar('train/loss', model.train_loss.avg, global_step=model.updates)

            # 评估和保存模型
            if args.save_per_updates_on and ((model.local_updates) % (args.save_per_updates * args.grad_accumulation_step) == 0) and args.local_rank in [-1, 0]:
                model_file = os.path.join(output_dir, 'model_{}_{}.pt'.format(epoch, model.updates))
                evaluation(model, args.test_datasets, dev_data_list, task_defs, output_dir, epoch, n_updates=args.save_per_updates, with_label=True, tensorboard=tensorboard, glue_format_on=args.glue_format_on, test_on=False, device=device, logger=logger)
                evaluation(model, args.test_datasets, test_data_list, task_defs, output_dir, epoch, n_updates=args.save_per_updates, with_label=False, tensorboard=tensorboard, glue_format_on=args.glue_format_on, test_on=True, device=device, logger=logger)
                print_message(logger, '每步都保存模型: {}'.format(model_file))
                model.save(model_file)

        evaluation(model, args.test_datasets, dev_data_list, task_defs, output_dir, epoch, with_label=True, tensorboard=tensorboard, glue_format_on=args.glue_format_on, test_on=False, device=device, logger=logger)
        evaluation(model, args.test_datasets, test_data_list, task_defs, output_dir, epoch, with_label=False, tensorboard=tensorboard, glue_format_on=args.glue_format_on, test_on=True, device=device, logger=logger)
        print_message(logger, '[new test scores at {} saved.]'.format(epoch))
        if args.local_rank in [-1, 0]:
            model_file = os.path.join(output_dir, 'model_{}.pt'.format(epoch))
            print_message(logger, 'epoch结束保存模型: {}'.format(model_file))
            model.save(model_file)
    # 保存最后的模型
    if args.local_rank in [-1, 0]:
        model_file = os.path.join(output_dir, 'model_final.pt')
        print_message(logger, '最终保存模型: {}'.format(model_file))
        model.save(model_file)
    if args.tensorboard:
        tensorboard.close()

if __name__ == '__main__':
    main()
