# This scripts is to convert Google's TF BERT to the pytorch version which is used by mt-dnn.
# It is a supplementary script.
# Note that it relies on tensorflow==1.12.0 which does not support by our released docker. 
# If you want to use this, please install tensorflow==1.12.0 by: pip install tensorflow==1.12.0
# Some codes are adapted from https://github.com/huggingface/pytorch-pretrained-BERT
# by: xiaodl
from __future__ import absolute_import
from __future__ import division
import re
import os
import argparse
import torch
import numpy as np
from pytorch_pretrained_bert.modeling import BertConfig
from transformers import ElectraConfig, AlbertConfig
from sys import path
path.append(os.getcwd())
from mt_dnn.matcher import SANBertNetwork
from data_utils.log_wrapper import create_logger

logger = create_logger(__name__, to_disk=False)
def model_config(parser):
    parser.add_argument('--update_bert_opt',  default=0, type=int)
    parser.add_argument('--multi_gpu_on', action='store_true')
    parser.add_argument('--mem_cum_type', type=str, default='simple',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_num_turn', type=int, default=5)
    parser.add_argument('--answer_mem_drop_p', type=float, default=0.1)
    parser.add_argument('--answer_att_hidden_size', type=int, default=128)
    parser.add_argument('--answer_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_rnn_type', type=str, default='gru',
                        help='rnn/gru/lstm')
    parser.add_argument('--answer_sum_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_merge_opt', type=int, default=1)
    parser.add_argument('--answer_mem_type', type=int, default=1)
    parser.add_argument('--answer_dropout_p', type=float, default=0.1)
    parser.add_argument('--answer_weight_norm_on', action='store_true')
    parser.add_argument('--dump_state_on', action='store_true')
    parser.add_argument('--answer_opt', type=int, default=0, help='0,1')
    parser.add_argument('--label_size', type=str, default='3')
    parser.add_argument('--mtl_opt', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--mix_opt', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--init_ratio', type=float, default=1)
    parser.add_argument('--encoder_type', type=int, default=1, help='1代表bert，参考data_utils/task_def.py')
    return parser

def train_config(parser):
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument('--log_per_updates', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd, adadelta, adam')
    parser.add_argument('--grad_clipping', type=float, default=0)
    parser.add_argument('--global_grad_clipping', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--warmup_schedule', type=str, default='warmup_linear')

    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--tasks_dropout_p', type=float, default=0.1)
    parser.add_argument('--dropout_w', type=float, default=0.000)
    parser.add_argument('--bert_dropout_p', type=float, default=0.1)
    parser.add_argument('--dump_feature', action='store_false')

    # EMA
    parser.add_argument('--ema_opt', type=int, default=0)
    parser.add_argument('--ema_gamma', type=float, default=0.995)

    # scheduler
    parser.add_argument('--have_lr_scheduler', dest='have_lr_scheduler', action='store_false')
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    parser.add_argument('--freeze_layers', type=int, default=-1)
    parser.add_argument('--embedding_opt', type=int, default=0)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--bert_l2norm', type=float, default=0.0)
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--output_dir', default='checkpoint')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed for data shuffling, embedding init, etc.')
    return parser


def convert(args, modeldirs):
    """
    转换torch模型到mtdnn模型
    :param args:
    :type args:
    :param modeldirs:
    :type modeldirs:
    :return:
    :rtype:
    """
    # 模型名称
    model_name = args.model_name
    #对应的基本信息
    checkpoint_path = modeldirs[model_name]['torch_checkpoint']
    config_name = modeldirs[model_name]['model_config_name']
    model_config_file = os.path.join(checkpoint_path, config_name)
    pytorch_dump_path = modeldirs[model_name]['save_path']
    model_bin_name = modeldirs[model_name]['model_bin_name']
    model_type = modeldirs[model_name]['model_type']
    # 对应的mtdnn类型
    """
    class EncoderModelType(IntEnum):
    BERT = 1
    ROBERTA = 2
    XLNET = 3
    SAN = 4
    XLM = 5
    DEBERTA = 6
    ELECTRA = 7
    T5 = 8"""
    if model_type == 'bert':
        config = BertConfig.from_json_file(model_config_file)
        args.encoder_type = 1
    elif model_type == 'electra':
        config = ElectraConfig.from_json_file(model_config_file)
        args.encoder_type = 7
    else:
        raise Exception("位置的模型类型")
    opt = vars(args)
    opt.update(config.to_dict())
    model = SANBertNetwork(opt, initial_from_local=True)
    bin_path = os.path.join(checkpoint_path, model_bin_name)
    logger.info('即将转换 checkpoint 从文件 {}中'.format(bin_path))
    state_dict = torch.load(bin_path, map_location='cpu')
    # 重新更改下原始的模型参数名称，要和SAN模型的参数名字匹配，这样才能加载，SAN模型把所有参数都改成了bert开头的参数
    state_weight = {k.replace(model_type,'bert'):v for k, v in state_dict.items()}
    missing_keys, _ = model.load_state_dict(state_weight, strict=False)
    print(f"missing_keys，注意丢失的参数{missing_keys}")
    assert len(missing_keys) < 10, '是不是丢失的参数太多了?'
    nstate_dict = model.state_dict()
    params = {'state':nstate_dict, 'config': config.to_dict()}
    torch.save(params, pytorch_dump_path)
    print(f"完成，保存模型到{pytorch_dump_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='把tf模型转换成torch的包含config配置的pt模型的路径')
    parser.add_argument('--model_name', type=str, default='electra', help='哪个模型名称')
    modeldirs = {
        "macbert": {
            "torch_checkpoint": '/Users/admin/git/TextBrewer/huazhuang/mac_bert_model',   #原始的torch的模型checkpoint的位置，这里是中文模型
            "model_config_name": 'config.json',   #原始的torch的模型checkpoint的位置中的模型配置名字
            "model_bin_name": 'pytorch_model.bin',   #原始的torch的模型checkpoint的位置中的模型配置名字
            "save_path": "mt_dnn_models/macbert.pt",   #保存模型pt文件到哪个位置
            "model_type": "bert"   #模型的类型
        },
        "bert": {
            "torch_checkpoint": '/Users/admin/git/TextBrewer/huazhuang/bert_model',
            # 原始的torch的模型checkpoint的位置，这里是中文模型
            "model_config_name": 'config.json',  # 原始的torch的模型checkpoint的位置中的模型配置名字
            "model_bin_name": 'pytorch_model.bin',  # 原始的torch的模型checkpoint的位置中的模型配置名字
            "save_path": "mt_dnn_models/bert.pt",  # 保存模型pt文件到哪个位置
            "model_type": "bert"  # 模型的类型
        },
        "electra": {
            "torch_checkpoint": '/Users/admin/git/TextBrewer/huazhuang/electra_model',  # 原始的torch的模型checkpoint的位置,这里是中文模型
            "model_config_name": 'config.json',  # 原始的torch的模型checkpoint的位置中的模型配置名字
            "model_bin_name": 'pytorch_model.bin',  # 原始的torch的模型checkpoint的位置中的模型配置名字
            "save_path": "mt_dnn_models/electra.pt",  # 保存模型pt文件到哪个位置
            "model_type": "electra"  # 模型的类型，
        },

    }
    # parser.add_argument('--torch_checkpoint', type=str, default='/Users/admin/git/TextBrewer/huazhuang/mac_bert_model', help='原始的torch的模型checkpoint的位置')
    # parser.add_argument('--save_path', type=str, default='mt_dnn_models/macbert.pt', help='保存模型pt文件到哪个位置')
    parser = model_config(parser)
    parser = train_config(parser)
    args = parser.parse_args()
    logger.info(args)
    convert(args, modeldirs)
