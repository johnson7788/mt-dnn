# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import copy
import sys
import torch
import tasks
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from data_utils.utils import AverageMeter
from pytorch_pretrained_bert import BertAdam as Adam
from module.bert_optim import Adamax, RAdam
from mt_dnn.loss import LOSS_REGISTRY
from mt_dnn.matcher import SANBertNetwork
from mt_dnn.perturbation import SmartPerturbation
from mt_dnn.loss import *
from data_utils.task_def import TaskType, EncoderModelType
from experiments.exp_def import TaskDef

logger = logging.getLogger(__name__)


class MTDNNModel(object):
    def __init__(self, opt, device=None, state_dict=None, num_train_step=-1):
        self.config = opt   #模型和我们设置的所有参数，dict，len：103个
        # 更新的步数，即训练了多少步
        self.updates = state_dict['updates'] if state_dict and 'updates' in state_dict else 0
        self.local_updates = 0
        self.device = device
        #初始化一个metric，平均值的metric，用于训练
        self.train_loss = AverageMeter()
        # 对抗训练的损失
        self.adv_loss = AverageMeter()
        # 保存embedding的值，没有加noise的embedding的向量
        self.emb_val = AverageMeter()
        self.eff_perturb = AverageMeter()
        self.initial_from_local = True if state_dict else False
        # 初始化一个随机答案网络,用于问答系统，stochastic answer network
        model = SANBertNetwork(opt, initial_from_local=self.initial_from_local)
        # 总的参数量: eg: 110075139
        self.total_param = sum([p.nelement() for p in model.parameters() if p.requires_grad])
        if opt['cuda']:
            if self.config['local_rank'] != -1:
                model = model.to(self.device)
            else:
                model = model.to(self.device)
        self.network = model
        if state_dict:
            # 检查丢失的参数
            missing_keys, unexpected_keys = self.network.load_state_dict(state_dict['state'], strict=False)
        # 优化器，部分参数学习率不需要decay
        optimizer_parameters = self._get_param_groups()
        #设置优化器
        self._setup_optim(optimizer_parameters, state_dict, num_train_step)
        self.optimizer.zero_grad()

        #if self.config["local_rank"] not in [-1, 0]:
        #    torch.distributed.barrier()
        # 分布式训练设置
        if self.config['local_rank'] != -1:
            self.mnetwork = torch.nn.parallel.DistributedDataParallel(self.network, device_ids=[self.config["local_rank"]], output_device=self.config["local_rank"], find_unused_parameters=True)
        elif self.config['multi_gpu_on']:
            self.mnetwork = nn.DataParallel(self.network)
        else:
            self.mnetwork = self.network
        #设置损失类型，例如交叉熵损失
        self._setup_lossmap(self.config)
        #设置回归损失，如果用户args存在的话， 蒸馏损失
        self._setup_kd_lossmap(self.config)
        #设置对抗训练的损失，如果配置中设置的话
        self._setup_adv_lossmap(self.config)
        # 设置对抗训练模型的初始化
        self._setup_adv_training(self.config)


    def _setup_adv_training(self, config):
        self.adv_teacher = None
        if config.get('adv_train', False):
            self.adv_teacher = SmartPerturbation(config['adv_epsilon'],
                    config['multi_gpu_on'],
                    config['adv_step_size'],
                    config['adv_noise_var'],
                    config['adv_p_norm'],
                    config['adv_k'],
                    config['fp16'],
                    config['encoder_type'],
                    loss_map=self.adv_task_loss_criterion,
                    norm_level=config['adv_norm_level'])


    def _get_param_groups(self):
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.network.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.network.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        return optimizer_parameters

    def _setup_optim(self, optimizer_parameters, state_dict=None, num_train_step=-1):
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(optimizer_parameters, self.config['learning_rate'],
                                       weight_decay=self.config['weight_decay'])

        elif self.config['optimizer'] == 'adamax':
            self.optimizer = Adamax(optimizer_parameters,
                                    self.config['learning_rate'],
                                    warmup=self.config['warmup'],
                                    t_total=num_train_step,
                                    max_grad_norm=self.config['grad_clipping'],
                                    schedule=self.config['warmup_schedule'],
                                    weight_decay=self.config['weight_decay'])
            if self.config.get('have_lr_scheduler', False): self.config['have_lr_scheduler'] = False
        elif self.config['optimizer'] == 'radam':
            self.optimizer = RAdam(optimizer_parameters,
                                    self.config['learning_rate'],
                                    warmup=self.config['warmup'],
                                    t_total=num_train_step,
                                    max_grad_norm=self.config['grad_clipping'],
                                    schedule=self.config['warmup_schedule'],
                                    eps=self.config['adam_eps'],
                                    weight_decay=self.config['weight_decay'])
            if self.config.get('have_lr_scheduler', False): self.config['have_lr_scheduler'] = False
            # The current radam does not support FP16.
            self.config['fp16'] = False
        elif self.config['optimizer'] == 'adam':
            self.optimizer = Adam(optimizer_parameters,
                                  lr=self.config['learning_rate'],
                                  warmup=self.config['warmup'],
                                  t_total=num_train_step,
                                  max_grad_norm=self.config['grad_clipping'],
                                  schedule=self.config['warmup_schedule'],
                                  weight_decay=self.config['weight_decay'])
            if self.config.get('have_lr_scheduler', False): self.config['have_lr_scheduler'] = False
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])

        if state_dict and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

        if self.config['fp16']:
            try:
                from apex import amp
                global amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(self.network, self.optimizer, opt_level=self.config['fp16_opt_level'])
            self.network = model
            self.optimizer = optimizer

        if self.config.get('have_lr_scheduler', False):
            if self.config.get('scheduler_type', 'rop') == 'rop':
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.config['lr_gamma'], patience=3)
            elif self.config.get('scheduler_type', 'rop') == 'exp':
                self.scheduler = ExponentialLR(self.optimizer, gamma=self.config.get('lr_gamma', 0.95))
            else:
                milestones = [int(step) for step in self.config.get('multi_step_lr', '10,20,30').split(',')]
                self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=self.config.get('lr_gamma'))
        else:
            self.scheduler = None

    def _setup_lossmap(self, config):
        task_def_list: List[TaskDef] = config['task_def_list']
        self.task_loss_criterion = []
        for idx, task_def in enumerate(task_def_list):
            cs = task_def.loss
            lc = LOSS_REGISTRY[cs](name='Loss func of task {}: {}'.format(idx, cs))
            self.task_loss_criterion.append(lc)

    def _setup_kd_lossmap(self, config):
        task_def_list: List[TaskDef] = config['task_def_list']
        self.kd_task_loss_criterion = []
        if config.get('mkd_opt', 0) > 0:
            for idx, task_def in enumerate(task_def_list):
                cs = task_def.kd_loss
                assert cs is not None
                lc = LOSS_REGISTRY[cs](name='KD Loss func of task {}: {}'.format(idx, cs))
                self.kd_task_loss_criterion.append(lc)

    def _setup_adv_lossmap(self, config):
        task_def_list: List[TaskDef] = config['task_def_list']
        self.adv_task_loss_criterion = []
        if config.get('adv_train', False):
            for idx, task_def in enumerate(task_def_list):
                cs = task_def.adv_loss
                assert cs is not None, "如果使用对抗训练，对抗损失不能为None"
                lc = LOSS_REGISTRY[cs](name='Adv Loss func of task {}: {}'.format(idx, cs))
                self.adv_task_loss_criterion.append(lc)

    def _to_cuda(self, tensor):
        if tensor is None: return tensor

        if isinstance(tensor, list) or isinstance(tensor, tuple):
            #y = [e.cuda(non_blocking=True) for e in tensor]
            y = [e.to(self.device) for e in tensor]
            for e in y:
                e.requires_grad = False
        else:
            #y = tensor.cuda(non_blocking=True)
            y = tensor.to(self.device)
            y.requires_grad = False
        return y

    def update(self, batch_meta, batch_data):
        """
        训练模型
        :param batch_meta: 这一批次训练数据的元信息，dict
        :type batch_meta: eg: {'token_id': 0, 'segment_id': 1, 'mask': 2, 'premise_mask': 3, 'hypothesis_mask': 4, 'task_id': 0, 'input_len': 5, 'task_def': {'label_vocab': <data_utils.vocab.Vocabulary object at 0x7fadc19e6550>, 'n_class': 3, 'data_type': <DataFormat.PremiseAndOneHypothesis: 2>, 'task_type': <TaskType.Classification: 1>, 'metric_meta': (<Metric.ACC: 0>,), 'split_names': ['train', 'matched_dev', 'mismatched_dev', 'matched_test', 'mismatched_test'], 'enable_san': False, 'dropout_p': 0.1, 'loss': <LossCriterion.CeCriterion: 0>, 'kd_loss': <LossCriterion.MseCriterion: 1>, 'adv_loss': <LossCriterion.SymKlCriterion: 7>}, 'pairwise_size': 1, 'label': 5, 'uids': ['1632', '1633', '1634', '1635', '1636', '1637', '1638', '1639']}
        :param batch_data:  一个批次的数据的向量信息集合[[token_id*batch_size],[type_id * batch_size],[attention_mask * batch_size],....]
        :type batch_data: 包含的数据包括： token_ids, type_ids, masks, premise_masks(前提mask), hypothesis_masks(假设mask)，label
        :return:
        :rtype:
        """
        #设置模型为train模式
        self.network.train()
        # 获取label的向量，eg: tensor([0, 0, 1, 2, 2, 2, 2, 2])
        y = batch_data[batch_meta['label']]
        # 重新检查下，是否放到gpu
        y = self._to_cuda(y) if self.config['cuda'] else y
        # 任务的序号task_id eg：0
        task_id = batch_meta['task_id']
        # 获取输入数据向量
        inputs = batch_data[:batch_meta['input_len']]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)
        weight = None
        if self.config.get('weighted_on', False):
            if self.config['cuda']:
                weight = batch_data[batch_meta['factor']].cuda(non_blocking=True)
            else:
                weight = batch_data[batch_meta['factor']]

        # fw to get logits， 真正的把数据放入模型，开始前向网络
        logits = self.mnetwork(*inputs)

        # compute loss, 每个任务的标准损失
        loss = 0
        if self.task_loss_criterion[task_id] and (y is not None):
            loss_criterion = self.task_loss_criterion[task_id]
            if isinstance(loss_criterion, RankCeCriterion) and batch_meta['pairwise_size'] > 1:
                # reshape the logits for ranking.
                loss = self.task_loss_criterion[task_id](logits, y, weight, ignore_index=-1, pairwise_size=batch_meta['pairwise_size'])
            else:
                loss = self.task_loss_criterion[task_id](logits, y, weight, ignore_index=-1)

        # 如果有其它损失，也计算其它损失，compute kd loss， 蒸馏学习损失
        if self.config.get('mkd_opt', 0) > 0 and ('soft_label' in batch_meta):
            soft_labels = batch_meta['soft_label']
            soft_labels = self._to_cuda(soft_labels) if self.config['cuda'] else soft_labels
            kd_lc = self.kd_task_loss_criterion[task_id]
            kd_loss = kd_lc(logits, soft_labels, weight, ignore_index=-1) if kd_lc else 0
            loss = loss + kd_loss

        # 对抗学习 training
        if self.config.get('adv_train', False) and self.adv_teacher:
            # task info， 任务类型 TaskType.Classification
            task_type = batch_meta['task_def']['task_type']
            # 对抗学习的输入
            adv_inputs = [self.mnetwork, logits] + inputs + [task_type, batch_meta.get('pairwise_size', 1)]
            adv_loss, emb_val, eff_perturb = self.adv_teacher.forward(*adv_inputs)
            # 损失就是2部分损失，根据ALUM论文的公式3，缩放的对抗损失和原始损失
            loss = loss + self.config['adv_alpha'] * adv_loss
        # eg：8
        batch_size = batch_data[batch_meta['token_id']].size(0)
        # rescale loss as dynamic batching
        if self.config['bin_on']:
            loss = loss * (1.0 * batch_size / self.config['batch_size'])
        if self.config['local_rank'] != -1:
            #print('Rank ', self.config['local_rank'], ' loss ', loss)
            copied_loss = copy.deepcopy(loss.data)
            torch.distributed.all_reduce(copied_loss)
            copied_loss = copied_loss / self.config['world_size']
            self.train_loss.update(copied_loss.item(), batch_size)
        else:
            # 更新统计信息
            self.train_loss.update(loss.item(), batch_size)
            
        if self.config.get('adv_train', False) and self.adv_teacher:
            if self.config['local_rank'] != -1:
                copied_adv_loss = copy.deepcopy(adv_loss.data)
                torch.distributed.all_reduce(copied_adv_loss)
                copied_adv_loss = copied_adv_loss / self.config['world_size']
                self.adv_loss.update(copied_adv_loss.item(), batch_size)

                copied_emb_val = copy.deepcopy(emb_val.data)
                torch.distributed.all_reduce(copied_emb_val)
                copied_emb_val = copied_emb_val / self.config["world_size"]
                self.emb_val.update(copied_emb_val.item(), batch_size)               

                copied_eff_perturb = copy.deepcopy(eff_perturb.data)
                torch.distributed.all_reduce(copied_eff_perturb)
                copied_eff_perturb = copied_eff_perturb / self.config["world_size"]
                self.eff_perturb.update(copied_eff_perturb.item(), batch_size)               
            else:
                self.adv_loss.update(adv_loss.item(), batch_size)
                self.emb_val.update(emb_val.item(), batch_size)
                self.eff_perturb.update(eff_perturb.item(), batch_size)

        # scale loss
        loss = loss / self.config.get('grad_accumulation_step', 1)
        if self.config['fp16']:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.local_updates += 1
        if self.local_updates % self.config.get('grad_accumulation_step', 1) == 0:
            if self.config['global_grad_clipping'] > 0:
                if self.config['fp16']:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer),
                                                   self.config['global_grad_clipping'])
                else:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                  self.config['global_grad_clipping'])
            self.updates += 1
            # reset number of the grad accumulation
            self.optimizer.step()
            self.optimizer.zero_grad()

    def encode(self, batch_meta, batch_data):
        self.network.eval()
        inputs = batch_data[:3]
        sequence_output = self.network.encode(*inputs)[0]
        return sequence_output

    # TODO: similar as function extract, preserve since it is used by extractor.py
    # will remove after migrating to transformers package
    def extract(self, batch_meta, batch_data):
        self.network.eval()
        # 'token_id': 0; 'segment_id': 1; 'mask': 2
        inputs = batch_data[:3]
        all_encoder_layers, pooled_output = self.mnetwork.bert(*inputs)
        return all_encoder_layers, pooled_output

    def predict(self, batch_meta, batch_data, full_score=False, softmax=True, both_softmax_logits=False):
        """

        :param batch_meta:
        :type batch_meta:
        :param batch_data:
        :type batch_data:
        :param full_score: score返回一个列表，返回所有的score
        :type full_score:
        :return:
        :rtype:
        """
        self.network.eval()
        task_id = batch_meta['task_id']
        task_def = TaskDef.from_dict(batch_meta['task_def'])
        task_type = task_def.task_type
        task_obj = tasks.get_task_obj(task_def)
        inputs = batch_data[:batch_meta['input_len']]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)
        score = self.mnetwork(*inputs)
        if task_obj is not None:
            score, predict = task_obj.test_predict(score, full_score, softmax, both_softmax_logits)
        elif task_type == TaskType.Ranking:
            score = score.contiguous().view(-1, batch_meta['pairwise_size'])
            assert task_type == TaskType.Ranking
            score = F.softmax(score, dim=1)
            score = score.data.cpu()
            score = score.numpy()
            predict = np.zeros(score.shape, dtype=int)
            positive = np.argmax(score, axis=1)
            for idx, pos in enumerate(positive):
                predict[idx, pos] = 1
            predict = predict.reshape(-1).tolist()
            score = score.reshape(-1).tolist()
            return score, predict, batch_meta['true_label']
        elif task_type == TaskType.SeqenceLabeling:
            mask = batch_data[batch_meta['mask']]
            score = score.contiguous()
            score = score.data.cpu()
            score = score.numpy()
            predict = np.argmax(score, axis=1).reshape(mask.size()).tolist()
            valied_lenght = mask.sum(1).tolist()
            final_predict = []
            for idx, p in enumerate(predict):
                final_predict.append(p[: valied_lenght[idx]])
            score = score.reshape(-1).tolist()
            return score, final_predict, batch_meta['label']
        elif task_type == TaskType.Span:
            start, end = score
            predictions = []
            if self.config['encoder_type'] == EncoderModelType.BERT:
                import experiments.squad.squad_utils as mrc_utils
                scores, predictions = mrc_utils.extract_answer(batch_meta, batch_data, start, end, self.config.get('max_answer_len', 5), do_lower_case=self.config.get('do_lower_case', False))
            return scores, predictions, batch_meta['answer']
        else:
            raise ValueError("Unknown task_type: %s" % task_type)
        return score, predict, batch_meta['label']

    def save(self, filename):
        if isinstance(self.mnetwork, torch.nn.parallel.DistributedDataParallel):
            model = self.mnetwork.module
        else:
            model = self.network
        #network_state = dict([(k, v.cpu()) for k, v in self.network.state_dict().items()])
        network_state = dict([(k, v.cpu()) for k, v in model.state_dict().items()])
        params = {
            'state': network_state,
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
        }
        torch.save(params, filename)
        logger.info('model saved to {}'.format(filename))

    def load(self, checkpoint):
        model_state_dict = torch.load(checkpoint)
        if 'state' in model_state_dict:
            self.network.load_state_dict(model_state_dict['state'], strict=False)
        if 'optimizer' in model_state_dict:
            self.optimizer.load_state_dict(model_state_dict['optimizer'])
        if 'config' in model_state_dict:
            self.config.update(model_state_dict['config'])

    def cuda(self):
        self.network.cuda()
