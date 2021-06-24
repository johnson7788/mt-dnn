[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Travis-CI](https://travis-ci.org/namisan/mt-dnn.svg?branch=master)](https://github.com/namisan/mt-dnn)

**New Release** <br/>
我们发布了LM预训练/FineTuning和f-divercence的对抗性训练。


Large-scale Adversarial training for LMs: [ALUM code](https://github.com/namisan/mt-dnn/blob/master/alum/README.md). <br/>
如果你想使用旧版本，请使用以下cmd来克隆代码。<br/>
```git clone -b v0.1 https://github.com/namisan/mt-dnn.git ```



# 用于自然语言理解的多任务深神经网络

这个PyTorch包实现了用于自然语言理解的多任务深度神经网络(MT-DNN)，如图所示。

Xiaodong Liu\*, Pengcheng He\*, Weizhu Chen and Jianfeng Gao<br/>
用于自然语言理解的多任务深神经网络，论文 <br/>
Multi-Task Deep Neural Networks for Natural Language Understanding
[ACL 2019](https://aclweb.org/anthology/papers/P/P19/P19-1441/) <br/>
\*: Equal contribution <br/>

Xiaodong Liu, Pengcheng He, Weizhu Chen and Jianfeng Gao<br/>
通过自然语言理解的知识蒸馏改进多任务深度神经网络 <br/>
Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding
[arXiv version](https://arxiv.org/abs/1904.09482) <br/>


Pengcheng He, Xiaodong Liu, Weizhu Chen and Jianfeng Gao<br/>
Commonsense推理的混合神经网络模型<br/> 
Hybrid Neural Network Model for Commonsense Reasoning <br/>
[arXiv version](https://arxiv.org/abs/1907.11983) <br/>


Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao and Jiawei Han <br/>
On the Variance of the Adaptive Learning Rate and Beyond <br/>
关于自适应学习率及超越的方差<br/>
[arXiv version](https://arxiv.org/abs/1908.03265) <br/>

Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao and Tuo Zhao <br/>
SMART: 通过原则的正则化优化对预训练的自然语言模型进行鲁棒和高效的微调 <br/>
SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization <br/>
[arXiv version](https://arxiv.org/abs/1911.03437) <br/>

Xiaodong Liu, Yu Wang, Jianshu Ji, Hao Cheng, Xueyun Zhu, Emmanuel Awa, Pengcheng He, Weizhu Chen, Hoifung Poon, Guihong Cao, Jianfeng Gao<br/>
用于自然语言理解的微软多任务深度神经网络工具包 <br/>
The Microsoft Toolkit of Multi-Task Deep Neural Networks for Natural Language Understanding <br/>
[arXiv version](https://arxiv.org/abs/2002.07972) <br/>

Xiaodong Liu, Hao Cheng, Pengcheng He, Weizhu Chen, Yu Wang, Hoifung Poon and Jianfeng Gao<br/>
大型神经语言模型的对抗训练<br/> 
Adversarial Training for Large Neural Language Models <br/>
[arXiv version](https://arxiv.org/abs/2004.08994) <br/>

Hao Cheng and Xiaodong Liu and Lis Pereira and Yaoliang Yu and Jianfeng Gao<br/>
改善模型鲁棒性的f-divergence后微分正则化<br/> 
Posterior Differential Regularization with f-divergence for Improving Model Robustness <br/>
[arXiv version](https://arxiv.org/abs/2010.12638) <br/>


## Quickstart

### 配置环境
#### Install via pip:
1. python3.6 </br>
   安装python3.6 : https://www.python.org/downloads/release/python-360/

2. 安装依赖 </br>
   ```> pip install -r requirements.txt```

#### Use docker:
1. Pull docker </br>
   ```> docker pull allenlao/pytorch-mt-dnn:v0.5```

2. Run docker </br>
   ```> docker run -it --rm --runtime nvidia  allenlao/pytorch-mt-dnn:v0.5 bash``` </br>
   Please refer to the following link if you first use docker: https://docs.docker.com/

### 训练一个迷你的MT-DNN
1. 下载数据，包括bert模型，roberta模型，MT-DNN模型，GLUE数据集，SciTail，SQuAD，NER数据集 </br>
   ```> sh download.sh all``` </br>
   更多详情，请参阅下载GLUE数据集 : https://gluebenchmark.com/
```buildoutcfg
# mt_dnn_models下的内容
tree mt_dnn_models/
mt_dnn_models/
├── bert_model_base_chinese.pt
├── bert_model_base_uncased.pt
├── bert_model_large_uncased.pt
├── mt_dnn_base_uncased.pt
├── mt_dnn_kd_large_cased.pt
├── mt_dnn_large_uncased.pt
├── roberta
│   ├── encoder.json
│   ├── ict.txt
│   └── vocab.bpe
├── roberta.base
│   ├── NOTE
│   ├── dict.txt
│   └── model.pt
└── roberta.large
    ├── NOTE
    ├── dict.txt
    └── model.pt

3 directories, 15 files

data下的目录内容
#tree data
.
├── CoLA
│   ├── dev.tsv
│   ├── original
│   │   ├── raw
│   │   │   ├── in_domain_dev.tsv
│   │   │   ├── in_domain_train.tsv
│   │   │   └── out_of_domain_dev.tsv
│   │   └── tokenized
│   │       ├── in_domain_dev.tsv
│   │       ├── in_domain_train.tsv
│   │       └── out_of_domain_dev.tsv
│   ├── test.tsv
│   └── train.tsv
├── MNLI
│   ├── README.txt
│   ├── dev_matched.tsv
│   ├── dev_mismatched.tsv
│   ├── diagnostic-full.tsv
│   ├── diagnostic.tsv
│   ├── original
│   │   ├── multinli_1.0_dev_matched.jsonl
│   │   ├── multinli_1.0_dev_matched.txt
│   │   ├── multinli_1.0_dev_mismatched.jsonl
│   │   ├── multinli_1.0_dev_mismatched.txt
│   │   ├── multinli_1.0_train.jsonl
│   │   └── multinli_1.0_train.txt
│   ├── test_matched.tsv
│   ├── test_mismatched.tsv
│   └── train.tsv
├── MRPC
│   ├── dev.tsv
│   ├── dev_ids.tsv
│   ├── msr_paraphrase_test.txt
│   ├── msr_paraphrase_train.txt
│   ├── test.tsv
│   └── train.tsv
├── QNLI
│   ├── dev.tsv
│   ├── test.tsv
│   └── train.tsv
├── QQP
│   ├── dev.tsv
│   ├── test.tsv
│   └── train.tsv
├── RTE
│   ├── dev.tsv
│   ├── test.tsv
│   └── train.tsv
├── SNLI
│   ├── README.txt
│   ├── dev.jsonl
│   ├── dev.tsv
│   ├── original
│   │   ├── snli_1.0_dev.txt
│   │   ├── snli_1.0_test.txt
│   │   └── snli_1.0_train.txt
│   ├── test.jsonl
│   ├── test.tsv
│   ├── train.jsonl
│   └── train.tsv
├── SST-2
│   ├── dev.tsv
│   ├── original
│   │   ├── README.txt
│   │   ├── SOStr.txt
│   │   ├── STree.txt
│   │   ├── datasetSentences.txt
│   │   ├── datasetSplit.txt
│   │   ├── dictionary.txt
│   │   ├── original_rt_snippets.txt
│   │   └── sentiment_labels.txt
│   ├── test.tsv
│   └── train.tsv
├── STS-B
│   ├── LICENSE.txt
│   ├── dev.tsv
│   ├── original
│   │   ├── sts-dev.tsv
│   │   ├── sts-test.tsv
│   │   └── sts-train.tsv
│   ├── readme.txt
│   ├── test.tsv
│   └── train.tsv
├── SciTail
│   ├── README.txt
│   ├── all_annotations.tsv
│   ├── dgem_format
│   │   ├── README.txt
│   │   ├── scitail_1.0_structure_dev.tsv
│   │   ├── scitail_1.0_structure_test.tsv
│   │   └── scitail_1.0_structure_train.tsv
│   ├── predictor_format
│   │   ├── README.txt
│   │   ├── scitail_1.0_structure_dev.jsonl
│   │   ├── scitail_1.0_structure_test.jsonl
│   │   └── scitail_1.0_structure_train.jsonl
│   ├── snli_format
│   │   ├── README.txt
│   │   ├── scitail_1.0_dev.txt
│   │   ├── scitail_1.0_test.txt
│   │   └── scitail_1.0_train.txt
│   └── tsv_format
│       ├── scitail_1.0_dev.tsv
│       ├── scitail_1.0_test.tsv
│       └── scitail_1.0_train.tsv
├── WNLI
│   ├── dev.tsv
│   ├── test.tsv
│   └── train.tsv
├── domain_adaptation
│   ├── scitail_001_train.json
│   ├── scitail_01_train.json
│   ├── scitail_1_train.json
│   ├── scitail_5_train.json
│   ├── scitail_dev.json
│   ├── scitail_test.json
│   ├── scitail_train.json
│   ├── scitail_train_shuff.json
│   ├── snli_001_train.json
│   ├── snli_01_train.json
│   ├── snli_1_train.json
│   ├── snli_5_train.json
│   ├── snli_dev.json
│   ├── snli_test.json
│   ├── snli_train.json
│   └── snli_train_shuff.json
├── ner
│   ├── test.txt
│   ├── train.txt
│   └── valid.txt
├── squad
│   ├── dev.json
│   └── train.json
└── squad_v2
    ├── dev.json
    └── train.json

26 directories, 110 files
```

2. 预处理数据 </br>
```
#规范化数据，把GLUE数据变成标准格式，即数据预处理, 保存到 /data/canonical_data目录下, ls canonical_data/ | wc -l #共有处理好的数据35个
python experiments/glue/glue_prepro.py   
# 数据预处理tokenizer化，保存到对应数据下的对应模型下，如下，会保存到data/canonical_data/bert-large-uncased/目录下，格式为json文件，包含uid，label，和token_id, type_id, attention_mask
python prepro_std.py --model bert-large-uncased --root_dir data/canonical_data --task_def experiments/glue/glue_task_def.yml --do_lower_case
#同样生成35个处理好的文件,json格式
ls mt-dnn/data/canonical_data/bert-large-uncased/ | wc -l
```

3. 训练模型 </br>
   ```> python train.py --data_dir data/canonical_data/bert-large-uncased```

**请注意，我们在4个V100 GPU上进行了基础MT-DNN模型的实验。你可能需要减少其他GPU的批次大小。** <br/>

### GLUE结果重现 
1. MTL refinement: 改进 MT-DNN（共享层），用预训练好的BERT模型初始化，通过MTL使用所有GLUE任务（不包括WNLI）来学习新的共享表示。
**请注意，我们在8个V100 GPU（32G）上运行这个实验，批次大小为32。**
   + 通过上述脚本预处理GLUE数据
   + Training: </br>
   ```>scripts\run_mt_dnn.sh```

2. 微调：根据GLUE的每项任务对MT-DNN进行微调，以获得特定任务的模型。 </br>
这里，我们提供了两个例子，STS-B和RTE。你可以使用类似的脚本来微调所有的GLUE任务。 </br>
   + Finetune在STS-B任务上 </br>
   ```> scripts\run_stsb.sh``` </br>
   就Pearson/Spearman相关性而言，你应该在STS-B dev上得到大约90.5/90.4。</br>
   + Finetune在RTE任务上 </br>
   ```> scripts\run_rte.sh``` </br>
   就准确性而言，你应该在RTE dev上得到约83.8的结果。</br>

### SCITAIL和SNIL结果重现（域适应） 
1. SCITAIL上的域适应  </br>
   ```>scripts\scitail_domain_adaptation_bash.sh```

2. Snli上的域适应 </br>
  ```>scripts\snli_domain_adaptation_bash.sh```

### 序列标注任务 
1. 预处理数据 </br>
   a) 将ner数据下载到data/ner，包括：{train/valid/test}.txt </br>
   b) 将ner数据转换为规范格式：```> python experiments\ner\prepro.py --data data\ner --output_dir data\canonical_data``` </br>
   c) 预处理规范数据到MT-DNN格式：```> python prepro_std.py --do_lower_case --root_dir data\canonical_data --task_def experiments\ner\ner_task_def.yml --model bert-base-uncased``` </br>

2. 训练 </br>
   ```> python train.py --data_dir <data-path> --init_checkpoint <bert/ner-model> --train_dataset ner --test_dataset ner --task_def experiments\ner\ner_task_def.yml```

### SMART
在微调阶段的对抗训练：:
   ```> python train.py --data_dir <data-path> --init_checkpoint <bert/mt-dnn-model> --train_dataset mnli --test_dataset mnli_matched,mnli_mismatched --task_def experiments\glue\glue_task_def.yml --adv_train --adv_opt 1```


### HNN
重现HNN的代码在`hnn`文件夹下，要重现HNN的结果，运行
```> hnn/script/hnn_train_large.sh```


### Extract embeddings
1. 提取对文本样本的嵌入 </br>
   ```>python extractor.py --do_lower_case --finput input_examples\pair-input.txt --foutput input_examples\pair-output.json --bert_model bert-base-uncased --checkpoint mt_dnn_models\mt_dnn_base.pt``` </br>
   请注意，这对文本是由一个特殊的token分割的 ```|||```. You may refer ``` input_examples\pair-output.json``` as example. </br>

2. 提取单个句子样本的嵌入 </br>
   ```>python extractor.py  --do_lower_case --finput input_examples\single-input.txt --foutput input_examples\single-output.json --bert_model bert-base-uncased --checkpoint mt_dnn_models\mt_dnn_base.pt``` </br>


### 加速训练
1. 梯度累积Gradient Accumulation </br>
   如果你有小的GPU，你可能需要使用梯度累加来使训练稳定。 </br>
   例如，如果你在训练过程中使用标志：```--grad_accumulation_step 4 ```，那么实际的批次大小将是``` batch_size * 4 ```。</br>

2. FP16
   当前版本的MT-DNN也支持FP16训练，请安装apex。</br>
   您只需要在训练期间使用 ```--fp16 ```  </br>
Please refer the script: ``` scripts\run_mt_dnn_gc_fp16.sh```



### 将Tensorflow Bert模型转换为MT-DNN格式
在这里，我们将通过如何将一个中文的Tensorflow BERT模型转换为mt-dnn格式。<br/>
1. 从Google Bert Web下载Bert Model：https://github.com/google-research/bert <br/>

2. 为MT-DNN格式运行以下脚本 </br>
   ```python scripts\convert_tf_to_pt.py --tf_checkpoint_root chinese_L-12_H-768_A-12\ --pytorch_checkpoint_path chinese_L-12_H-768_A-12\bert_base_chinese.pt```

### TODO
- [ ] Publish pretrained Tensorflow checkpoints. <br/>


## FAQ

### 您是否分享了预训练的MT-DNN模型？ 
是的，我们通过MTL发布了预训练的共享嵌入，这些嵌入与BERT基础/大型模型一致。``mt_dnn_base.pt``和``mt_dnn_large.pt``。</br>
获取类似模型： 
1. 运行 ```>sh scripts\run_mt_dnn.sh```, 然后根据MNLI/RTE的平均开发性能挑选最佳checkpoint。 </br>
2. 通过剥离特定任务的层 ```scritps\strip_model.py```. </br>

### 为什么SCITAIL/SNLI不启用SAN网络
对于SciTail/SNLI任务，目的是测试学到的嵌入的通用性，以及它是如何容易适应新的领域，而不是复杂的模型结构，以便与BERT直接比较。因此，我们在所有的**域适应**设置上使用线性投影。

### V1和V2之间有什么区别
区别在于QNLI数据集。更多细节请参考GLUE官方主页。如果你想像我们的论文一样将QNLI作为pair-wise排名任务，请确保你使用旧的QNLI数据。</br>
然后使用flag运行prepro脚本：   ```> sh experiments/glue/prepro.sh --old_glue``` </br>
如果您在访问旧版数据时遇到问题，请联系GLUE团队。

### 您是否为您的GLUE排行榜提交进行了微调单次任务？
我们可以使用多任务细化模型来运行预测并产生一个合理的结果。但要达到一个更好的结果，需要对每个任务进行微调。值得注意的是，arxiv中的论文已经有点过时了，而且是在旧的GLUE数据集上。我们将按下文所述更新该论文。

## Notes and Acknowledgments
BERT pytorch is from: https://github.com/huggingface/pytorch-pretrained-BERT <br/>
BERT: https://github.com/google-research/bert <br/>
We also used some code from: https://github.com/kevinduh/san_mrc <br/>

## Related Projects/Codebase
1. Pretrained UniLM: https://github.com/microsoft/unilm <br/>
2. Pretrained Response Generation Model: https://github.com/microsoft/DialoGPT <br/>
3. Internal MT-DNN repo: https://github.com/microsoft/mt-dnn <br/>

### How do I cite MT-DNN?

```
@inproceedings{liu2019mt-dnn,
    title = "Multi-Task Deep Neural Networks for Natural Language Understanding",
    author = "Liu, Xiaodong and He, Pengcheng and Chen, Weizhu and Gao, Jianfeng",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1441",
    pages = "4487--4496"
}


@article{liu2019mt-dnn-kd,
  title={Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding},
  author={Liu, Xiaodong and He, Pengcheng and Chen, Weizhu and Gao, Jianfeng},
  journal={arXiv preprint arXiv:1904.09482},
  year={2019}
}


@article{he2019hnn,
  title={A Hybrid Neural Network Model for Commonsense Reasoning},
  author={He, Pengcheng and Liu, Xiaodong and Chen, Weizhu and Gao, Jianfeng},
  journal={arXiv preprint arXiv:1907.11983},
  year={2019}
}


@article{liu2019radam,
  title={On the Variance of the Adaptive Learning Rate and Beyond},
  author={Liu, Liyuan and Jiang, Haoming and He, Pengcheng and Chen, Weizhu and Liu, Xiaodong and Gao, Jianfeng and Han, Jiawei},
  journal={arXiv preprint arXiv:1908.03265},
  year={2019}
}


@article{jiang2019smart,
  title={SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization},
  author={Jiang, Haoming and He, Pengcheng and Chen, Weizhu and Liu, Xiaodong and Gao, Jianfeng and Zhao, Tuo},
  journal={arXiv preprint arXiv:1911.03437},
  year={2019}
}


@article{liu2020mtmtdnn,
  title={The Microsoft Toolkit of Multi-Task Deep Neural Networks for Natural Language Understanding},
  author={Liu, Xiaodong and Wang, Yu and Ji, Jianshu and Cheng, Hao and Zhu, Xueyun and Awa, Emmanuel and He, Pengcheng and Chen, Weizhu and Poon, Hoifung and Cao, Guihong and Jianfeng Gao},
  journal={arXiv preprint arXiv:2002.07972},
  year={2020}
}


@article{liu2020alum,
  title={Adversarial Training for Large Neural Language Models},
  author={Liu, Xiaodong and Cheng, Hao and He, Pengcheng and Chen, Weizhu and Wang, Yu and Poon, Hoifung and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2004.08994},
  year={2020}
}

@article{cheng2020posterior,
  title={Posterior Differential Regularization with f-divergence for Improving Model Robustness},
  author={Cheng, Hao and Liu, Xiaodong and Pereira, Lis and Yu, Yaoliang and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2010.12638},
  year={2020}
}
```
### Contact Information

For help or issues using MT-DNN, please submit a GitHub issue.

For personal communication related to this package, please contact Xiaodong Liu (`xiaodl@microsoft.com`), Yu Wang (`yuwan@microsoft.com`), Pengcheng He (`penhe@microsoft.com`), Weizhu Chen (`wzchen@microsoft.com`), Jianshu Ji (`jianshuj@microsoft.com`), Hao Cheng (`chehao@microsoft.com`) or Jianfeng Gao (`jfgao@microsoft.com`).
