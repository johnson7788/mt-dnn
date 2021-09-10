## 目录结构
```angular2html
.
├── LICENSE
├── README.md
├── README_CN.md
├── README_DIR.md
├── README_myexample.md
├── README_paper.md
├── README_varify.md
├── alum                    用于从头预训练模型和继续训练模型
│   ├── README.md
│   ├── __init__.py
│   ├── adv_masked_lm.py
│   ├── adv_masked_lm_task.py
│   ├── adv_model
│   │   ├── __init__.py
│   │   ├── alignment_utils.py
│   │   ├── hub_interface.py
│   │   └── model.py
│   └── alum_train.sh   训练脚本
├── calc_metrics.py     计算预测后的模型结果和gold-label之间的metric
├── checkpoint  保存训练好的模型
│   └── config.json
├── data    原始数据目录
├── data_utils   数据处理
│   ├── __init__.py
│   ├── gpt2_bpe.py
│   ├── log_wrapper.py
│   ├── metrics.py
│   ├── mrc_eval.py
│   ├── roberta_utils.py
│   ├── squad_eval.py
│   ├── task_def.py
│   ├── task_def.pyc
│   ├── utils.py
│   ├── vocab.py
│   └── xlnet_utils.py
├── docker
│   └── Dockerfile    做成docker
├── download.sh  下载数据集和训练好的模型
├── experiments   各种任务和任务配置文件
│   ├── __init__.py
│   ├── common_utils.py
│   ├── common_utils.pyc
│   ├── exp_def.py
│   ├── glue
│   │   ├── generate_task_def.py
│   │   ├── glue_label_map.py
│   │   ├── glue_prepro.log
│   │   ├── glue_prepro.py
│   │   ├── glue_task_def.yml
│   │   ├── glue_utils.py
│   │   └── prepro.sh
│   ├── mlm
│   │   ├── mlm.yml
│   │   └── mlm_utils.py
│   ├── myexample
│   │   ├── __init__.py
│   │   ├── my_task_def.yml
│   │   ├── mydata_prepro.log
│   │   └── mydata_prepro.py
│   ├── mypaper
│   │   ├── paper_prepro.log
│   │   ├── paper_prepro.py
│   │   └── paper_task_def.yml
│   ├── ner
│   │   ├── ner_task_def.yml
│   │   ├── ner_task_def_roberta.yml
│   │   ├── ner_utils.py
│   │   └── prepro.py
│   ├── squad
│   │   ├── __init__.py
│   │   ├── squad_prepro.py
│   │   ├── squad_task_def.yml
│   │   ├── squad_utils.py
│   │   └── verify_calc_span.py
│   └── xnli
│       ├── README.md
│       ├── extract_cat.py
│       ├── xnli_eval.py
│       ├── xnli_prepro.py
│       └── xnli_task_def.yml
├── extractor.py     提取对文本样本的嵌入
├── filter_wrong.py   过滤错误样本
├── hnn 常识推理的混合网络模型
│   ├── README.md
│   ├── bert
│   │   └── uncased_L-24_H-1024_A-16
│   │       └── vocab.txt
│   ├── data
│   │   ├── DPRD
│   │   │   ├── convert.py
│   │   │   ├── resource.txt
│   │   │   ├── test.c.txt
│   │   │   ├── test_annotated.tsv
│   │   │   ├── train.c.txt
│   │   │   └── train_annotated.tsv
│   │   ├── WNLI
│   │   │   ├── dev_annotated.tsv
│   │   │   ├── extract_candidate.py
│   │   │   ├── ranking_data.py
│   │   │   ├── test_annotated.tsv
│   │   │   └── train_annotated.tsv
│   │   └── WSC273
│   │       ├── WSCollection.xml
│   │       ├── associative.tsv
│   │       ├── convert.py
│   │       ├── non_associative.tsv
│   │       ├── pdp60.tsv
│   │       ├── switched.tsv
│   │       ├── unswitched.tsv
│   │       └── wsc273.tsv
│   ├── script
│   │   ├── hnn_config_large.json
│   │   ├── hnn_init_large.spec
│   │   ├── hnn_train_large.sh
│   │   └── requirements.txt
│   └── src
│       ├── __init__.py
│       ├── apps
│       │   ├── arguments.py
│       │   ├── dataloader.py
│       │   ├── dataparallel.py
│       │   ├── hnn_dataset.py
│       │   ├── hnn_model.py
│       │   ├── metrics.py
│       │   ├── run_hnn.py
│       │   └── training_utils.py
│       ├── bert
│       │   ├── __init__.py
│       │   ├── __main__.py
│       │   ├── convert_tf_checkpoint_to_pytorch.py
│       │   ├── init_spec.py
│       │   ├── modeling.py
│       │   ├── optimization.py
│       │   └── tokenization.py
│       ├── functions
│       │   ├── __init__.py
│       │   └── ops.py
│       ├── module
│       │   ├── __init__.py
│       │   ├── helper.py
│       │   ├── loss.py
│       │   ├── pooling.py
│       │   └── tf_utils.py
│       ├── optims
│       │   ├── __init__.py
│       │   └── lr_schedulers.py
│       └── utils
│           ├── __init__.py
│           ├── argument_types.py
│           └── logger_util.py
├── input_examples   输入样本示例
│   ├── mlm_train.json  MLM模型的训练样本
│   ├── pair-input.txt
│   └── single-input.txt
├── int_test_data   测试模型的数据，包含少量样本
│   └── glue
│       ├── expected
│       │   ├── encoder
│       │   │   ├── bert_uncased_lower
│       │   │   │   └── cola_encoding.pt
│       │   │   └── roberta_cased_lower
│       │   │       └── cola_encoding.pt
│       │   └── prepro_std
│       │       ├── bert_base_uncased_lower
│       │       │   ├── cola_train.json
│       │       │   ├── mnli_train.json
│       │       │   └── stsb_train.json
│       │       └── roberta_base_cased
│       │           ├── cola_train.json
│       │           ├── mnli_train.json
│       │           └── stsb_train.json
│       └── input
│           ├── encoder
│           │   ├── bert_uncased_lower
│           │   │   ├── cola_dev.json
│           │   │   ├── cola_test.json
│           │   │   └── cola_train.json
│           │   └── roberta_cased_lower
│           │       ├── cola_dev.json
│           │       ├── cola_test.json
│           │       └── cola_train.json
│           └── prepro_std
│               ├── cola_train.tsv
│               ├── glue_task_def.yml
│               ├── mnli_train.tsv
│               └── stsb_train.tsv
├── int_test_encoder.py   使用测试数据测试编码
├── int_test_prepro_std.py  使用测试数据测试准备数据，首先准备数据，然后编码，然后用测试训练
├── module
│   ├── __init__.py
│   ├── bert_optim.py
│   ├── common.py
│   ├── dropout_wrapper.py
│   ├── my_optim.py
│   ├── pooler.py
│   ├── san.py
│   ├── san_model.py
│   ├── similarity.py
│   └── sub_layers.py
├── mt_dnn
│   ├── __init__.py
│   ├── batcher.py
│   ├── inference.py
│   ├── loss.py
│   ├── matcher.py
│   ├── model.py
│   └── perturbation.py     #对抗学习的扰动配置
├── mt_dnn_models   下载好的mtdnn模型
│   ├── bert_model_base_chinese.pt
│   ├── bert_model_base_uncased.pt
│   ├── bert_model_large_uncased.pt
│   ├── mt_dnn_base_uncased.pt
│   ├── mt_dnn_kd_large_cased.pt
│   ├── mt_dnn_large_uncased.pt
│   ├── roberta
│   │   ├── encoder.json
│   │   ├── ict.txt
│   │   └── vocab.bpe
│   ├── roberta.base
│   │   ├── NOTE
│   │   ├── dict.txt
│   │   └── model.pt
│   └── roberta.large
│       ├── NOTE
│       ├── dict.txt
│       └── model.pt
├── myexample
├── predict.py   #模型预测脚本
├── predict_api.py   #一个flask api
├── predict_api_test.py
├── predict_papertext_api.py
├── predict_papertext_api_test.py
├── prepare_distillation_data.py   准备数据蒸馏的数据
├── prepro_std.py    #准备数据
├── pretrained_models.py
├── requirements.txt
├── run_toy.sh
├── sample_data  #迷你数据集，测试模型
│   ├── checkpoint
│   │   ├── config.json
│   │   ├── log.txt
│   │   ├── mnli_matched_dev_scores_0.json
│   │   ├── mnli_matched_test_scores_0.json
│   │   ├── mnli_mismatched_dev_scores_0.json
│   │   └── mnli_mismatched_test_scores_0.json
│   ├── input
│   │   ├── CoLA
│   │   │   ├── dev.tsv
│   │   │   ├── test.tsv
│   │   │   └── train.tsv
│   │   ├── MNLI
│   │   │   ├── dev_matched.tsv
│   │   │   ├── dev_mismatched.tsv
│   │   │   ├── test_matched.tsv
│   │   │   ├── test_mismatched.tsv
│   │   │   └── train.tsv
│   │   ├── MRPC
│   │   │   ├── dev.tsv
│   │   │   ├── test.tsv
│   │   │   └── train.tsv
│   │   ├── QNLI
│   │   │   ├── dev.tsv
│   │   │   ├── test.tsv
│   │   │   └── train.tsv
│   │   ├── QQP
│   │   │   ├── dev.tsv
│   │   │   ├── test.tsv
│   │   │   └── train.tsv
│   │   ├── RTE
│   │   │   ├── dev.tsv
│   │   │   ├── test.tsv
│   │   │   └── train.tsv
│   │   ├── SNLI
│   │   │   ├── dev.tsv
│   │   │   ├── test.tsv
│   │   │   └── train.tsv
│   │   ├── SST-2
│   │   │   ├── dev.tsv
│   │   │   ├── test.tsv
│   │   │   └── train.tsv
│   │   ├── STS-B
│   │   │   ├── dev.tsv
│   │   │   ├── test.tsv
│   │   │   └── train.tsv
│   │   ├── SciTail
│   │   │   └── tsv_format
│   │   │       ├── scitail_1.0_dev.tsv
│   │   │       ├── scitail_1.0_test.tsv
│   │   │       └── scitail_1.0_train.tsv
│   │   ├── WNLI
│   │   │   ├── dev.tsv
│   │   │   ├── test.tsv
│   │   │   └── train.tsv
│   │   └── my_head.sh
│   └── output
│       ├── cola_dev.json
│       ├── cola_test.json
│       ├── cola_train.json
│       ├── mnli_matched_dev.json
│       ├── mnli_matched_test.json
│       ├── mnli_mismatched_dev.json
│       ├── mnli_mismatched_test.json
│       ├── mnli_train.json
│       ├── mrpc_dev.json
│       ├── mrpc_test.json
│       ├── mrpc_train.json
│       ├── qnli_dev.json
│       ├── qnli_test.json
│       ├── qnli_train.json
│       ├── qqp_dev.json
│       ├── qqp_test.json
│       ├── qqp_train.json
│       ├── rte_dev.json
│       ├── rte_test.json
│       ├── rte_train.json
│       ├── scitail_dev.json
│       ├── scitail_test.json
│       ├── scitail_train.json
│       ├── snli_dev.json
│       ├── snli_test.json
│       ├── snli_train.json
│       ├── sst_dev.json
│       ├── sst_test.json
│       ├── sst_train.json
│       ├── stsb_dev.json
│       ├── stsb_test.json
│       ├── stsb_train.json
│       ├── wnli_dev.json
│       ├── wnli_test.json
│       └── wnli_train.json
├── scripts  一些运行的脚本
│   ├── convert_tf_to_pt.py  把TensorFlow模型转换成pt模型
│   ├── domain_adaptation_run.sh  域迁移
│   ├── run_mt_dnn.sh 训练一个多任务，mnli,rte,qqp,qnli,mrpc,sst,cola,stsb
│   ├── run_mt_dnn_gc_fp16.sh  梯度裁剪和fp16训练一个多任务
│   ├── run_rte.sh 训练一个rte任务
│   ├── run_rte_mt_dnn_kd.sh  使用知识蒸馏模型训练rte
│   ├── run_rte_roberta.sh 使用roberta模型训练rte
│   ├── run_stsb.sh 训练一个stsb
│   ├── scitail_domain_adaptation_bash.sh  调用domain_adaptation_run.sh
│   ├── snli_domain_adaptation_bash.sh  调用domain_adaptation_run.sh
│   └── strip_model.py
├── setup.cfg
├── tasks
│   ├── __init__.py  任务头
├── tests  测试任务
│   ├── _test_train.py
│   ├── test.sh
│   └── test_prepro.py
├── train.py 训练脚本
├── trained_model 训练好的模型
│   ├── absa_dem8.pt
│   └── papertext.pt
├── tutorials 向导
│   ├── Run_Your_Own_Task_in_MT-DNN.ipynb
│   ├── bert_base_uncased
│   │   ├── snlisample_dev.json
│   │   ├── snlisample_test.json
│   │   └── snlisample_train.json
│   ├── snlisample_dev.tsv
│   ├── snlisample_test.tsv
│   ├── snlisample_train.tsv
│   └── tutorial_task_def.yml
└── wrong_sample
    └── README.md

119 directories, 580 files
```