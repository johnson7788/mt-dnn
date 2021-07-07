# 测试中文的多任务模型

# 需要准备的文件
experiments/mypaper/paper_task_def.yml  # 定义你的任务，多个任务
experiments/mypaper/paper_prepro.py  #定义你的每个任务的处理方法，规范处理格式
data_my   # 你的数据的保存目录，包括原始数据，规范后的数据，和tokenize后的数据

## 任务1， paper text  分类

1. 生成规范式的数据
```buildoutcfg
experiments/mypaper/paper_prepro.py 
# tree data_my/
data_my/
└── paper_data

```
2. 预处理数据为tokenizer后的格式, my_task_def.yml中定义的所有任务，都会进行处理
```buildoutcfg
python prepro_std.py --model bert-base-uncased --root_dir data_my/paper_data --task_def experiments/mypaper/paper_task_def.yml --do_lower_case

```
3. 训练任务
```buildoutcfg
python train.py --init_checkpoint mt_dnn_models/bert_model_base_uncased.pt --task_def experiments/mypaper/paper_task_def.yml --data_dir data_my/paper_data/bert-base-uncased --train_datasets papertext --test_datasets papertext```
```

# 测试模型
## 测试任务
predict.py --task_def experiments/myexample/my_task_def.yml --task absa --task_id 0 --checkpoint trained_model/absa_dem8.pt --prep_input data_my/canonical_data/bert-base-chinese/absa_test.json --score predict_score.txt --with_label

# 新建一个flask的api接口
predict_api.py
