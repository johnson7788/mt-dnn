# 测试中文的多任务模型

# 需要准备的文件
experiments/myexample/my_task_def.yml  # 定义你的任务，多个任务
experiments/myexample/mydata_prepro.py  #定义你的每个任务的处理方法，规范处理格式
data_my   # 你的数据的保存目录，包括原始数据，规范后的数据，和tokenize后的数据

## 任务1， Aspect的情感分析，3分类
## 任务2， 8个维度的词性判别，2分类

1. 生成规范式的数据
```buildoutcfg
python experiments/myexample/mydata_prepro.py
# tree data_my/
data_my/
└── canonical_data
    ├── absa_dev.tsv
    ├── absa_test.tsv
    ├── absa_train.tsv
    ├── dem8_dev.tsv
    ├── dem8_test.tsv
    └── dem8_train.tsv
```
2. 预处理数据为tokenizer后的格式, my_task_def.yml中定义的所有任务，都会进行处理
```buildoutcfg
python prepro_std.py --model bert-base-chinese --root_dir data_my/canonical_data --task_def experiments/myexample/my_task_def.yml --do_lower_case

#tree data_my/canonical_data/
data_my/
└── canonical_data
    ├── absa_dev.tsv
    ├── absa_test.tsv
    ├── absa_train.tsv
    ├── bert-base-chinese
    │   ├── absa_dev.json
    │   ├── absa_test.json
    │   ├── absa_train.json
    │   ├── dem8_dev.json
    │   ├── dem8_test.json
    │   └── dem8_train.json
    ├── dem8_dev.tsv
    ├── dem8_test.tsv
    └── dem8_train.tsv
```
3. 训练任务
```buildoutcfg
python train.py --init_checkpoint mt_dnn_models/bert_model_base_chinese.pt --task_def experiments/myexample/my_task_def.yml --data_dir data_my/canonical_data/bert-base-chinese --train_datasets absa --test_datasets absa```
```

# 测试模型
## 测试情感分析的任务
predict.py --task_def experiments/myexample/my_task_def.yml --task absa --task_id 0 --checkpoint trained_model/absa_dem8.pt --prep_input data_my/canonical_data/bert-base-chinese/absa_test.json --score predict_score.txt --with_label

## 测试8个维度的任务
predict.py --task_def experiments/myexample/my_task_def.yml --task dem8 --task_id 1 --checkpoint trained_model/absa_dem8.pt --prep_input data_my/canonical_data/bert-base-chinese/dem8_test.json --score predict_score.txt --with_label

# 新建一个flask的api接口