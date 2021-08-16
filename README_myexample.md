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
data_my/canonical_data/
├── absa_dev.tsv
├── absa_test.tsv
├── absa_train.tsv
├── bert-base-chinese
│         ├── absa_dev.json
│         ├── absa_test.json
│         ├── absa_train.json
│         ├── dem8_dev.json
│         ├── dem8_test.json
│         └── dem8_train.json
├── dem8_dev.tsv
├── dem8_test.tsv
└── dem8_train.tsv

```
3. 训练任务
单任务
```buildoutcfg
python train.py --init_checkpoint mt_dnn_models/bert_model_base_chinese.pt --task_def experiments/myexample/my_task_def.yml --data_dir data_my/canonical_data/bert-base-chinese --train_datasets absa --test_datasets absa```
```
多任务
```angular2html
# 2个任务
python train.py --data_dir data_my/canonical_data/bert-base-chinese --init_checkpoint mt_dnn_models/bert_model_base_chinese.pt --batch_size 8 --task_def experiments/myexample/my_task_def.yml --output_dir checkpoints/mt-dnn-absa --log_file checkpoints/mt-dnn-absa/log.log  --answer_opt 1  --optimizer adamax  --train_datasets absa,dem8 --test_datasets absa,dem8 --grad_clipping 0  --global_grad_clipping 1  --learning_rate 5e-5

# 3个任务
python train.py --data_dir data_my/canonical_data/bert-base-chinese --init_checkpoint mt_dnn_models/bert_model_base_chinese.pt --batch_size 32 --task_def experiments/myexample/my_task_def.yml --output_dir checkpoints/mt-dnn-absa --log_file checkpoints/mt-dnn-absa/log.log  --answer_opt 1  --optimizer adamax  --train_datasets absa,dem8,purchase --test_datasets absa,dem8,purchase --grad_clipping 0  --global_grad_clipping 1  --learning_rate 5e-5
```

# 测试模型
## 测试情感分析的任务
predict.py --task_def experiments/myexample/my_task_def.yml --task absa --task_id 0 --checkpoint trained_model/absa_dem8.pt --prep_input data_my/canonical_data/bert-base-chinese/absa_test.json --score predict_score.txt --with_label

## 测试8个维度的任务
predict.py --task_def experiments/myexample/my_task_def.yml --task dem8 --task_id 1 --checkpoint trained_model/absa_dem8.pt --prep_input data_my/canonical_data/bert-base-chinese/dem8_test.json --score predict_score.txt --with_label

## 测试购买意向
predict.py --task_def experiments/myexample/my_task_def.yml --task purchase --task_id 2 --checkpoint trained_model/absa_dem8.pt --prep_input data_my/canonical_data/bert-base-chinese/purchase_test.json --score predict_score.txt --with_label

## 保存预测错误的样本到单独的文件, wrong_sample/purchase_wrong_seed_13.pkl
predict.py --task_def experiments/myexample/my_task_def.yml --task purchase --task_id 2 --checkpoint trained_model/absa_dem8.pt --prep_input data_my/canonical_data/bert-base-chinese/purchase_test.json --score predict_score.txt --with_label --do_collection --collection_file wrong_sample/purchase_wrong_seed_13.pkl

# 新建一个flask的api接口
predict_api.py



# 添加一个新的关系判断的类型的任务
修改任务的定义：data_utils/task_def.py, 例如加入RELATION
```angular2html
class DataFormat(IntEnum):
    PremiseOnly = 1
    PremiseAndOneHypothesis = 2
    PremiseAndMultiHypothesis = 3
    MRC = 4
    Seqence = 5
    MLM = 6
    RELATION = 7
```

