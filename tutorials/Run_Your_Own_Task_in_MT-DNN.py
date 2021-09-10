#!/usr/bin/env python
# coding: utf-8

# # 使用MT-DNN用于你自己的任务
# 要使用MT-DNN运行自己的任务实际上是简单的3步：
# #1.将任务添加到任务定义配置中
# #2.使用正确的架构准备任务数据
# #3.在train.py中指定您的任务名称

# ## Step 1 - 定义你的任务配置
# MT-DNN 使用yaml格式 [yaml](https://en.wikipedia.org/wiki/YAML) 文件.
# 示例配置如下 :
# <pre>snlisample:
#   data_format: PremiseAndOneHypothesis
#   enable_san: true
#   labels:
#   - contradiction
#   - neutral
#   - entailment
#   metric_meta:
#   - ACC
#   loss: CeCriterion
#   kd_loss: MseCriterion
#   adv_loss: SymKlCriterion
#   n_class: 3
#   task_type: Classification</pre>
#
# 我们将把这个“snlisample”任务为例，以向您展示这些字段是什么，然后一步一步地添加它们。您也可以在“教程”文件夹下找到完整配置文件。

# <pre>snlisample
#   task_type: Classification
#   n_class: 3</pre> 
# 
#   支持的任务类型
#     1. Classification
#     2. Regression
#     3. Ranking
#     4. Span
#     5. SequenceLabeling
#     6. MaskLM
#   More details in [data_utils/task_def.py](../data_utils/task_def.py)
#   
# Also, specify how many classes in total in your task, under "n_class" field.
# 

# ### Add data information for your task
# 
# <pre>snlisample:
#   data_format: PremiseAndOneHypothesis
#   enable_san: true
#   labels:
#   - contradiction
#   - neutral
#   - entailment
#   n_class: 3
#   task_type: Classification
#   </pre> 
#   
#   choose the correct data format based on your task, currently we support 4 types of data formats, coresponds to different tasks:
#   1. "PremiseOnly" : single text, i.e. premise. Data format is "id" \t "label" \t "premise" .
#   2. "PremiseAndOneHypothesis" : two texts, i.e. one premise and one hypothesis. Data format is "id" \t "label" \t "premise" \t "hypothesis".
#   3. "PremiseAndMultiHypothesis" : one text as premise and multiple candidates of texts as hypothesis. Data format is "id" \t "label" \t "premise" \t "hypothesis_1" \t "hypothesis_2" \t ... \t "hypothesis_n".
#   4. "Sequence" : sequence tagging. Data format is "id" \t "label" \t "premise".
#   
#   More details in [data_utils/task_def.py](../data_utils/task_def.py)
#
# 代码使用Surfix来区分它是什么类型（“_train”，“_ dev”和“_test”）。所以：
# #1.确保您的训练集被命名为“Task_train”（用任务名称替换任务）
# #2.确保DEV设置和测试设置以“_dev”和“_test”结尾。
# #如果您更喜欢使用可读标签（文本），则可以在“labels”字段下的数据集中指定标签。

# ### Add hyper-parameters for your task
# 
# <pre>snlisample:
#   data_format: PremiseAndOneHypothesis
#   enable_san: true
#   labels:
#   - contradiction
#   - neutral
#   - entailment
#   n_class: 3
#   task_type: Classification
#   </pre>
# 
#

# 如果您想使用随机答案网络（[san]（https://www.aclweb.org/anthology/p18-1157.pdf），请设置“true”。
# #我们还支持为不同的任务分配不同的dropout prob，请在“Dropout_p”字段中为您的任务分配prob。更多样本请参阅其他GLUE任务def文件。

# ### 增加 metric 和损失
# 
# <pre>snlisample:
#   data_format: PremiseAndOneHypothesis
#   enable_san: true
#   labels:
#   - contradiction
#   - neutral
#   - entailment
#   metric_meta:
#   - ACC
#   loss: CeCriterion
#   kd_loss: MseCriterion
#   adv_loss: SymKlCriterion
#   n_class: 3
#   task_type: Classification
#   </pre>
#   
#   More details about metrics,please refer to [data_utils/metrics.py](../data_utils/metrics.py);
#
# 您可以选择loss，kd_loss（知识蒸馏）和adv_loss（对抗训练）从文件中的预定义损失文件 [data_utils/loss.py](../data_utils/loss.py)，您可以实现您的定制损失此文件并在任务配置中指定它。
#
#   

# ## Step 2 - 准备数据，用正确的格式
#
# 记住您在配置中设置的“data_format”， 请按照下面的详细数据格式，准备您的数据：
#
# 1. "PremiseOnly" : single text, i.e. premise. Data format is "id" \t "label" \t "premise" .
# 2. "PremiseAndOneHypothesis" : two texts, i.e. one premise and one hypothesis. Data format is "id" \t "label" \t "premise" \t "hypothesis".
# 3. "PremiseAndMultiHypothesis" : one text as premise and multiple candidates of texts as hypothesis. Data format is "id" \t "label" \t "premise" \t "hypothesis_1" \t "hypothesis_2" \t ... \t "hypothesis_n".
# 4. "Sequence" : sequence tagging. Data format is "id" \t "label" \t "premise".

# ### Tokenization 并且转换成json格式保存
#
#训练代码以JSON格式读取tokenized数据。请使用“prepro_std.py”进行tokenization并将数据转换为JSON格式。
# #例如，让我们试一试例任务“Snlisample”！请尝试使用以下命令进行此任务的预处理数据：

# <pre>python prepro_std.py --model bert-base-uncased --root_dir tutorials/ --task_def tutorials/tutorial_task_def.yml</pre>

# 示例输出:
# <pre>
# 07/02/2020 08:42:26 Task snlisample
# 07/02/2020 08:42:26 tutorials/bert_base_uncased/snlisample_train.json
# 07/02/2020 08:42:26 tutorials/bert_base_uncased/snlisample_dev.json
# 07/02/2020 08:42:26 tutorials/bert_base_uncased/snlisample_test.json
# </pre>

# ## Step 3 -开始训练你的任务
# 
# 1. 为所有任务添加您的配置
# 2.如果您正在进行多任务学习，请加入新的任务与exsting任务，请在train.py args中附加任务和test_set前缀：
#  “--train_datasets现有_tasks，your_new_task --test_datasets现有_task_test_sets，your_new_task_sets”;
#  如果您正在进行单一任务微调，请仅在训练参数中保留您的新任务名字即可。
#

# for example, we would like to finetune this "snlisample" task, with the sampled data in tutorials folder, here is the command:
# 例如，我们想微调此“snlisample”任务，使用Tutorials文件夹中的采样数据，这是命令：
# <pre>
# python train.py --task_def tutorials/tutorial_task_def.yml --data_dir tutorials/bert_base_uncased/ --train_datasets snlisample --test_datasets snlisample
# </pre>
#
# 如果您想添加对抗训练，请确保在任务配置文件中定义“Adv_loss”，并请添加“--adv_train”：
# <pre>
# python train.py --task_def tutorials/tutorial_task_def.yml --data_dir tutorials/bert_base_uncased/ --train_datasets snlisample --test_datasets snlisample --adv_train
# </pre>
# 
# Example Output:
# <pre>
# 07/02/2020 09:13:38 Total number of params: 109484547
# 07/02/2020 09:13:38 At epoch 0
# 07/02/2020 09:13:38 Task [ 0] updates[     1] train loss[1.25835] remaining[0:00:06]
# predicting 0
# 07/02/2020 09:13:42 Task snlisample -- epoch 0 -- Dev ACC: 36.000
# predicting 0
# 07/02/2020 09:13:42 [new test scores saved.]
# 07/02/2020 09:13:44 At epoch 1
# predicting 0
# 07/02/2020 09:13:47 Task snlisample -- epoch 1 -- Dev ACC: 43.000
# predicting 0
# 07/02/2020 09:13:48 [new test scores saved.]
# </pre>
