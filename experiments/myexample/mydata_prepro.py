import collections
import os
import argparse
import random
from sys import path
import pickle
import re

path.append(os.getcwd())
from experiments.common_utils import dump_rows
from data_utils.task_def import DataFormat
from data_utils.log_wrapper import create_logger
import sys

logger = create_logger(__name__, to_disk=True, log_file='mydata_prepro.log')

data_configs = {
    'absa': {
        'cache_file': "data_my/canonical_data/source_data/absa.pkl",
        'DataFormat': DataFormat.PremiseAndOneHypothesis,
        'do_truncate': True,
        'todict': True,
    },
    'dem8': {
        'cache_file': "data_my/canonical_data/source_data/dem8.pkl",
        'DataFormat': DataFormat.PremiseAndOneHypothesis,
        'do_truncate': True,
        'todict': True,
    },
    'purchase': {
        'cache_file': "data_my/canonical_data/source_data/purchase.pkl",
        'DataFormat': DataFormat.PremiseAndOneHypothesis,
        'do_truncate': True,
        'todict': True,
        'only_addidx': True,
    },
    'wholesentiment': {
        'cache_file': "data_my/canonical_data/source_data/wholesentiment.pkl",
        'DataFormat': DataFormat.PremiseOnly,
        'do_truncate': True,
        'todict': True,
        'only_addidx': True,
    },
    'brand': {
        'cache_file': "data_my/canonical_data/source_data/brand.pkl",
        'DataFormat': DataFormat.RELATION,
        'do_truncate': True,
        'todict': False,
    },
    # 'nersentiment': {
    #     'cache_file': "data_my/canonical_data/source_data/nersentiment.pkl",
    #     'DataFormat': DataFormat.Sequence,
    #     'do_truncate': True,
    #     'todict': True,
    #     'only_addidx': True,
    # },
    'pinpainer': {
        'cache_file': "data_my/canonical_data/source_data/pinpainer.pkl",
        'DataFormat': DataFormat.Sequence,
        'do_truncate': True,
        'todict': True,
        'only_addidx': True,
    },
}

def truncate(input_text, max_len, trun_post='post'):
    """
    实施截断数据
    :param input_text:
    :param max_len:   eg: 15
    :param trun_post: 截取方向，向前还是向后截取，
                    "pre"：截取前面的， "post"：截取后面的
    :return:
    """
    if max_len is not None and len(input_text) > max_len:
        if trun_post == "post":
            return input_text[-max_len:]
        else:
            return input_text[:max_len]
    else:
        return input_text

def aspect_truncate(content,aspect,aspect_start,aspect_end,left_max_seq_len,aspect_max_seq_len,right_max_seq_len):
    """
    截断函数
    :param content:句子
    :param aspect:关键字
    :param aspect_start:开始位置
    :param aspect_end:结束位置
    :return:
    """
    text_left = content[:aspect_start]
    text_right = content[aspect_end:]
    text_left = truncate(text_left, left_max_seq_len)
    aspect = truncate(aspect, aspect_max_seq_len)
    text_right = truncate(text_right, right_max_seq_len, trun_post="pre")
    new_content = text_left + aspect + text_right
    return new_content

def do_truncate_data(task, data, left_max_seq_len=60, aspect_max_seq_len=10, right_max_seq_len=60):
    """
    对数据做truncate
    :param data:针对不同类型的数据进行不同的截断
    :return:返回列表，是截断后的文本，aspect
    所以如果一个句子中有多个aspect关键字，那么就会产生多个截断的文本+关键字，组成的列表，会产生多个预测结果
    和locations: [(start_idx,end_idx),...]
    """
    contents = []
    #保存关键字的索引，[(start_idx, end_idx)...]
    locations = []
    #保留原始数据，一同返回
    original_data = []
    for idx, one_data in enumerate(data):
        if len(one_data) == 2:
            #不带aspect关键字的位置信息，自己查找位置
            content, aspect = one_data
            iter = re.finditer(aspect, content)
            for m in iter:
                aspect_start, aspect_end = m.span()
                new_content = aspect_truncate(content, aspect, aspect_start, aspect_end,left_max_seq_len,aspect_max_seq_len,right_max_seq_len)
                contents.append((new_content, aspect))
                locations.append((aspect_start,aspect_end))
                original_data.append(one_data)
        elif len(one_data) == 3:
            #不带aspect关键字的位置信息，带label
            content, aspect, label = one_data
            iter = re.finditer(aspect, content)
            for m in iter:
                aspect_start, aspect_end = m.span()
                new_content = aspect_truncate(content, aspect, aspect_start, aspect_end)
                contents.append((new_content, aspect, label))
                locations.append((aspect_start,aspect_end))
                original_data.append(one_data)
        elif len(one_data) == 4:
            # 不带label时，长度是4，
            content, aspect, aspect_start, aspect_end = one_data
            new_content = aspect_truncate(content, aspect, aspect_start,aspect_end,left_max_seq_len,aspect_max_seq_len,right_max_seq_len)
            contents.append((new_content, aspect))
            locations.append((aspect_start, aspect_end))
            original_data.append(one_data)
        elif len(one_data) == 5:
            content, aspect, aspect_start, aspect_end, label = one_data
            new_content = aspect_truncate(content, aspect, aspect_start, aspect_end,left_max_seq_len,aspect_max_seq_len,right_max_seq_len)
            contents.append((new_content, aspect, label))
            locations.append((aspect_start, aspect_end))
            original_data.append(one_data)
        elif len(one_data) == 7:
            content, aspect, aspect_start, aspect_end, label, channel,wordtype = one_data
            new_content = aspect_truncate(content, aspect, aspect_start, aspect_end,left_max_seq_len,aspect_max_seq_len,right_max_seq_len)
            contents.append((new_content, aspect, label))
            locations.append((aspect_start, aspect_end))
            original_data.append(one_data)
        else:
            print(f"这条数据异常: {one_data},数据长度或者为2, 4，或者为5，跳过")
            continue
            # raise Exception(f"这条数据异常: {one_data},数据长度或者为2, 4，或者为5")
    assert len(contents) == len(locations) == len(original_data)
    print(f"截断的参数left_max_seq_len: {left_max_seq_len}, aspect_max_seq_len: {aspect_max_seq_len}, right_max_seq_len:{right_max_seq_len}。截断后的数据总量是{len(contents)}")
    return original_data, contents, locations

def truncate_relation(data, max_seq_len=150):
    """
    只对text的长度进行截取，根据
    :param data: 源数据
    :type data:
    :param max_seq_len: 最大序列长度
    :type max_seq_len:
    :return:
    :rtype:
    """
    # 把最大长度减去20，作为实体词的长度的备用
    max_length = max_seq_len - 20
    truncate_data = []
    length_counter = collections.Counter()
    for one in data:
        text = one['text']
        if len(text) > max_seq_len:
            length_counter['超过最大长度'] += 1
            #开始截断
            h_entity = one['h']['name']
            t_entity = one['t']['name']
            h_length = len(one['h']['name'])
            t_length = len(one['t']['name'])
            h_start = one['h']['pos'][0]
            h_end = one['h']['pos'][1]
            t_start = one['t']['pos'][0]
            t_end = one['t']['pos'][1]
            # 先判断2个实体词之间的距离是否大于max_seq_len,如果大于，那么就2个实体词的2层分别保留一段位置，否则就从2个实体词的2层剪断
            if h_start < t_start:
                # 实体词h在前，t在后
                if t_end - h_start > max_length:
                    # 实体词的2册都进行截取,  形式是,被截断的示例是: xx|xxx entity1 xxx|xx   +  xxx|xx entity2 xxxx|x, 其中|表示被截断的标记
                    half_length = max_length/2
                    # 第一个实体前后的句子开始和结束位置
                    l1_start = h_start - int(half_length/2)
                    if l1_start < 0:
                        l1_start = 0
                    l1_end = h_end + int(half_length/2)
                    l2_start = t_start - int(half_length/2)
                    l2_end = t_end + int(half_length/2)
                    newtext = text[l1_start:l1_end] + text[l2_start:l2_end]
                    h_start = h_start - l1_start
                    h_end = h_start + h_length
                    #第二个实体位置新的开始
                    t_start = t_start - l1_start - (l2_start- l1_end)
                    t_end = t_start + t_length
                    assert newtext[h_start:h_end] == h_entity, "截断后的实体位置信息不对"
                    assert newtext[t_start:t_end] == t_entity, "截断后的实体位置信息不对"
                    assert len(newtext) <= max_seq_len, f"最大长度截断后过长{len(newtext)}"
                else:
                    # 在2侧分别剪断, 计算下2侧分别可以保存的长度, 形式是: xx|xxx entity1 xxxxx entity2 xxx|xx, |表示被截断
                    can_keep_length = max_length - (t_end - h_start)
                    #实体1左侧可以保留的长度
                    left_keep = int(can_keep_length/2)
                    right_keep = can_keep_length - left_keep
                    # 句子的索引位置
                    left_start = h_start - left_keep
                    if left_start < 0:
                        left_start = 0
                    right_end = t_end + right_keep
                    #截取后的文本长度
                    newtext = text[left_start:right_end]
                    h_start = h_start - left_start
                    h_end = h_start + h_length
                    t_start = t_start - left_start
                    t_end = t_start + t_length
                    assert newtext[h_start:h_end] == h_entity, "截断后的实体位置信息不对"
                    assert newtext[t_start:t_end] == t_entity, "截断后的实体位置信息不对"
                    assert len(newtext) <= max_seq_len, f"最大长度截断后过长{len(newtext)}"
            else:
                # 实体词h在后，t在前, 尚未修改, xx|xxx entity2 xxx|xx   +  xxx|xx entity1 xxxx|x, 其中|表示被截断的标记
                if h_end - t_start > max_length:
                    half_length = max_length / 2
                    # 第一个实体前后的句子开始和结束位置
                    l1_start = t_start - int(half_length / 2)
                    if l1_start < 0:
                        l1_start = 0
                    l1_end = t_end + int(half_length / 2)
                    l2_start = h_start - int(half_length / 2)
                    l2_end = h_end + int(half_length / 2)
                    newtext = text[l1_start:l1_end] + text[l2_start:l2_end]
                    h_start = h_start - l1_start - (l2_start- l1_end)
                    h_end = h_start + h_length
                    t_start = t_start - l1_start
                    t_end = t_start + t_length
                    assert newtext[h_start:h_end] == h_entity, "截断后的实体位置信息不对"
                    assert newtext[t_start:t_end] == t_entity, "截断后的实体位置信息不对"
                    assert len(newtext) <= max_seq_len, f"最大长度截断后过长{len(newtext)}"
                else:
                    # 在2层分别剪断, 计算下2层分别可以保存的长度, xx|xxx entity2 xxxxx entity1 xxx|xx, |表示被截断
                    can_keep_length = max_length - (h_start - t_end)
                    # 实体1左侧可以保留的长度
                    left_keep = int(can_keep_length / 2)
                    right_keep = can_keep_length - left_keep
                    # 句子的索引位置
                    left_start = t_start - left_keep
                    if left_start < 0:
                        left_start = 0
                    right_end = h_end + right_keep
                    # 截取后的文本长度
                    newtext = text[left_start:right_end]
                    h_start = h_start - left_start
                    h_end = h_start + h_length
                    t_start = t_start - left_start
                    if t_start < 0:
                        t_start = 0
                    t_end = t_start + t_length
                    assert newtext[h_start:h_end] == h_entity, "截断后的实体位置信息不对"
                    assert newtext[t_start:t_end] == t_entity, "截断后的实体位置信息不对"
                    assert len(newtext) <= max_seq_len, f"最大长度截断后过长{len(newtext)}"
            one['text'] = newtext
            one['h']['pos'][0] = h_start
            one['h']['pos'][1] = h_end
            one['t']['pos'][0] = t_start
            one['t']['pos'][1] = t_end
        else:
            length_counter['未超最大长度'] += 1
        truncate_data.append(one)
    print(f"超过和未超过最大长度{max_seq_len}的统计结果{length_counter}, 超过最大长度后将动态根据2个实体所在的位置对句子进行截断")
    return truncate_data

def save_source_data(task_name):
    sys.path.append('/Users/admin/git/label-studio/myexample')
    from convert_label_studio_data import get_all, get_demision8, get_all_purchase, get_all_brand, get_all_nersentiment, get_all_pinpainer, get_all_wholesentiment
    #保存三个数据集的原始数据，方便以后不从label-studio读取
    if task_name == "absa":
        absa_data = get_all(split=False, dirpath=f"/opt/lavector/absa", do_save=False, withmd5=True)
        pickle.dump(absa_data, open(data_configs[task_name]['cache_file'], "wb"))
    if task_name == "dem8":
        dem8_data = get_demision8(split=False,
                                 dirpath_list=['/opt/lavector/effect', '/opt/lavector/pack', '/opt/lavector/promotion',
                                               '/opt/lavector/component', '/opt/lavector/fragrance','/opt/lavector/dem8_verify','/opt/lavector/price_service_skin'],withmd5=True)
        pickle.dump(dem8_data, open(data_configs[task_name]['cache_file'], "wb"))
    if task_name == "purchase":
        purchase_data = get_all_purchase(dirpath=f"/opt/lavector/purchase", split=False, do_save=False,withmd5=True, max_label_num=-1, return_dict=True)
        pickle.dump(purchase_data, open(data_configs[task_name]['cache_file'], "wb"))
    if task_name == "brand":
        brand_data = get_all_brand(dirpath="/opt/lavector/relation/",split=False, do_save=False, withmd5=True)
        pickle.dump(brand_data, open(data_configs[task_name]['cache_file'], "wb"))
    if task_name == "nersentiment":
        nersentiment_data = get_all_nersentiment(dirpath="/opt/lavector/ner_sentiment/",split=False, do_save=False, withmd5=True, select_channels=["tmall", "jd"])
        pickle.dump(nersentiment_data, open(data_configs[task_name]['cache_file'], "wb"))
    if task_name == "wholesentiment":
        wholesentiment_data = get_all_wholesentiment(dirpath="/opt/lavector/ner_sentiment/",split=False, do_save=False, withmd5=True, select_channels=["tmall", "jd"])
        pickle.dump(wholesentiment_data, open(data_configs[task_name]['cache_file'], "wb"))
    if task_name == "pinpainer":
        pinpainer_data = get_all_pinpainer(dirpath="/opt/lavector/pinpainer/",split=False, do_save=False, withmd5=True,drop_keywords=[])
        pickle.dump(pinpainer_data, open(data_configs[task_name]['cache_file'], "wb"))
    if task_name == "absa":
        return absa_data
    elif task_name == "dem8":
        return dem8_data
    elif task_name == "brand":
        return brand_data
    elif task_name == "purchase":
        return purchase_data
    elif task_name == "nersentiment":
        return nersentiment_data
    elif task_name == "wholesentiment":
        return wholesentiment_data
    elif task_name == "pinpainer":
        return pinpainer_data

def do_truncate_nersentiment(data, do_truncate=True, max_seq_length = 180):
    """
    nertsentiment 的数据处理
    :param data: eg: [{'text': '非常满意', 'channel': 'jd', 'seq_label': '非常满意', 'md5': '67e6ef489951383e8722b2787df381fa', 'sentiment_label': '积极'}]
    :type data: list
    :param do_truncate: 截断 bool
    :param max_seq_length 最大序列长度 int
    :return:
    :rtype:
    """
    labels_dict = {
        "积极": "POS",
        "中性": "NEU",
        "消极": "NEG",
    }
    # all_data 里面是label和text的token的列表格式
    all_data = []
    #把data转变成序列标注的形式
    for uid, d in enumerate(data):
        text = d['text']
        text_len = len(text)
        token_label = d['seq_label']
        token_label_len = len(token_label)
        sentiment_label = d['sentiment_label']
        if token_label != 'empty':
            # 搜索token_label在原文text中的位置
            search_res = re.search(re.escape(token_label), text)
            assert search_res, f"请检查token_label{token_label}是否在原文{text}中出现"
            start,end = search_res.regs[0]
            # 在token_label的前后范围截断, 在token_label的前后保留的长度是keep_token_len
            keep_token_len = int((max_seq_length - token_label_len)/2)
            start_idx = start - keep_token_len
            end_ids = end + keep_token_len
            if start_idx < 0:
                #起始位置必须大于0
                start_idx = 0
            truncate_text = text[start_idx:end_ids]
            # 默认都是O标签
            labels = ["O"] * len(truncate_text)
            tokens = [i for i in truncate_text]
            search_res = re.search(re.escape(token_label), truncate_text)
            new_start,new_end = search_res.regs[0]
            label_word = labels_dict[sentiment_label]
            labels[new_start] = f"B-{label_word}"
            labels[new_start+1:new_end] = [f"I-{label_word}"] * (end-start-1)
        else:
            # 默认都是O标签
            labels = ["O"] * text_len
            tokens = [i for i in text]
            # 截断
            labels = labels[:max_seq_length]
            tokens = tokens[:max_seq_length]
        assert len(labels) == len(tokens), 'label和token长度不相等，有错误'
        one_data = {"label": labels, "premise": tokens}
        all_data.append(one_data)
    return all_data
def do_truncate_wholesentiment(data, max_seq_length = 150, left_max_seq_len=60, aspect_max_seq_len=10, right_max_seq_len=60):
    """
    整体情感的判断
    :param data:
    :type data:
    :param do_truncate:
    :type do_truncate:
    :return:
    :rtype:
    """
    assert isinstance(data[0], dict), "支持的数据格式是字典格式"
    new_data = []
    for d in data:
        text = d["text"]
        channel = d["channel"]
        label = d["sentiment_label"]
        # 添加新数据
        one_data = {
            "premise": text,
            "label": label,
        }
        new_data.append(one_data)
    return new_data

def do_truncate_pinpainer(data, do_truncate=True, max_seq_length = 180):
    """
    品牌ner识别的标签
    :param data: eg: {'text': '【有情】防脱发生姜洗发水500ml ?后22.9元 go:有情生姜洗发水防脱发生发去屑止痒去控油蓬松男女士专用姜汁膏露  ?', 'channel': 'weibo', 'keyword': '有情,后', 'labels': [{'end': 3, 'labels': '品牌', 'start': 1, 'text': '有情'}, {'end': 20, 'labels': '不是品牌', 'start': 19, 'text': '后'}, {'end': 31, 'labels': '品牌', 'start': 29, 'text': '有情'}], 'md5': '9f5dbe372f879aff28b6e91473b1d7bf'}
    :type data: list， 标签分为3种，即 B-PIN, I-PIN, O 的BIO格式
    :param do_truncate: 截断 bool
    :param max_seq_length 最大序列长度 int
    :return:
    :rtype:
    """
    labels_dict = {
        "品牌": "PIN",
    }
    # all_data 里面是label和text的token的列表格式
    all_data = []
    #把data转变成序列标注的形式
    for uid, d in enumerate(data):
        text = d['text']
        text_len = len(text)
        labels = d['labels']
        token_labels = ["O"] * text_len
        tokens = list(text)
        for label in labels:
            start = label['start']
            end = label['end']
            label_name = label['labels']
            keyword_name = label['text']
            #校验下品牌的名字对应的位置在原文中是存在，并且正确的
            if keyword_name != text[start:end]:
                print(f"原文中对应的开始和结束位置的词和标签给定的词不一致，请检查: {text}: label是{label_name}, keyword是{keyword_name}, 位置是{start}和{end}")
                continue
            if label_name in labels_dict:
                # 只需要标注为 "品牌的字段的内容", 这里即 "PIN"，
                token_label = labels_dict[label_name]
                token_labels[start] = f"B-{token_label}"
                token_labels[start+1:end] = [f"I-{token_label}"] * (end-start-1)
        if do_truncate:
            token_labels = token_labels[:max_seq_length]
            tokens = tokens[:max_seq_length]
        one_data = {"label": token_labels, "premise": tokens}
        all_data.append(one_data)
    return all_data

def do_truncate_purchase(data, do_truncate=True, max_seq_length = 150, left_max_seq_len=60, aspect_max_seq_len=10, right_max_seq_len=60):
    """
    购买的截断
    :param data:
    :type data:
    :param do_truncate:
    :type do_truncate:
    :return:
    :rtype:
    """
    channel_dict = {
        "redbook":"小红书:",
        "weibo":"微博:",
        "tmall":"天猫:",
        "jd":"京东:",
        "tiktok":"抖音:",
    }
    assert isinstance(data[0], dict), "支持的数据格式是字典格式"
    new_data = []
    for d in data:
        text = d["text"]
        title = d["title"]
        channel = d["channel"]
        label = d["label"]
        chinese_channel = channel_dict.get(channel)
        if not chinese_channel:
            # 如果不存在映射，那么设置chinese_channel等于英文的channel
            chinese_channel = channel
        aspect = d["keyword"]
        # 如果有索引，使用索引，如果没有，获取索引
        start_idx = d.get('start_idx')
        end_idx = d.get('end_idx')
        if title:
            text = title + text
            if start_idx and end_idx:
                # 如果给定了起始位置，那么更新下起始位置
                start_idx = start_idx + len(title)
                end_idx = end_idx + len(title)
        if not start_idx or not end_idx:
            # 如果每个，那么通过自己查找，只查找一个即可
            iter = re.finditer(aspect, text)
            for m in iter:
                start_idx, end_idx = m.span()
                break
            else:
                # 品牌词找不到，说明有问题
                start_idx, end_idx = 0, 0
        #进行截断, 截断后start_idx 和end_idx就不准了
        new_content = aspect_truncate(text, aspect, start_idx, end_idx,left_max_seq_len,aspect_max_seq_len,right_max_seq_len)
        # 加上渠道
        new_content = chinese_channel + new_content
        # 添加新数据
        one_data = {
            "premise": new_content,
            "hypothesis": aspect,
            "label": label,
        }
        new_data.append(one_data)
    return new_data

def load_absa_dem8(task_name='absa',left_max_seq_len=60, aspect_max_seq_len=10, right_max_seq_len=60, use_pickle=False, do_truncate=True):
    """
    Aspect Base sentiment analysis
    :param task_name: 是加载absa数据，还是dem8的数据
    :param do_truncate: 是否做裁剪
    :return:
    :rtype:
    """
    if task_name == 'absa':
        if use_pickle:
            assert os.path.exists(data_configs[task_name]['cache_file']), "源数据的pickle文件不存在，请检查"
            with open(data_configs[task_name]['cache_file'], 'rb') as f:
                all_data = pickle.load(f)
        else:
            all_data = save_source_data(task_name="absa")

        # 去除md5的位置，这个暂时不用
        all_data = [d[:-1] for d in all_data]
    elif task_name == 'dem8':
        # 注意dirpath_list使用哪些数据进行训练，那么预测时，也是用这样的数据
        # labels2id = {
        #     "是": 0,
        #     "否": 1,
        # }
        if use_pickle:
            assert os.path.exists(data_configs[task_name]['cache_file']), "源数据的pickle文件不存在，请检查"
            with open(data_configs[task_name]['cache_file'], 'rb') as f:
                all_data = pickle.load(f)
        else:
            all_data = save_source_data(task_name="dem8")
        # 去除md5的位置，这个暂时不用
        all_data = [d[:-1] for d in all_data]
    elif task_name == 'purchase':
        # 返回数据格式[(text, title, keyword, start_idx, end_idx, label),...]
        if use_pickle:
            assert os.path.exists(data_configs[task_name]['cache_file']), "源数据的pickle文件不存在，请检查"
            with open(data_configs[task_name]['cache_file'], 'rb') as f:
                a_data = pickle.load(f)
        else:
            a_data = save_source_data(task_name="purchase")
        data = do_truncate_purchase(data=a_data, do_truncate=do_truncate)
        return data
    elif task_name == 'brand':
        if use_pickle:
            assert os.path.exists(data_configs[task_name]['cache_file']), "源数据的pickle文件不存在，请检查"
            with open(data_configs[task_name]['cache_file'], 'rb') as f:
                all_data = pickle.load(f)
        else:
            all_data = save_source_data(task_name="brand")
    elif task_name == 'nersentiment':
        if use_pickle:
            assert os.path.exists(data_configs[task_name]['cache_file']), "源数据的pickle文件不存在，请检查"
            with open(data_configs[task_name]['cache_file'], 'rb') as f:
                all_data = pickle.load(f)
        else:
            all_data = save_source_data(task_name="nersentiment")
        # 处理nersentiment数据并返回
        data = do_truncate_nersentiment(data=all_data, do_truncate=do_truncate)
        return data
    elif task_name == 'wholesentiment':
        if use_pickle:
            assert os.path.exists(data_configs[task_name]['cache_file']), "源数据的pickle文件不存在，请检查"
            with open(data_configs[task_name]['cache_file'], 'rb') as f:
                all_data = pickle.load(f)
        else:
            all_data = save_source_data(task_name="wholesentiment")
        # 处理nersentiment数据并返回
        data = do_truncate_wholesentiment(data=all_data)
        return data
    elif task_name == 'pinpainer':
        if use_pickle:
            assert os.path.exists(data_configs[task_name]['cache_file']), "源数据的pickle文件不存在，请检查"
            with open(data_configs[task_name]['cache_file'], 'rb') as f:
                all_data = pickle.load(f)
        else:
            all_data = save_source_data(task_name)
        # 处理nersentiment数据并返回
        data = do_truncate_pinpainer(data=all_data, do_truncate=do_truncate)
        return data
    else:
        print("数据的种类不存在，退出")
        sys.exit(1)
    if do_truncate:
        if task_name == 'brand':
            data = truncate_relation(data=all_data)
        else:
            original_data, data, locations = do_truncate_data(task_name,all_data,left_max_seq_len, aspect_max_seq_len, right_max_seq_len)
    else:
        data = all_data
    if task_name == 'dem8':
        #处理完成的数据加前缀
        # 加上前缀，给每条数据
        new_data = []
        for dall, d in zip(all_data, data):
            word_type = dall[6]
            content = word_type + ':' + d[0]
            new_one = list(d)
            new_one[0] = content
            new_data.append(new_one)
        data = new_data
    return data

def split_save_data(data, random_seed, train_rate=0.8, dev_rate=0.1, test_rate=0.1, todict=True, only_addidx=False):
    """
    :param data:
    :type data:
    :param random_seed:
    :type random_seed:
    :param train_rate:
    :type train_rate:
    :param dev_rate:
    :type dev_rate:
    :param test_rate:
    :param todict: 变成字典的格式
    :type test_rate:
    :return:
    :rtype:
    """
    random.seed(random_seed)
    # 可以通过id找到对应的源数据
    data_id = list(range(len(data)))
    random.shuffle(data_id)
    total = len(data)
    train_num = int(total * train_rate)
    dev_num = int(total * dev_rate)
    test_num = int(total * test_rate)
    train_data_id = data_id[:train_num]
    dev_data_id = data_id[train_num:train_num+dev_num]
    test_data_id = data_id[train_num+dev_num:]
    train_data = [data[id] for id in train_data_id]
    dev_data = [data[id] for id in dev_data_id]
    test_data = [data[id] for id in test_data_id]
    # 处理一下，保存的格式
    def change_data(kind_data, only_addidx=False):
        """
        :param kind_data:
        :type kind_data:
        :param only_addidx: 数据是ok的，只需加一个idx即可
        :type only_addidx:
        :return:
        :rtype:
        """
        rows = []
        if only_addidx:
            for idx, one_data in enumerate(kind_data):
                one_data['uid'] = idx
                rows.append(one_data)
        else:
            for idx, one_data in enumerate(kind_data):
                content, keyword, label = one_data
                # label_id = labels2id[label]
                # assert label in ['消极','中性','积极','是','否'], "label不是特定的关键字，那么my_task_def.yml配置文件中的labels就不能解析，会出现错误"
                sample = {'uid': idx, 'premise': content, 'hypothesis': keyword, 'label': label}
                rows.append(sample)
        return rows
    if todict:
        train_data = change_data(train_data, only_addidx)
        dev_data = change_data(dev_data,only_addidx)
        test_data = change_data(test_data,only_addidx)
    print(f"训练集数量{len(train_data)}, 开发集数量{len(dev_data)}, 测试集数量{len(test_data)}")
    return train_data, dev_data, test_data, train_data_id, dev_data_id, test_data_id


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing GLUE/SNLI/SciTail dataset.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--root_dir', type=str, default='data_my',help='数据集的保存路径')
    parser.add_argument('--dataset', type=str, default='all',help='默认处理哪个数据集，all代表所有, 或者部分数据集，例如absa,dem8这样用逗号分割的参数')
    parser.add_argument('--save_source_pkl', action="store_true", help='把原始数据导出到本地的data_my/canonical_data/source_data文件夹')
    parser.add_argument('--use_pkl', action="store_true", help='使用本地的pkl缓存的原始数据，不使用label-studio产生的数据')
    args = parser.parse_args()
    return args


def do_prepro(root, use_pkl, seed, dataset='all'):
    """

    :param root: 数据的处理目录
    :type root:
    :param use_pkl: 是否使用已缓存的pkl读取
    :type use_pkl:
    :param seed: 随机数种子
    :type seed:
    :param dataset: 要处理的数据集，absa， dem8，purchase，还是all
    :type dataset:
    :return:
    :rtype:
    """
    assert os.path.exists(root), f"运行的路径是系统的根目录吗，请确保{root}文件夹存在"
    canonical_data_suffix = "canonical_data"
    canonical_data_root = os.path.join(root, canonical_data_suffix)
    if not os.path.isdir(canonical_data_root):
        os.mkdir(canonical_data_root)
    print(f"将要保存到:{canonical_data_root}, 是否使用缓存的文件:{use_pkl}, 使用的随机数种子是:{seed}")
    # 每个任务的数据的ids
    data_ids = {}
    datasets = dataset.split(',')
    for task_name, data_cf in data_configs.items():
        if all in datasets or task_name in datasets :
            print(f"开始处理任务{task_name}")
            ##############处理数据##############
            data_format = data_cf['DataFormat']
            do_truncate = data_cf['do_truncate']
            todict = data_cf['todict']
            only_addidx = data_cf.get('only_addidx')
            #保存成tsv文件
            data = load_absa_dem8(task_name=task_name, use_pickle=use_pkl, do_truncate=do_truncate)
            train_data, dev_data, test_data, train_data_id, dev_data_id, test_data_id = split_save_data(data=data,random_seed=seed, todict=todict,only_addidx=only_addidx)
            #保存文件
            train_fout = os.path.join(canonical_data_root, f'{task_name}_train.tsv')
            dev_fout = os.path.join(canonical_data_root, f'{task_name}_dev.tsv')
            test_fout = os.path.join(canonical_data_root, f'{task_name}_test.tsv')
            dump_rows(train_data, train_fout, data_format)
            dump_rows(dev_data, dev_fout, data_format)
            dump_rows(test_data, test_fout, data_format)
            logger.info(f'初步处理{task_name}数据完成, 保存规范后的数据到{train_fout}, {dev_fout}, {test_fout}')
            print()
            data_ids[task_name] = (train_data_id, dev_data_id, test_data_id)
    return data_ids

if __name__ == '__main__':
    args = parse_args()
    if args.save_source_pkl:
        if args.dataset == "all":
            for task in data_configs.keys():
                save_source_data(task_name=task)
        else:
            save_source_data(task_name=args.dataset)
    else:
        do_prepro(root=args.root_dir, use_pkl=args.use_pkl, seed=args.seed, dataset=args.dataset)
