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

absa_source_file = "data_my/canonical_data/source_data/absa.pkl"
dem8_source_file = "data_my/canonical_data/source_data/dem8.pkl"
purchase_source_file = "data_my/canonical_data/source_data/purchase.pkl"
brand_source_file = "data_my/canonical_data/source_data/brand.pkl"

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

def do_truncate_data(data, left_max_seq_len=60, aspect_max_seq_len=10, right_max_seq_len=60):
    """
    对数据做truncate
    :param data:针对不同类型的数据进行不同的截断
    :return:返回列表，是截断后的文本，aspect
    所以如果一个句子中有多个aspect关键字，那么就会产生多个截断的文本+关键字，组成的列表，会产生多个预测结果
    和locations: [(start_idx,end_idx),...]
    """
    def aspect_truncate(content,aspect,aspect_start,aspect_end):
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
                new_content = aspect_truncate(content, aspect, aspect_start, aspect_end)
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
            new_content = aspect_truncate(content, aspect, aspect_start,aspect_end)
            contents.append((new_content, aspect))
            locations.append((aspect_start, aspect_end))
            original_data.append(one_data)
        elif len(one_data) == 5:
            content, aspect, aspect_start, aspect_end, label = one_data
            new_content = aspect_truncate(content, aspect, aspect_start, aspect_end)
            contents.append((new_content, aspect, label))
            locations.append((aspect_start, aspect_end))
            original_data.append(one_data)
        elif len(one_data) == 7:
            content, aspect, aspect_start, aspect_end, label, channel,wordtype = one_data
            new_content = aspect_truncate(content, aspect, aspect_start, aspect_end)
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

def truncate_relation(data, max_seq_len=450):
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

def save_source_data(task_name="all"):
    sys.path.append('/Users/admin/git/TextBrewer/huazhuang/utils')
    from convert_label_studio_data import get_all, get_demision8, get_all_purchase, get_all_brand
    #保存三个数据集的原始数据，方便以后不从label-studio读取
    if task_name == "absa" or task_name == "all":
        absa_data = get_all(split=False, dirpath=f"/opt/lavector/absa", do_save=False, withmd5=True)
        pickle.dump(absa_data, open(absa_source_file, "wb"))
    if task_name == "dem8" or task_name == "all":
        dem8_data = get_demision8(split=False,
                                 dirpath_list=['/opt/lavector/effect', '/opt/lavector/pack', '/opt/lavector/promotion',
                                               '/opt/lavector/component', '/opt/lavector/fragrance','/opt/lavector/dem8_verify'],withmd5=True)
        pickle.dump(dem8_data, open(dem8_source_file, "wb"))
    if task_name == "purchase" or task_name == "all":
        purchase_data = get_all_purchase(dirpath=f"/opt/lavector/purchase", split=False, do_save=False,withmd5=True)
        pickle.dump(purchase_data, open(purchase_source_file, "wb"))
    if task_name == "brand" or task_name == "all":
        brand_data = get_all_brand(dirpath="/opt/lavector/relation/",split=False, do_save=False, withmd5=True)
        pickle.dump(brand_data, open(brand_source_file, "wb"))
    if task_name == "all":
        return absa_data, dem8_data, purchase_data, brand_data
    elif task_name == "absa":
        return absa_data
    elif task_name == "dem8":
        return dem8_data
    elif task_name == "brand":
        return brand_data
    elif task_name == "purchase":
        return purchase_data

def load_absa_dem8(kind='absa',left_max_seq_len=60, aspect_max_seq_len=10, right_max_seq_len=60, use_pickle=False, do_truncate=True):
    """
    Aspect Base sentiment analysis
    :param kind: 是加载absa数据，还是dem8的数据
    :param do_truncate: 是否做裁剪
    :return:
    :rtype:
    """
    if kind == 'absa':
        if use_pickle:
            assert os.path.exists(absa_source_file), "源数据的pickle文件不存在，请检查"
            with open(absa_source_file, 'rb') as f:
                all_data = pickle.load(f)
        else:
            all_data = save_source_data(task_name="absa")

        # 去除md5的位置，这个暂时不用
        all_data = [d[:-1] for d in all_data]
    elif kind == 'dem8':
        # 注意dirpath_list使用哪些数据进行训练，那么预测时，也是用这样的数据
        # labels2id = {
        #     "是": 0,
        #     "否": 1,
        # }
        if use_pickle:
            assert os.path.exists(dem8_source_file), "源数据的pickle文件不存在，请检查"
            with open(dem8_source_file, 'rb') as f:
                all_data = pickle.load(f)
        else:
            all_data = save_source_data(task_name="dem8")
        # 去除md5的位置，这个暂时不用
        all_data = [d[:-1] for d in all_data]
    elif kind == 'purchase':
        # 返回数据格式[(text, title, keyword, start_idx, end_idx, label),...]
        if use_pickle:
            assert os.path.exists(purchase_source_file), "源数据的pickle文件不存在，请检查"
            with open(purchase_source_file, 'rb') as f:
                a_data = pickle.load(f)
        else:
            a_data = save_source_data(task_name="purchase")
        # 把title+text拼接在一起
        all_data = []
        for d in a_data:
            title_len = len(d[1])
            text = d[1] + d[0]
            start_idx = d[3] + title_len
            end_idx = d[4] + title_len
            one_data = [text, d[2],start_idx,end_idx,d[5]]
            all_data.append(one_data)
    elif kind == 'brand':
        if use_pickle:
            assert os.path.exists(brand_source_file), "源数据的pickle文件不存在，请检查"
            with open(brand_source_file, 'rb') as f:
                all_data = pickle.load(f)
        else:
            all_data = save_source_data(task_name="brand")
    else:
        print("数据的种类不存在，退出")
        sys.exit(1)
    if do_truncate:
        if kind == 'brand':
            data = truncate_relation(data=all_data)
        else:
            original_data, data, locations = do_truncate_data(all_data,left_max_seq_len, aspect_max_seq_len, right_max_seq_len)
    else:
        data = all_data
    if kind == 'dem8':
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

def split_save_data(data, random_seed, train_rate=0.8, dev_rate=0.1, test_rate=0.1, todict=True):
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
    def change_data(kind_data):
        rows = []
        for idx, one_data in enumerate(kind_data):
            content, keyword, label = one_data
            # label_id = labels2id[label]
            # assert label in ['消极','中性','积极','是','否'], "label不是特定的关键字，那么my_task_def.yml配置文件中的labels就不能解析，会出现错误"
            sample = {'uid': idx, 'premise': content, 'hypothesis': keyword, 'label': label}
            rows.append(sample)
        return rows
    if todict:
        train_data = change_data(train_data)
        dev_data = change_data(dev_data)
        test_data = change_data(test_data)
    print(f"训练集数量{len(train_data)}, 开发集数量{len(dev_data)}, 测试集数量{len(test_data)}")
    return train_data, dev_data, test_data, train_data_id, dev_data_id, test_data_id


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing GLUE/SNLI/SciTail dataset.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--root_dir', type=str, default='data_my',help='数据集的保存路径')
    parser.add_argument('--dataset', type=str, default='all',help='默认处理哪个数据集，all代表所有')
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

    if dataset == 'all' or dataset == 'absa':
        ##############ABSA 的数据##############
        #保存成tsv文件
        data = load_absa_dem8(kind='absa', use_pickle=use_pkl)
        absa_train_data, absa_dev_data, absa_test_data, absa_train_data_id, absa_dev_data_id, absa_test_data_id = split_save_data(data=data,random_seed=seed)
        #保存文件
        absa_train_fout = os.path.join(canonical_data_root, 'absa_train.tsv')
        absa_dev_fout = os.path.join(canonical_data_root, 'absa_dev.tsv')
        absa_test_fout = os.path.join(canonical_data_root, 'absa_test.tsv')
        dump_rows(absa_train_data, absa_train_fout, DataFormat.PremiseAndOneHypothesis)
        dump_rows(absa_dev_data, absa_dev_fout, DataFormat.PremiseAndOneHypothesis)
        dump_rows(absa_test_data, absa_test_fout, DataFormat.PremiseAndOneHypothesis)
        logger.info(f'初步处理absa数据完成, 保存规范后的数据到{absa_train_fout}, {absa_dev_fout}, {absa_test_fout}')
        print()
    if dataset == 'all' or dataset == 'dem8':
        ##############8个维度的数据##############
        data = load_absa_dem8(kind='dem8', use_pickle=use_pkl)
        dem8_train_data, dem8_dev_data, dem8_test_data,dem8_train_data_id, dem8_dev_data_id, dem8_test_data_id = split_save_data(data=data,random_seed=seed)
        #保存文件
        dem8_train_fout = os.path.join(canonical_data_root, 'dem8_train.tsv')
        dem8_dev_fout = os.path.join(canonical_data_root, 'dem8_dev.tsv')
        dem8_test_fout = os.path.join(canonical_data_root, 'dem8_test.tsv')
        dump_rows(dem8_train_data, dem8_train_fout, DataFormat.PremiseAndOneHypothesis)
        dump_rows(dem8_dev_data, dem8_dev_fout, DataFormat.PremiseAndOneHypothesis)
        dump_rows(dem8_test_data, dem8_test_fout, DataFormat.PremiseAndOneHypothesis)
        logger.info(f'初步处理dem8数据完成, 保存规范后的数据到{dem8_train_fout}, {dem8_dev_fout}, {dem8_test_fout}')
        print()

    if dataset == 'all' or dataset == 'absa':
        ##############购买意向数据##############
        data = load_absa_dem8(kind='purchase', use_pickle=use_pkl)
        purchase_train_data, purchase_dev_data, purchase_test_data, purchase_train_data_id, purchase_dev_data_id, purchase_test_data_id = split_save_data(data=data,random_seed=seed)
        #保存文件
        purchase_train_fout = os.path.join(canonical_data_root, 'purchase_train.tsv')
        purchase_dev_fout = os.path.join(canonical_data_root, 'purchase_dev.tsv')
        purchase_test_fout = os.path.join(canonical_data_root, 'purchase_test.tsv')
        dump_rows(purchase_train_data, purchase_train_fout, DataFormat.PremiseAndOneHypothesis)
        dump_rows(purchase_dev_data, purchase_dev_fout, DataFormat.PremiseAndOneHypothesis)
        dump_rows(purchase_test_data, purchase_test_fout, DataFormat.PremiseAndOneHypothesis)
        logger.info(f'初步处理purchase数据完成, 保存规范后的数据到{purchase_train_fout}, {purchase_dev_fout}, {purchase_test_fout}')
        print()
    if dataset == 'all' or dataset == 'brand':
        data = load_absa_dem8(kind='brand', use_pickle=use_pkl, do_truncate=True)
        brand_train_data, brand_dev_data, brand_test_data, brand_train_data_id, brand_dev_data_id, brand_test_data_id = split_save_data(data=data,random_seed=seed, todict=False)
        #保存文件
        brand_train_fout = os.path.join(canonical_data_root, 'brand_train.tsv')
        brand_dev_fout = os.path.join(canonical_data_root, 'brand_dev.tsv')
        brand_test_fout = os.path.join(canonical_data_root, 'brand_test.tsv')
        dump_rows(brand_train_data, brand_train_fout, DataFormat.RELATION)
        dump_rows(brand_dev_data, brand_dev_fout, DataFormat.RELATION)
        dump_rows(brand_test_data, brand_test_fout, DataFormat.RELATION)
        logger.info(f'初步处理purchase数据完成, 保存规范后的数据到{brand_train_fout}, {brand_dev_fout}, {brand_test_fout}')
        print()
    if dataset == 'all':
        return (absa_train_data_id, absa_dev_data_id, absa_test_data_id), (dem8_train_data_id, dem8_dev_data_id, dem8_test_data_id), (purchase_train_data_id, purchase_dev_data_id, purchase_test_data_id), (brand_train_data_id, brand_dev_data_id, brand_test_data_id)
    elif dataset == 'brand':
        return brand_train_data_id, brand_dev_data_id, brand_test_data_id
    elif dataset == 'absa':
        absa_train_data_id, absa_dev_data_id, absa_test_data_id
    elif dataset == 'dem8':
        dem8_train_data_id, dem8_dev_data_id, dem8_test_data_id
    elif dataset == 'purchase':
        purchase_train_data_id, purchase_dev_data_id, purchase_test_data_id

if __name__ == '__main__':
    args = parse_args()
    if args.save_source_pkl:
        save_source_data()
    else:
        do_prepro(root=args.root_dir, use_pkl=args.use_pkl, seed=args.seed, dataset=args.dataset)
