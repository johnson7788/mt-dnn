import os
import argparse
import random
from sys import path

path.append(os.getcwd())
from experiments.common_utils import dump_rows
from data_utils.task_def import DataFormat
from data_utils.log_wrapper import create_logger
import sys

logger = create_logger(__name__, to_disk=True, log_file='mydata_prepro.log')

sys.path.append('/Users/admin/git/TextBrewer/huazhuang/utils')
from convert_label_studio_data import get_all_and_weibo_75, get_all, get_demision8, do_truncate_data

def load_absa_dem8(kind='absa', train_rate=0.8, dev_rate=0.1, test_rate=0.1, left_max_seq_len=50, aspect_max_seq_len=8, right_max_seq_len=50):
    """
    Aspect Base sentiment analysis
    :param kind: 是加载absa数据，还是dem8的数据
    :return:
    :rtype:
    """
    if kind == 'absa':
        all_data = get_all(split=False, dirpath=f"/opt/lavector/absa", do_save=False)
        # labels2id = {
        #     "消极": 0,
        #     "中性": 1,
        #     "积极": 2,
        # }
    elif kind == 'dem8':
        # 注意dirpath_list使用哪些数据进行训练，那么预测时，也是用这样的数据
        # labels2id = {
        #     "是": 0,
        #     "否": 1,
        # }
        all_data = get_demision8(split=False, dirpath_list=['/opt/lavector/effect', '/opt/lavector/pack', '/opt/lavector/promotion','/opt/lavector/component', '/opt/lavector/fragrance'])
    else:
        print("数据的种类不存在，退出")
        sys.exit(1)
    original_data, data, locations = do_truncate_data(all_data,left_max_seq_len, aspect_max_seq_len, right_max_seq_len)
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
    random.seed(30)
    random.shuffle(data)
    total = len(data)
    train_num = int(total * train_rate)
    dev_num = int(total * dev_rate)
    test_num = int(total * test_rate)
    train_data = data[:train_num]
    dev_data = data[train_num:train_num+dev_num]
    test_data = data[train_num+dev_num:]
    # 处理一下，保存的格式
    def change_data(kind_data):
        rows = []
        for idx, one_data in enumerate(kind_data):
            content, keyword, label = one_data
            # label_id = labels2id[label]
            sample = {'uid': idx, 'premise': content, 'hypothesis': keyword, 'label': label}
            rows.append(sample)
        return rows
    absa_train_data = change_data(train_data)
    absa_dev_data = change_data(dev_data)
    absa_test_data = change_data(test_data)
    return absa_train_data, absa_dev_data, absa_test_data


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing GLUE/SNLI/SciTail dataset.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--root_dir', type=str, default='data_my')
    args = parser.parse_args()
    return args


def main(args):
    root = args.root_dir
    assert os.path.exists(root), f"运行的路径是系统的根目录吗，请确保{root}文件夹存在"
    canonical_data_suffix = "canonical_data"
    canonical_data_root = os.path.join(root, canonical_data_suffix)
    if not os.path.isdir(canonical_data_root):
        os.mkdir(canonical_data_root)
    print(f"将要保存到{canonical_data_root}")

    ##############ABSA 的数据##############
    #保存成tsv文件
    absa_train_data, absa_dev_data, absa_test_data = load_absa_dem8(kind='absa')
    #保存文件
    absa_train_fout = os.path.join(canonical_data_root, 'absa_train.tsv')
    absa_dev_fout = os.path.join(canonical_data_root, 'absa_dev.tsv')
    absa_test_fout = os.path.join(canonical_data_root, 'absa_test.tsv')
    dump_rows(absa_train_data, absa_train_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(absa_dev_data, absa_dev_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(absa_test_data, absa_test_fout, DataFormat.PremiseAndOneHypothesis)
    logger.info(f'初步处理absa数据完成, 保存规范后的数据到{absa_train_fout}, {absa_dev_fout}, {absa_test_fout}')

    ##############8个维度的数据##############
    dem8_train_data, dem8_dev_data, dem8_test_data = load_absa_dem8(kind='dem8')
    #保存文件
    dem8_train_fout = os.path.join(canonical_data_root, 'dem8_train.tsv')
    dem8_dev_fout = os.path.join(canonical_data_root, 'dem8_dev.tsv')
    dem8_test_fout = os.path.join(canonical_data_root, 'dem8_test.tsv')
    dump_rows(dem8_train_data, dem8_train_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(dem8_dev_data, dem8_dev_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(dem8_test_data, dem8_test_fout, DataFormat.PremiseAndOneHypothesis)
    logger.info(f'初步处理dem8数据完成, 保存规范后的数据到{dem8_train_fout}, {dem8_dev_fout}, {dem8_test_fout}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
