import os
import argparse
import random
from sys import path

path.append(os.getcwd())
from experiments.common_utils import dump_rows
from data_utils.task_def import DataFormat
from data_utils.log_wrapper import create_logger
import sys
sys.path.append('/Users/admin/git/TextBrewer/huazhuang/utils')
from convert_label_studio_data import get_all_and_weibo_75, get_all, get_demision8, do_truncate_data

logger = create_logger(__name__, to_disk=True, log_file='mydata_prepro.log')

def load_absa(train_rate=0.8, dev_rate=0.1, test_rate=0.1, left_max_seq_len=50, aspect_max_seq_len=8, right_max_seq_len=50):
    """
    Aspect Base sentiment analysis
    :return:
    :rtype:
    """
        # 注意dirpath_list使用哪些数据进行训练，那么预测时，也是用这样的数据
    # get_demision8(save_path='../data_root_dir/demision8',
    #               dirpath_list=['/opt/lavector/effect', '/opt/lavector/pack', '/opt/lavector/promotion',
    #                             '/opt/lavector/component', '/opt/lavector/fragrance'])
    all_data = get_all(dirpath=f"/opt/lavector/absa", do_save=False)
    original_data, data, locations = do_truncate_data(all_data,left_max_seq_len, aspect_max_seq_len, right_max_seq_len)
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
            content, keyword, start_idx, stop_idx, sentiment, channel, word_type = one_data
            sample = {'uid': idx, 'premise': content, 'hypothesis': keyword, 'label': sentiment}
            rows.append(sample)
        return rows
    absa_train_data = change_data(train_data)
    absa_dev_data = change_data(dev_data)
    absa_test_data = change_data(test_data)
    return absa_train_data, absa_dev_data, absa_test_data

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing GLUE/SNLI/SciTail dataset.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--old_glue', action='store_true', help='whether it is old GLUE, refer official GLUE webpage for details')
    args = parser.parse_args()
    return args


def main(args):
    root = args.root_dir
    assert os.path.exists(root)
    canonical_data_suffix = "canonical_data"
    canonical_data_root = os.path.join(root, canonical_data_suffix)
    if not os.path.isdir(canonical_data_root):
        os.mkdir(canonical_data_root)
    print(f"将要保存到{canonical_data_root}")
    #保存成tsv文件
    absa_train_data, absa_dev_data, absa_test_data = load_absa()
    #保存文件
    abas_train_fout = os.path.join(canonical_data_root, 'abas_train.tsv')
    abas_dev_fout = os.path.join(canonical_data_root, 'abas_dev.tsv')
    abas_test_fout = os.path.join(canonical_data_root, 'abas_test.tsv')
    dump_rows(absa_train_data, abas_train_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(absa_dev_data, abas_dev_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(absa_test_data, abas_test_fout, DataFormat.PremiseAndOneHypothesis)
    logger.info('初步处理absa数据完成')

if __name__ == '__main__':
    args = parse_args()
    main(args)
