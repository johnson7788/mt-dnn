import os
import argparse
import random
from sys import path

path.append(os.getcwd())
from experiments.common_utils import dump_rows
from data_utils.task_def import DataFormat
from data_utils.log_wrapper import create_logger
import sys
import glob
import json

sys.path.append('/Users/admin/git/label-studio/myexample')
from papertext_api import export_data

logger = create_logger(__name__, to_disk=True, log_file='paper_prepro.log')


def collect_json(dirpath):
    """
    收集目录下的所有json文件，合成一个大的列表
    :param dirpath:如果是目录，那么搜索所有json文件，如果是文件，那么直接读取文件
    :return: 返回列表
    """
    #所有文件的读取结果
    data = []
    if os.path.isdir(dirpath):
        search_file = os.path.join(dirpath, "*.json")
        # 搜索所有json文件
        json_files = glob.glob(search_file)
    else:
        json_files = [dirpath]
    for file in json_files:
        with open(file, 'r') as f:
            file_data = json.load(f)
            print(f"{file}中包含数据{len(file_data)} 条")
            data.extend(file_data)
    print(f"共收集数据{len(data)} 条")
    return data

def load_papertext(train_rate=0.8, dev_rate=0.1, test_rate=0.1, max_length=50, download_from_label_studio=True):
    """
    Aspect Base sentiment analysis
    :param kind: 是加载papertext数据，还是dem8的数据
    :return:
    :rtype:
    """
    export_dir = "/opt/nlp/data/papertext/"
    if download_from_label_studio:
        json_path = export_data(hostname='http://127.0.0.1:8080/api/', dirpath=export_dir, jsonfile="0707.json")
    data = collect_json(dirpath=export_dir)
    valid_data = []
    for one in data:
        for complete in one['completions']:
            if complete.get('was_cancelled'):
                # 被取消了，那么跳过
                continue
            else:
                # 只取第一个标注结果就行了，我们只有一个标注结果
                if complete['result']:
                    result_one = complete['result'][0]
                    label = result_one['value']['choices'][0]
                    location = one['data']['location']
                    location = location.replace('行数','lines num').replace('段落宽度','paragraph width').replace('段落高度','paragraph height').replace('页面宽','page width').replace('页面高','page height')
                    text = one['data']['text']
                    valid_data.append([text,location,label])
    print(f'从总的数据{len(data)}中, 共收集到有效数据{len(valid_data)}')
    random.seed(30)
    random.shuffle(valid_data)
    total = len(valid_data)
    train_num = int(total * train_rate)
    dev_num = int(total * dev_rate)
    test_num = int(total * test_rate)
    train_data = valid_data[:train_num]
    dev_data = valid_data[train_num:train_num+dev_num]
    test_data = valid_data[train_num+dev_num:]
    # 处理一下，保存的格式
    def change_data(kind_data):
        rows = []
        for idx, one_data in enumerate(kind_data):
            content, location, label = one_data
            # label_id = labels2id[label]
            assert label in ['作者','页眉','页脚','段落','标题','参考','表格','图像','公式','其它'], "label不是特定的关键字，那么paper_task_def.yml配置文件中的labels就不能解析，会出现错误"
            sample = {'uid': idx, 'premise': content, 'hypothesis': location, 'label': label}
            rows.append(sample)
        return rows
    papertext_train_data = change_data(train_data)
    papertext_dev_data = change_data(dev_data)
    papertext_test_data = change_data(test_data)
    return papertext_train_data, papertext_dev_data, papertext_test_data


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing paper text dataset.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--root_dir', type=str, default='data_my')
    args = parser.parse_args()
    return args


def main(args):
    root = args.root_dir
    assert os.path.exists(root), f"运行的路径是系统的根目录吗，请确保{root}文件夹存在"
    canonical_data_suffix = "paper_data"
    canonical_data_root = os.path.join(root, canonical_data_suffix)
    if not os.path.isdir(canonical_data_root):
        os.mkdir(canonical_data_root)
    print(f"将要保存到{canonical_data_root}")

    ##############papertext 的数据##############
    #保存成tsv文件
    papertext_train_data, papertext_dev_data, papertext_test_data = load_papertext()
    #保存文件
    papertext_train_fout = os.path.join(canonical_data_root, 'papertext_train.tsv')
    papertext_dev_fout = os.path.join(canonical_data_root, 'papertext_dev.tsv')
    papertext_test_fout = os.path.join(canonical_data_root, 'papertext_test.tsv')
    dump_rows(papertext_train_data, papertext_train_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(papertext_dev_data, papertext_dev_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(papertext_test_data, papertext_test_fout, DataFormat.PremiseAndOneHypothesis)
    logger.info(f'初步处理papertext数据完成, 保存规范后的数据到{papertext_train_fout}, {papertext_dev_fout}, {papertext_test_fout}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
