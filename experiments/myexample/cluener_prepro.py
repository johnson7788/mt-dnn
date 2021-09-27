import json
import os
import argparse
from sys import path
path.append(os.getcwd())
from data_utils.task_def import DataFormat
from data_utils.log_wrapper import create_logger
from experiments.ner.ner_utils import load_conll_chunk, load_conll_ner, load_conll_pos
from experiments.common_utils import dump_rows
logger = create_logger(__name__, to_disk=True, log_file='bert_ner_data_proc_512_cased.log')

def parse_args():
    parser = argparse.ArgumentParser(description='处理中文CLUE的NER数据集')
    parser.add_argument('--data_dir', type=str, default='data_my/cluener', help="源数据的目录")
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--output_dir', type=str, default='data_my/canonical_data',help='处理完成的数据集的保存路径')
    args = parser.parse_args()
    return args


def load_clue_ner(file):
    """
    处理ner的数据
    数据分为10个标签类别，分别为:
    地址（address），
    书名（book），
    公司（company），
    游戏（game），
    政府（government），
    电影（movie），
    姓名（name），
    组织机构（organization），
    职位（position），
    景点（scene）
    :param file:
    :type file:
    :return:
    :rtype:
    """
    gold_labels = ["address", "book", "company", "game", "government", "movie", "name", "organization", "position", "scene"]
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            line_dict = json.loads(line)
            text = line_dict['text']
            label_dict = line_dict['label']
            sentence = list(text)
            #初始化默认的标签都是O
            label = ['O'] * len(sentence)
            for lname, lvalue in label_dict.items():
                # eg: lname: company,
                for keyword, positions in lvalue.items():
                    for position in positions:
                        start, end = position
                        assert text[start:end+1] == keyword, "这个位置在原文中对应的关键字和给定的label中不一致"
                        assert lname in gold_labels, f"这个标签{lname}不在我们的标签列表，请检查"
                        # 变成大写，加以区分
                        lname_upper = lname.upper()
                        # BIO的标注规则,开始位置变成  eg: B-COMPANY,  I-COMPANY
                        label[start] = f"B-{lname_upper}"
                        label[start+1:end+1] = [f"I-{lname_upper}"] * (end-start)
                        # 根据lname
            # 计数
            cnt += 1
            # 一条样本
            sample = {'uid': cnt, 'premise': sentence, 'label': label}
            rows.append(sample)
    return rows

def main(args):
    data_dir = args.data_dir
    data_dir = os.path.abspath(data_dir)
    if not os.path.exists(data_dir):
        raise Exception("确定是在项目的跟目标运行的吗？")
    #源文件位置
    train_path = os.path.join(data_dir, 'train.json')
    dev_path = os.path.join(data_dir, 'dev.json')
    # 测试集不处理了
    train_data = load_clue_ner(train_path)
    dev_data = load_clue_ner(dev_path)
    logger.info('训练集加载了 {}条 NER样本'.format(len(train_data)))
    logger.info('开发集加载了 {}条 NER样本'.format(len(dev_data)))

    bert_root = args.output_dir
    if not os.path.isdir(bert_root):
        os.mkdir(bert_root)
    train_fout = os.path.join(bert_root, 'cluener_train.tsv')
    dev_fout = os.path.join(bert_root, 'cluener_dev.tsv')

    dump_rows(train_data, train_fout, DataFormat.Sequence)
    dump_rows(dev_data, dev_fout, DataFormat.Sequence)
    logger.info('CLUE NER数据处理完成')

if __name__ == '__main__':
    args = parse_args()
    main(args)