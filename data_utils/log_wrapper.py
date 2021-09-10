# Copyright (c) Microsoft. All rights reserved.
import logging
from time import gmtime, strftime
import sys

def create_logger(name, silent=False, to_disk=False, log_file=None):
    """
    创建日志配置，默认是Debug模式
    :param name:  logger的名称，在日志中显示
    :type name: 必须提供
    :param silent: 是否打印日志到标准输出
    :type silent: bool
    :param to_disk: 是否保存日志到文件
    :type to_disk: bool
    :param log_file: 保存日志的文件的文件名，默认是根据时间创建
    :type log_file: None 或者文件名
    :return:
    :rtype:
    """
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    # 可选fmt格式： %(filename)s %(funcName)s  %(name)s %(pathname)s 等
    # 日志输出的格式, 可选 fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log
