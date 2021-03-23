import os
import yaml
import os.path as osp

"""
加载配置文件，读取配置文件中数据集的路径
"""


def get_config(dir):
    # add direction join function when parse the yaml file  解析yaml文件时添加方向联接功能
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.sep.join(seq)

    # add string concatenation function when parse the yaml file  解析yaml文件时添加字符串连接功能
    def concat(loader, node):
        seq = loader.construct_sequence(node)
        return ''.join(seq)

    yaml.add_constructor('!join', join)
    yaml.add_constructor('!concat', concat)
    with open(dir, 'r') as f:
        cfg = yaml.load(f)

    return cfg


def check_dir(folder):
    if not osp.exists(folder):
        os.mkdir(folder)
