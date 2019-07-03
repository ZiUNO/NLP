# -*- coding: utf-8 -*-
"""
* @Author: ziuno
* @Software: PyCharm
* @Time: 2019/7/3 10:26
"""

from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import pearsonr
import os
import string
import zhon.hanzi as hanzi
import numpy as np
import matplotlib.pyplot as plt
import scipy


def get_data(path):
    """
    获取人工打分结果和原始输入数据
    :param path: 含有input和golden文件的路径
    :return:golden文件内容&input文件内容
    """
    path = os.path.join(path)
    golden_file = os.path.join(path, 'golden.txt')
    input_file = os.path.join(path, 'input.txt')
    with open(golden_file, 'r') as f:
        g_content = f.readlines()
    with open(input_file, 'r', encoding='utf-8') as f:
        i_content = f.readlines()
    g_content = [float(g.strip()) for g in g_content]
    i_content = [i.strip().split('\t') for i in i_content]
    return g_content, i_content


def remove_useless_info(s):
    """
    去除中英文标点及停用词
    :param s: 原始字符串
    :return: 处理后的的字符串
    """
    en_trantab = str.maketrans({key: None for key in string.punctuation})
    ch_trantab = str.maketrans({key: None for key in hanzi.punctuation})
    s = s.translate(en_trantab)
    s = s.translate(ch_trantab)
    return s


def vectorization(v):
    """
    当前文本列表向量化
    :param v: 文本列表
    :return: 向量化后的列表
    """
    tfidf = CountVectorizer()  # 可以更改为其他的向量化方法
    re = tfidf.fit_transform(v).toarray()
    return re


def similarity(v1, v2):
    """
    相似度
    :param v1:向量1
    :param v2: 向量2
    :return: 相似度值
    """
    v1 = np.mat(v1)
    v2 = np.mat(v2)
    num = float(v1 * v2.T)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    sim = 0.5 + 0.5 * num / den
    return sim


def pearson(l1, l2):
    """
    皮尔森相关系数
    :param l1: 数值列表1
    :param l2: 数值列表2
    :return: 相关系数
    """
    l1 = scipy.array(l1)
    l2 = scipy.array(l2)
    return pearsonr(l1, l2)


dir_path = r'G:\PycharmWorkspace\NLP'
golden_content, input_content = get_data(dir_path)
output = []
for index, ic in enumerate(input_content):
    ic0 = ic[0]
    ic1 = ic[1]
    ic0_p = remove_useless_info(ic0)
    ic1_p = remove_useless_info(ic1)
    v0_p, v1_p = vectorization([ic0_p, ic1_p])
    sim = similarity(v0_p, v1_p)
    output.append(sim)
    print("similarity: %f, output: %f" % (sim, golden_content[index]))
r, p = pearson(output, golden_content)
print(r)
plt.scatter(golden_content, output, s=10)
plt.show()
