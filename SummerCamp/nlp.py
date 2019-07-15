# -*- coding: utf-8 -*-
"""
* @Author: ziuno
* @Software: PyCharm
* @Time: 2019/7/3 10:26
"""

import os
import string

import matplotlib.pyplot as plt
import numpy as np
import scipy
import zhon.hanzi as hanzi
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import CountVectorizer

stop_word = list(set(stopwords.words('english')))


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


def __lemmatizer(s):
    """
    词性归一化
    :param s: 原始文本字符串
    :return: 翻译为统一词性的文本串
    """
    s = s.split(' ')
    for i, item in enumerate(s):
        s[i] = item[:-1] if item.endswith('s') else item
        s[i] = item[:-2] if item.endswith('er') else item
    wordnet_lemmatizer = WordNetLemmatizer()
    s = [wordnet_lemmatizer.lemmatize(i, pos='v') for i in s]
    s = ' '.join(s)
    return s


def __remove_stop_word(s):
    """
    去除停用词
    :param s: 原始文本字符串
    :return: 去除停用词的文本字符串
    """
    s = word_tokenize(s)
    pos = pos_tag(s)
    s = [item for i, item in enumerate(s) if not item in stop_word and pos[i][1] != 'POS']

    return ' '.join(s)


def participle(s):
    """
    处理原始文本串，进行分词操作
    :param s: 原始文本串
    :return: 处理后的的文本串
    """
    en_trantab = str.maketrans({key: None for key in string.punctuation})
    ch_trantab = str.maketrans({key: " " for key in hanzi.punctuation})
    s = s.lower()
    s = __remove_stop_word(s)
    s = s.translate(en_trantab)
    s = s.translate(ch_trantab)
    s = __remove_stop_word(s)
    s = __lemmatizer(s)
    s = __remove_stop_word(s)
    return s


def vectorization(v):
    """
    当前文本列表向量化
    :param v: 文本列表
    :return: 向量化后的列表
    """
    tfidf = CountVectorizer()  # 可以更改为其他的向量化方法
    rho = tfidf.fit_transform(v).toarray()
    return rho


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
    ic0_p = participle(ic0)
    ic1_p = participle(ic1)
    v0_p, v1_p = vectorization([ic0_p, ic1_p])
    sim = similarity(v0_p, v1_p)
    output.append(sim)
    print("i: %d, similarity: %f, output: %f, s1: %s, s2: %s" % (index, sim, golden_content[index], ic0_p, ic1_p))
r, p = pearson(output, golden_content)
print(r)
plt.scatter(golden_content, output, s=10)
plt.show()
s = []
for i, item in enumerate(output):
    s.append('%f\t%s\t%s' % (item, input_content[i][0], input_content[i][1]))
    # s.append('%f' % item)
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(s))
