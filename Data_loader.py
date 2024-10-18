# %%
import codecs
import os
import random

import regex
import requests

# %% [markdown]
# regex库用来处理正则表达式：匹配，搜索，替换，拆分
# request库用来发送HTTP请求，实现网络通讯

# %%
min_cnt = 0
maxlen = 50

# %%
source_train = '../Data/cn.txt'
target_train = '../Data/en.txt'
source_test = '../Data/cn.test.txt'
target_test = '../Data/en.test.txt'

# %% [markdown]
# min_cnt为单词最小出现的频率，如果出现次数小于min_cnt则将单词encode为<UNK>
# 
# maxlen为一句话中最多的单词数量

# %%
def load_vocab(language):
    assert language in ['cn', 'en']
    vocab = [
        line.split()[0] for line in codecs.open(
            'Data/{}.txt.vocab.tsv'.format(language), 'r',
            'utf-8').read().splitlines() if int(line.split()[1]) >= min_cnt
    ]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

# %% [markdown]
# codecs.open()以规定形式打开文件
# read()方法读取文件所有内容
# splitline()方法将所有内容按行分割，生成一个包含每行内容的列表

# %%
def load_cn_vocab():
    word2idx, idx2word = load_vocab('cn')
    return word2idx, idx2word

def load_en_vocab():
    word2idx, idx2word = load_vocab('en')
    return word2idx, idx2word

# %%
def create_data(source_sents, target_sents):
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [
            cn2idx.get(word, 1)
            for word in ('<S> ' + source_sent + ' </S>').split()
        ]  # 1: OOV, </S>: End of Text
        y = [
            en2idx.get(word, 1)
            for word in ('<S> ' + target_sent + ' </S>').split()
        ]
        if max(len(x), len(y)) <= maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    # Pad
    X = np.zeros([len(x_list), maxlen], np.int32)
    Y = np.zeros([len(y_list), maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, maxlen - len(x)],
                        'constant',
                        constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, maxlen - len(y)],
                        'constant',
                        constant_values=(0, 0))
    return X, Y, Sources, Targets

# %%
def load_data(data_type):
    if data_type == 'train':
        source, target = source_train, target_train
    elif data_type == 'test':
        source, target = source_test, target_test
    assert data_type in ['train', 'test']
    cn_sents = [
        regex.sub("[^\s\p{L}']", '', line)  # noqa W605
        for line in codecs.open(source, 'r', 'utf-8').read().split('\n')
        if line and line[0] != '<'
    ]
    en_sents = [
        regex.sub("[^\s\p{L}']", '', line)  # noqa W605
        for line in codecs.open(target, 'r', 'utf-8').read().split('\n')
        if line and line[0] != '<'
    ]

    X, Y, Sources, Targets = create_data(cn_sents, en_sents)
    return X, Y, Sources, Targets


# %%
def load_train_data():
    X, Y, _, _ = load_data('train')
    return X, Y


def load_test_data():
    X, Y, _, _ = load_data('test')
    return X, Y


def get_batch_indices(total_length, batch_size):
    assert (batch_size <=
            total_length), ('Batch size is large than total data length.'
                            'Check your data or change batch size.')
    current_index = 0
    indexs = [i for i in range(total_length)]
    random.shuffle(indexs)
    while 1:
        if current_index + batch_size >= total_length:
            break
        current_index += batch_size
        yield indexs[current_index:current_index + batch_size], current_index


def idx_to_sentence(arr, vocab, insert_space=False):
    res = ''
    first_word = True
    for id in arr:
        word = vocab[id.item()]

        if insert_space and not first_word:
            res += ' '
        first_word = False

        res += word

    return res
