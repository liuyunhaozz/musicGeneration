import os
import re
import numpy as np

class Songs(object):
    songs_list = []

    """
    从当前目录读取abc格式文件，返回元素为每首乐曲的列表
    """
    def __init__(self, name):
        with open(os.path.join(os.getcwd(), 'dataset', name), 'r') as f:
            text = f.read()
            self.songs_list = self.extract_song_snippet(text)

    """
    利用正则表达式对输入的.abc格式文件进行处理，提取出每首乐曲，返回乐曲的列表
    """
    def extract_song_snippet(self, text):
        pattern = '(^|\n\n)(.*?)\n\n'
        search_results = re.findall(pattern, text, flags=re.MULTILINE | re.DOTALL)
        # print(search_results)
        songs = [song[1] for song in search_results]
        print("Found {} songs in text".format(len(songs)))
        return songs

    """
    将包含歌曲的列表转为一个字符串
    """
    def join_song(self, songs):
        songs_joined = "\n\n".join(songs)
        return songs_joined

    """
    对特定数据集生成字符串到数字的映射
    """
    def generate_map(self, songs_joined):
        self.vocab = sorted(set(songs_joined))
        print("There are", len(self.vocab), "unique characters in the dataset")
        self.char2idx = {u:i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab) 

    """
    将字符串转为对应的numpy数字向量
    """
    def vectorize_string(self, songs_joined):
        vectorized_list = np.array([self.char2idx[s] for s in songs_joined])
        return vectorized_list

    
    """
    创建训练数据集
    """
    def get_batch(self, vectorized_list, seq_length, batch_size):
        # the length of the vectorized songs string
        n = vectorized_list.shape[0] - 1
        # randomly choose the starting indices for the examples in the training batch
        idx = np.random.choice(n-seq_length, batch_size)
        # print(idx)
        # print(n - seq_length)

        input_batch = [vectorized_list[i:i+seq_length] for i in idx]
        # print(input_batch)
        output_batch = [vectorized_list[i+1: i+1+seq_length] for i in idx]
        # print(output_batch)

        # x_batch, y_batch provide the true inputs and targets for network training
        x_batch = np.reshape(input_batch, [batch_size, seq_length])
        y_batch = np.reshape(output_batch, [batch_size, seq_length])
        return x_batch, y_batch
    