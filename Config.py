import os


"""
配置类，存放训练需要用到的超参数
"""
class Config(object):

    def __init__(self, vocab_length):
        self.vocab_size = vocab_length

    # 优化参数
    num_training_iterations = 2000  # Increase this to train longer
    batch_size = 4  # Experiment between 1 and 64
    seq_length = 100  # Experiment between 50 and 500
    learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

    # 模型参数
    embedding_dim = 256 
    rnn_units = 1024  # Experiment between 1 and 2048

    # 配置文件存放位置
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")