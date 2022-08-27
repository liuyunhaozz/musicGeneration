import argparse
import os
import tensorflow as tf
from tqdm import tqdm

from Config import Config
from model import build_model
from Dataset import Songs
from musicPlay import save_midi, save_song_to_abc




"""
利用已经训练的模型model生成abc格式音乐
"""
def createabc(model, char2idx, idx2char, start_string="X", generation_length=1000):

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    model.reset_states()
    tqdm._instances.clear()

    for i in tqdm(range(generation_length)):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    
    return (start_string + ''.join(text_generated))



def generate(data, modelfile, startstr, length):
    songs = Songs(data)
    songs_joined = songs.join_song(songs.songs_list)
    songs.generate_map(songs_joined)
    # print(vectorized_list.shape)
    opt = Config(len(songs.vocab))
    model = build_model(opt.vocab_size, opt.embedding_dim, opt.rnn_units, batch_size=1)

    model.load_weights(os.path.join(opt.checkpoint_dir, modelfile))
    model.build(tf.TensorShape([1, None]))

    model.summary()
    generated_text = createabc(model, songs.char2idx, songs.idx2char, startstr, length)
    generate_songs = songs.extract_song_snippet(generated_text)
    songdir = 'songs'
    if not os.path.exists(songdir):
        os.makedirs(songdir)
    for i, song in enumerate(generate_songs):
        name = os.path.join(songdir, str(i) + '.abc')
        save_song_to_abc(song, name)
        save_midi(name)




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="", help="The name of abc notation music file in dataset/")
    p.add_argument("--modelfile", type=str, default="", help="model file in training_checkpoints/, eg:my_ckpt")
    p.add_argument("--startstr", type=str, default="", help="A random string you want the music to start with")
    p.add_argument("--length", type=int, default=1000, help="The length of the music you want to generate")

    args = p.parse_args()
    generate(args.data, args.modelfile, args.startstr, args.length)
