import tensorflow as tf
import argparse
from tqdm import tqdm

from Dataset import Songs
from Config import Config
from model import build_model
from visualize import PeriodicPlotter


def train(name):
    songs = Songs(name)
    songs_joined = songs.join_song(songs.songs_list)
    songs.generate_map(songs_joined)
    vectorized_list = songs.vectorize_string(songs_joined)
    # print(vectorized_list.shape)
    print(len(songs.songs_list))

    opt = Config(len(songs.vocab))

    model = build_model(len(songs.vocab), opt.embedding_dim, opt.rnn_units, opt.batch_size)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(opt.learning_rate)
    lossfunc = tf.keras.losses.sparse_categorical_crossentropy

    history = []
    plotter = PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
    
    if hasattr(tqdm, '_instances'): 
        tqdm._instances.clear()

    for iter in tqdm(range(opt.num_training_iterations)):
        x_batch, y_batch = songs.get_batch(vectorized_list, opt.seq_length, opt.batch_size)
        # print(x_batch.shape)
        
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = lossfunc(y_batch, y_pred, from_logits=True)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        history.append(loss.numpy().mean())
        plotter.plot(history)

        if iter % 100 == 0:     
            model.save_weights(opt.checkpoint_prefix)
        
    model.save_weights(opt.checkpoint_prefix)




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="", help="The name of abc notation music file in dataset/")
    args = p.parse_args()
    train(args.data)
