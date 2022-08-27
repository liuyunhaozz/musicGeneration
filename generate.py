import argparse
import os
import tensorflow as tf
from tqdm import tqdm

from Config import Config
from model import build_model
from Dataset import Songs
from musicPlay import play_song, save_song_to_abc




### Prediction of a generated song ###

def createabc(model, char2idx, idx2char, start_string="X", generation_length=1000):
  # Evaluation step (generating ABC text using the learned RNN model)

  '''TODO: convert the start string to numbers (vectorize)'''
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Here batch size == 1
  model.reset_states()
  tqdm._instances.clear()

  for i in tqdm(range(generation_length)):
      predictions = model(input_eval)
      
      # Remove the batch dimension
      predictions = tf.squeeze(predictions, 0)
      
      '''TODO: use a multinomial distribution to sample'''
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      
      # Pass the prediction along with the previous hidden state
      #   as the next inputs to the model
      input_eval = tf.expand_dims([predicted_id], 0)
      
      '''TODO: add the predicted character to the generated text!'''
      # Hint: consider what format the prediction is in vs. the output
      text_generated.append(idx2char[predicted_id])
    
  return (start_string + ''.join(text_generated))



def generate(data, modelfile, startstr, length):
    songs = Songs(data)
    songs_joined = songs.join_song(songs.songs_list)
    songs.generate_map(songs_joined)
    # print(vectorized_list.shape)
    opt = Config(len(songs.vocab))
    model = build_model(opt.vocab_size, opt.embedding_dim, opt.rnn_units, batch_size=1)

    # Restore the model weights for the last checkpoint after training
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
        play_song(name)




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="")
    p.add_argument("--modelfile", type=str, default="")
    p.add_argument("--startstr", type=str, default="")
    p.add_argument("--length", type=int, default=1000)
    # p.add_argument("--lr", type=float, default=0.00001)
    args = p.parse_args()
    generate(args.data, args.modelfile, args.startstr, args.length)
