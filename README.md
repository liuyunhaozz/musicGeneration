# musicGeneration

This repo contains the code to train a RNN mode and use this model to generate music in format `midi` and `wav`. The code has been tested with `Python 3.7` on `Windows 10` and `Google Colab`.

---

### Get Started with Google Colab

`Jupyter notebook` is a convenient tool to display the training process and training results. In `train_at_colab.ipynb`, we invoke the commands to train the model and generate the `ABC` music file. Finally we use the `IPython` module to display the HTML page to display the staves and play the music.

Open the notebook [train_at_colab.ipynb](https://colab.research.google.com/github/liuyunhaozz/musicGeneration/blob/master/train_at_colab.ipynb) in Google Colab and run all cells to get you music.

---

### Train

1. Put `abc` notation music file into folder `dataset/`
2. run `python -r requirements.txt ` to install the packages.
3. run `train.py` with the flag below.

```sh
usage: train.py [-h] [--data DATA]

optional arguments:
  -h, --help   show this help message and exit
  --data DATA  The name of abc notation music file in dataset/
```

​	eg:  `python train.py --data music.txt`

---

### Generate

1. Put `.abc` notation music file into folder `dataset`
2. Put `.ckpt` format model into folder `training_checkpoints/`
3. run `generate.py` with the flag below

```sh
usage: generate.py [-h] [--data DATA] [--modelfile MODELFILE] [--startstr STARTSTR]
                   [--length LENGTH]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           The name of abc notation music file in dataset/
  --modelfile MODELFILE
                        model file in training_checkpoints/, eg:my_ckpt
  --startstr STARTSTR   A random string you want the music to start with
  --length LENGTH       The length of the music you want to generate
```

eg: `python generate.py --data music.txt --model my_ckpt --startstr X --length 1000`

---

### Note

- The project need `abc2midi` and `timidity` to convert the music from `abc` notation to `mid` and `wav`. Make sure you install them on your OS before you start generating music. 







