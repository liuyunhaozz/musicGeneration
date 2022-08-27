import os
from tracemalloc import start
from IPython.display import Audio

def save_song_to_abc(song, filename="tmp"):
    with open(filename, "w") as f:
        f.write(song)
    return filename

def abc2wav(abc_file):
    suf = abc_file.rstrip('.abc')
    cmd = "abc2midi {} -o {}".format(abc_file, suf + ".mid")
    os.system(cmd)
    cmd = "timidity {}.mid -Ow {}.wav".format(suf, suf)
    return suf + ".mid"

def play_song(abc_file):
    wav_file = abc2wav(abc_file)
    cmd = "start {}".format(wav_file)
    return os.system(cmd)

# file = "test.wav"
# os.system("start " + file)
