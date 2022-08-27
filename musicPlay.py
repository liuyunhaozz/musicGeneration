import os
from IPython.display import Audio

def save_song_to_abc(song, filename="tmp"):
    save_name = "{}.abc".format(filename)
    with open(save_name, "w") as f:
        f.write(song)
    return filename

def abc2wav(abc_file):
    suf = abc_file.rstrip('.abc')
    cmd = "abc2midi {} -o {}".format(abc_file, suf + ".mid")
    os.system(cmd)
    cmd = "timidity {}.mid -Ow {}.wav".format(suf, suf)
    return os.system(cmd) 

def play_wav(wav_file):
    # f = open("demofile3.wav", "w")
    # f.write(wav_file)
    # f.close()
    return Audio(wav_file)

def play_song(song):
    basename = save_song_to_abc(song)
    print(basename)
    ret = abc2wav(basename + '.abc')
    if ret == 0: #did not suceed
        print('yes')
        return play_wav(basename+'.wav')
    return None


# file = "test.wav"
# os.system("start " + file)
