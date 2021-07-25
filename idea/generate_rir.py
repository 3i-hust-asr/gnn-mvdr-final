import matplotlib.pyplot as plt
from scipy.io import wavfile
import rir_generator as rir
import scipy.signal as ss
import sounddevice as sd
import soundfile as sf
import numpy as np

import util
import nnet

def mix_wav(args):
    print('mix_wav')

    data, fs = sf.read('data/clean.wav', always_2d=True)
    # print(data.shape) # (243760, 1)

    h = rir.generate(
        c=340,                          # Sound velocity (m/s)
        fs=fs,                          # Sample rate (samples/s)
        r=[[4, 4, 4], [4, 4, 1.1]],     # Microphone position(s) [x y z] (m)
        s=[7, 8, 9],                    # Source position [x y z] (m)
        L=[8, 9, 10],                   # Room dimensions [x y z] (m)
        reverberation_time=0.5,         # Reverberation time (s)
        nsample=4096,                   # Number of output samples
    )


    # print(h.shape) # (4096, 2)
    signal = ss.convolve(h[:, None, :], data[:, :, None])
    # print(signal.shape) # (247855, 1, 2)

    # sd.default.samplerate = fs

    # sd.play(data / np.max(np.abs(data)), fs)
    # sd.wait()
    
    # sd.play(signal[:, 0, 0]/np.max(np.abs(signal[:, 0, 0])), fs)
    # sd.wait()

    plt.plot(data / np.max(np.abs(data)) , linewidth=0.5, label='clean')
    plt.plot(signal[:, 0, 0]/ np.max(np.abs(signal[:, 0, 0])), linewidth=0.5, label='reverb')

    plt.legend()
    plt.show()

    # loader = util.get_loader(args)[0]
    # augment_model = nnet.Augmentation(args)

    # for cleans, noises, rirs in loader:
    #     break

    # inputs, cleans, noises, clean_reverbs, noise_reverbs = augment_model(cleans, noises, rirs)

    # print(cleans.shape, clean_reverbs.shape)
    # tmp1 = cleans[0]
    # tmp2 = clean_reverbs[0,:,0]
    # plt.plot(tmp1/tmp1.abs().max(), linewidth=0.5, label='clean')
    # plt.plot(tmp2/tmp2.abs().max(), linewidth=0.5, label='reverb')
    # plt.legend()
    # plt.show()
