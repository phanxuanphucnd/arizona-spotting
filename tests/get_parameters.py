# from arizona_spotting.models import Wav2KWS

# model = Wav2KWS(
#     num_classes=2,
#     encoder_hidden_dim=768,
#     out_channels=112,
#     pretrained_model='wav2vec-base-en'
# )

# a = sum([p.numel() for p in model.parameters()])

# print(f"\nThe total parameters of the model is {a}")

filename = 'data/gsc_v2.1/test/active/23mix_noise_ee6163d5_nohash_1.wav'

from random import sample
import numpy as np
import soundfile as sf
from pydub import AudioSegment

audio_clip = AudioSegment.from_wav(filename)

# # this is an array
samples = audio_clip.get_array_of_samples()
print(type(samples), type(samples[0]), np.array(samples.tolist()).dtype.itemsize)

wav, curr_sample_rate = sf.read(filename)

print('\n', type(wav), type(wav[0]), wav.dtype.itemsize)

# print(samples == wav)

# print(samples)
# print(wav)

from scipy.io.wavfile import read, write

sr, data = read('data/gsc_v2.1/test/active/23mix_noise_ee6163d5_nohash_1.wav')
print(sr, type(data), type(data[0]))
print(data)