import librosa
from scipy.signal import resample
import numpy as np
import matplotlib.pyplot as plt
import pyworld as pw
import soundfile as sf

def display_audio_stats(data, sampling_rate):
    fig, ax = plt.subplots(nrows=5, sharex=True)
    librosa.display.waveshow(data, sr=sampling_rate, ax=ax[0])
    ax[0].set(title='Envelope view, mono')
    ax[0].label_outer()

    #spektogram
    D = librosa.stft(data)
    magnitute = np.abs(D)
    s_db = librosa.amplitude_to_db(magnitute, ref=np.max)
    
    librosa.display.specshow(s_db, sr=sampling_rate, x_axis='time', y_axis='linear', ax=ax[1], cmap='viridis')
    
    #pitch contour
    data = data.astype(np.float64)
    
    _f0, t = pw.dio(data, sampling_rate)
    f0 = pw.stonemask(data, _f0, t, sampling_rate)
    ax[1].plot(t, f0, color='r')
    data = data.astype(np.float32)
    
    
    #mel spektogram
    
    S = librosa.feature.melspectrogram(y=data, sr= sampling_rate, n_mels=128, fmax=sampling_rate/2)
    S_db = librosa.power_to_db(S,ref=np.max)
    librosa.display.specshow(S_db, sr=sampling_rate, x_axis='time', y_axis='mel', ax=ax[2],fmax=sampling_rate/2
                             ,cmap='viridis')
    
    #iv
    energy = np.sum(data**2)
    
    rms = np.sqrt(energy/len(data))

    plt.suptitle(f'energy={energy : .3f}, rms={rms : .3f}')
    plt.show()

y, sampling_rate = librosa.load('recording.wav', mono=True)

print(f'{y.shape=}')
print(f'{sampling_rate=}')


# Resample data
new_rate = 32000

y = y.astype(np.float32)
number_of_samples = round(len(y) * float(new_rate) / sampling_rate)
data = resample(y, number_of_samples)

#Method 1
m1_down_sample = data[::2]

sf.write('manual.wav', m1_down_sample,round(new_rate/2))

#Method 2
m2_down_sample = resample(data, round(len(data)*0.5))
sf.write('libro.wav', m2_down_sample, round(new_rate/2))

    
display_audio_stats(m2_down_sample, new_rate/2)
    