import librosa
import numpy as np, matplotlib.pyplot as plt, pyworld as pw, soundfile as sf
from scipy.signal import resample


def graph_audio_stats(data, sampling_rate):
    window_size = round(sampling_rate * (20e-3))
    hop_size = round(sampling_rate * (10e-3))

    fig, ax = plt.subplots(nrows=4, sharex=True)
    duration = len(data) / sampling_rate

    librosa.display.waveshow(data, sr=sampling_rate, ax=ax[0])
    ax[0].set(title='Waveform', ylabel='Pressure')
    ax[0].label_outer() 

    #spectogram
    D = librosa.stft(data, win_length=window_size, hop_length=hop_size)
    magnitute = np.abs(D)
    s_db = librosa.amplitude_to_db(magnitute, ref=np.max)
    librosa.display.specshow(s_db, sr=sampling_rate,x_axis='time', y_axis='linear',win_length=window_size,hop_length=hop_size, ax=ax[1], cmap='viridis')
    

    #pitch contour
    data = data.astype(np.float64)

    _f0, t = pw.dio(data, sampling_rate)
    f0 = (pw.stonemask(data, _f0, t, sampling_rate))

    data = data.astype(np.float32)

    ax[1].plot(t, f0, color='r', label='Pitch contour')
    ax[1].set(title='Spectrogram and pitch contour')
    ax[1].legend()
    ax[1].label_outer()

    

    #mel spectogram
    S = librosa.feature.melspectrogram(y=data, sr= sampling_rate, n_mels=128, fmax=sampling_rate/2, win_length=window_size, hop_length=hop_size)
    S_db = librosa.power_to_db(S,ref=np.max)
    librosa.display.specshow(S_db, sr=sampling_rate, x_axis='time', y_axis='mel', fmax=sampling_rate/2,win_length=window_size, hop_length=hop_size, ax=ax[2], cmap='viridis')
    ax[2].set(title='Mel spectrogram')
    ax[2].label_outer()

    #Energy and RMS over time
    energy_curve = np.array([np.sum(data[i:i+window_size]**2) for i in range(0, len(data)-window_size, hop_size)])
    RMS_curve = np.array([np.sqrt(np.mean(data[i:i+window_size]**2)) for i in range(0, len(data)-window_size, hop_size)])


    time_points = np.arange(len(energy_curve)) * (hop_size / sampling_rate)
        
    ax[3].plot(time_points, energy_curve, label='Energy', color='b')
    ax[3].plot(time_points, RMS_curve, label='RMS', color='g')
    ax[3].set(title='Energy and RMS over time', xlabel='time')
    ax[3].label_outer()
    

    #For some reason the x axis is not being set correctly, this is a workaround
    for axes in ax:
        axes.set_xlim(0, duration)
    

    plt.subplots_adjust(hspace=1)
