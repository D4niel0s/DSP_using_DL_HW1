import librosa
import numpy as np, matplotlib.pyplot as plt, pyworld as pw, soundfile as sf
from scipy.signal import resample
from noise_cancelling import *

def main():

    y, sampling_rate = librosa.load('recording.wav', mono=True)

    print(f'{y.shape=}')
    print(f'{sampling_rate=}')


    # Resample data to 32KHz
    new_rate = 32000

    y = y.astype(np.float32)
    number_of_samples = round(len(y) * float(new_rate) / sampling_rate)
    clean_data = resample(y, number_of_samples)

    noisy_data = add_noise(clean_data)
    sf.write('orig.wav', noisy_data, new_rate)
    clean_data = cancel_noise(noisy_data, noise_floor=2,buffer_size=round(new_rate * (20e-3)),hop=round(new_rate * (10e-3)))
    
    sf.write('clean.wav', clean_data, new_rate)
    
    # sf.write('clean.wav', clean_data, new_rate)
    # graph_audio_stats(noisy_data, new_rate)

    plt.show()





def downsampling_two_methods(data, sampling_rate):
    print("Save audio files? (Y/N)")
    save = input()

    #Downsample to 16KHz using method 1
    m1_down_sample = data[::2]


    #Downsample to 16KHz using method 2
    m2_down_sample = resample(data, round(len(data)*0.5))

    if (save == 'Y' or save == 'y'):
        sf.write('even_samples_resample.wav', m1_down_sample,round(sampling_rate/2))
        sf.write('scipy_resampled.wav', m2_down_sample, round(sampling_rate/2))
        
    
    graph_audio_stats(m1_down_sample, round(sampling_rate/2))
    plt.suptitle("Even samples resample")

    graph_audio_stats(m2_down_sample, round(sampling_rate/2))
    plt.suptitle("Scipy resample")

    plt.show()



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

def add_noise(data):
    noise, _ = librosa.load('stationary_noise.wav', mono=True)
    noise = noise[:len(data)]
    data = data[:len(noise)]

    return data + noise



if __name__ == '__main__':
    main()