import librosa
import numpy as np, matplotlib.pyplot as plt, pyworld as pw, soundfile as sf
from scipy.signal import resample

#Our modules
from noise_cancelling import *
from AGC import *
from time_stretching import *

def main():

    print("Which section(s) do you want to run? (Close plots to run next section)\n \
          1 - Downsampling two methods\n \
          2 - Adding noise\n \
          3 - Spectral subtraction\n \
          4 - AGC\n \
          5 - Time stretching\n \
          (Enter a list e.g 1,2,3 or a single number)")
    
    sections = tuple(input().split(','))

    for i in sections:
        y, sampling_rate = librosa.load('recording.wav', mono=True)

        # Resample data to 32KHz
        new_rate = 32000
        y = y.astype(np.float32)
        number_of_samples = round(len(y) * float(new_rate) / sampling_rate)
        data = resample(y, number_of_samples)

        print(f"Save audio for section {i}? (y/n)")
        save = input()
        save = True if save == 'y' or save == 'Y' else False

        runSection(data, new_rate, int(i), save)
        plt.show()



def runSection(data, sampling_rate, sec_num, save):
    if sec_num == 1:
        downsampling_two_methods(data, sampling_rate, save, plot=True)
    elif sec_num == 2:
        add_noise(data, sampling_rate, save, plot=True)
    elif sec_num == 3:
        print("SHAHAR IMPLEMENT THIS WRAPPER נגמר לי הכוח")
    elif sec_num == 4:
        AGC_wrapper(data, sampling_rate, save, plot=True)
    elif sec_num == 5:
        time_stretching_wrapper(data, sampling_rate, save, plot=True)
    else:
        print(f"Section {sec_num} does not exist :(")


def downsampling_two_methods(data, sampling_rate, save=False, plot=False):
    #Downsample to 16KHz using method 1
    m1_down_sample = data[::2]


    #Downsample to 16KHz using method 2
    m2_down_sample = resample(data, round(len(data)*0.5))

    if (save):
        sf.write('even_samples_resample.wav', m1_down_sample,round(sampling_rate/2))
        sf.write('scipy_resampled.wav', m2_down_sample, round(sampling_rate/2))
        
    if (plot):
        graph_audio_stats(m1_down_sample, round(sampling_rate/2))
        plt.suptitle("Even samples resample")

        graph_audio_stats(m2_down_sample, round(sampling_rate/2))
        plt.suptitle("Scipy resample")



def add_noise(data, sampling_rate, save=False, plot=False):
    noise, sr = librosa.load('stationary_noise.wav', mono=True)

    new_rate = 16000
    number_of_samples = round(len(noise) * float(new_rate) / sr)
    noise = resample(noise, number_of_samples)
    
    #Truncating to the max-length
    noise = noise[:len(data)]
    data = data[:len(noise)]

    if (plot):
        graph_audio_stats(noise, new_rate)
        plt.suptitle("Noise")

        graph_audio_stats(data, sampling_rate)
        plt.suptitle("Clean data")

        graph_audio_stats(data+noise, sampling_rate)
        plt.suptitle("Noisy data")

    if (save):
        sf.write("noisy_data.wav", data+noise, sampling_rate)

    return data + noise



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


if __name__ == '__main__':
    main()