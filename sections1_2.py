from utils import *

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
