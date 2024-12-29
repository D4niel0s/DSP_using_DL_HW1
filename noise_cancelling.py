from main import *

def cancel_noise(data, noise_floor=1, sampling_rate=32000,save=False, plot=False):
    window_size = round(sampling_rate * (20e-3))
    hop_size = round(sampling_rate * (10e-3))
    noisy_data = add_noise(data, sampling_rate)
    # Use tukey window for better frequency resolution
    D = librosa.stft(noisy_data, win_length=window_size, hop_length=hop_size, window='tukey')
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # Find noise frames
    noise_indexes = np.unique([round(i/window_size) for i in range(0,len(noisy_data)-window_size, hop_size) if np.sum(noisy_data[i:i+window_size]**2) < noise_floor])

    if len(noise_indexes) == 0:
        print("Warning: No noise frames detected. Adjusting noise floor might be necessary.")
        return data
    
    noise = np.average(magnitude[:, noise_indexes], axis=1)
    
    new_mag = (magnitude.T - noise).T
    new_mag = np.maximum(new_mag, 0)
    
    # Reconstruct signal
    clean = librosa.istft(new_mag * np.exp(1j*phase), win_length=window_size, hop_length=hop_size, window='tukey')
    
    if(plot):
        graph_audio_stats(data, sampling_rate)
        plt.suptitle("Original Data")
        
        graph_audio_stats(noisy_data, sampling_rate)
        plt.suptitle("Noisy Data")

        graph_audio_stats(clean, sampling_rate)
        plt.suptitle("Clean Data")
    
    if(save):
        sf.write("clean.wav", clean, sampling_rate)
    
    return clean
