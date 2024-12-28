from main import *

def cancel_noise(data, noise_floor, sampling_rate= 32000):
    window_size = round(sampling_rate * (20e-3))
    hop_size = round(sampling_rate * (10e-3))
    
    # Use Hann window for better frequency resolution
    D = librosa.stft(data, win_length=window_size, hop_length=hop_size, window='tukey')
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # Find noise frames
    noise_indexes = np.unique([round(i/window_size) for i in range(0,len(data)-window_size, hop_size) if np.sum(data[i:i+window_size]**2) < noise_floor])

    if len(noise_indexes) == 0:
        print("Warning: No noise frames detected. Adjusting noise floor might be necessary.")
        return data
    
    noise_profile = np.average(magnitude[:, noise_indexes], axis=1)
    
    new_mag = (magnitude.T - noise_profile).T
    new_mag = np.maximum(new_mag, 0)
    
    # Reconstruct signal
    clean = librosa.istft(new_mag * np.exp(1j*phase), win_length=window_size, hop_length=hop_size, window='tukey')
    return clean
