import numpy as np
import librosa

def cancel_noise(data, noise_floor, buffer_size, hop):
    D = librosa.stft(data, win_length=buffer_size, hop_length=hop)
    noise_indexes = np.array([0])
    for i in range(0, len(data)-buffer_size, hop):
        energy = np.sum(data[i:i+buffer_size]**2)
        if(energy < noise_floor):
            noise_indexes = np.append(noise_indexes, round(i/buffer_size))

    avg_noise = np.average(np.abs(D[:, noise_indexes]))
    return librosa.istft(D - avg_noise)