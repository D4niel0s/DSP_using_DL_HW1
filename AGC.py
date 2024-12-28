from main import *

import collections

def main():
    
    y, sampling_rate = librosa.load('daniel_rec.wav', mono=True)

    print(f'{y.shape=}')
    print(f'{sampling_rate=}')


    # Resample data to 32KHz
    new_rate = 32000

    y = y.astype(np.float32)
    number_of_samples = round(len(y) * float(new_rate) / sampling_rate)
    data = resample(y, number_of_samples)

    
    window_size = round(new_rate * (20e-3))
    hop_size = round(new_rate * (10e-3))

    agc_data = AGC(data, window_size, hop_size, 10000, -40, -10)

    sf.write('daniel_rec_agc.wav', agc_data, new_rate)

    return



def AGC(data, window_size, hop_size, relevant_win_size, floor_threshold ,target_dB):
    relevant_frames = collections.deque(maxlen=relevant_win_size)

    new_data = data.copy()

    for t in range(0,len(data)-window_size, hop_size):
        frame = data[t:t+window_size]

        energy = np.sum(frame**2)
        RMS = np.sqrt(np.mean(frame**2))
        RMS_dB = 20 * np.log10(RMS + 1e-12)  # Add small constant to avoid log(0)
        
        if RMS_dB <= floor_threshold: #Noise
            continue

        if len(relevant_frames) == relevant_win_size:
            relevant_frames.popleft()

        relevant_frames.append(energy)

        relevant_window_RMS = np.sqrt(np.sum(np.array(relevant_frames))/(len(relevant_frames)*window_size))
        relevant_window_RMS_dB_divBy20 = np.log10(relevant_window_RMS + 1e-12)

        gain = 10 ** (target_dB/20 - relevant_window_RMS_dB_divBy20)
        
        new_data[t:t+window_size] = frame * gain

    return new_data
        


if __name__ == '__main__':
    main()