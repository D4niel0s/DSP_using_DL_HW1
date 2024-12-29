from main import *

import collections

def AGC_wrapper(data, sampling_rate, save=False, plot=False):
    window_size = round(sampling_rate * (20e-3))
    hop_size = round(sampling_rate * (10e-3))

    threshold = -40
    target = -10
    relevant_win_size = 10000

    agc_data = AGC(data, window_size, hop_size, relevant_win_size, threshold, target)

    if (save):
        sf.write('data_AGC.wav', agc_data, sampling_rate)

    if (plot):
        graph_audio_stats(agc_data, sampling_rate)
        plt.suptitle("data after applying AGC")



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