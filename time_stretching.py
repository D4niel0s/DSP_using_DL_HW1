from main import *


def time_stretching_wrapper(data, sampling_rate, save=False, plot=False):
    factor = 1.5
    time_stretched = increase_speed(data, factor)

    if (save):
        sf.write('time_stretched.wav', time_stretched, sampling_rate)
    
    if (plot):
        graph_audio_stats(time_stretched, sampling_rate)
        plt.suptitle(f"Time stretched data by a factor of {factor}")

def increase_speed(audio, factor):

    #Maping from output to input
    mapping = lambda t: t*factor

    D = librosa.stft(audio)
    magnitude = np.abs(D)
    phase = np.angle(D)
    

    time_stretched = np.zeros((D.shape[0], int(len(D[0])/factor)), dtype=np.complex64)
    
    previous_phase = phase[:,0]

    for t in range(1,len(time_stretched[0])):
        left = int(np.floor(mapping(t)))
        right = min(left+ 1, len(D[0]) - 1)
        
        #Magnitude
        lweight = right - mapping(t)
        rweight = 1-lweight

        new_mag = lweight*magnitude[:,left] + rweight*magnitude[:,right]

        #Phase
        phase_diff = phase[:,right] - phase[:,left]

        new_phase = previous_phase + phase_diff
        previous_phase = new_phase

        time_stretched[:,t] = new_mag * np.exp(1j*new_phase)


    time_stretched = librosa.istft(time_stretched)
    
    return time_stretched