from main import *




def main():
    y, sr = librosa.load('daniel-rec.wav', mono=True)
    
    graph_audio_stats(y, sr)

    increase_speed(y, sr, 2)



def increase_speed(audio, sampling_rate, factor):
    print("Save audio files? (Y/N)")
    save = input()

    #Maping from output
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

    
    if (save == 'Y' or save == 'y'):
        sf.write('time_stretched.wav', time_stretched, sampling_rate)
    
    graph_audio_stats(time_stretched, sampling_rate)
    plt.suptitle("Time stretched audio")
    
    plt.show()







if __name__ == '__main__':
    main()