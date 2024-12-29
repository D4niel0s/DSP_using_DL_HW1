from utils import *
from sections1_2 import *
from noise_cancelling import *
from AGC import *
from time_stretching import *

def main():

    print("Which section(s) do you want to run? (Close plots to run next section)\n \
          0 - Plot origin audio\n \
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

        save = False
        if int(i) != 0:
            print(f"Save audio for section {i}? (y/n)")
            save = input()
            save = True if save == 'y' or save == 'Y' else False

        runSection(data, new_rate, int(i), save)
        plt.show()



def runSection(data, sampling_rate, sec_num, save):
    if sec_num == 0:
        graph_audio_stats(data, sampling_rate)
        plt.suptitle("Origin audio signal")
    elif sec_num == 1:
        downsampling_two_methods(data, sampling_rate, save, plot=True)
    elif sec_num == 2:
        add_noise(data, sampling_rate, save, plot=True)
    elif sec_num == 3:
        noise_floor = 0.5
        cancel_noise(data, noise_floor, sampling_rate, save, plot=True)
    elif sec_num == 4:
        AGC_wrapper(data, sampling_rate, save, plot=True)
    elif sec_num == 5:
        time_stretching_wrapper(data, sampling_rate, save, plot=True)
    else:
        print(f"Section {sec_num} does not exist :(")



if __name__ == '__main__':
    main()