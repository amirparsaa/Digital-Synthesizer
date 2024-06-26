import numpy as np
from scipy.signal import butter, lfilter
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt
# Function to create a band-pass filter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply band-pass filter to a signal
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Zero-padding function
def zero_pad(signal, target_length):
    pad_length = max(0, target_length - len(signal))
    return np.pad(signal, (0, pad_length), mode='constant')

def voice_encoder(carrier_signal,output_wav_path):
    file=wav.read('Sports.wav')
    print(file[0])
    # Parameters
    fs = file[0]  # Sampling frequency
    num_bands = 40  # Number of frequency bands
    lowcut = 1.0  # Low cut for frequency band division
    highcut = 20000.0  # High cut for frequency band division
    order = 6
    # Generate frequency bands
    band_edges = np.logspace(np.log10(lowcut), np.log10(highcut), num_bands + 1)

    # Example carrier and modulator signals
    # Replace with actual carrier signal
    modulator_signal = file[1]  # Replace with actual modulator signal, different length

    # Determine the target length for zero-padding (maximum length of the two signals)
    target_length = max(len(carrier_signal), len(modulator_signal))




    # Apply zero-padding to both signals
    carrier_signal_padded = zero_pad(carrier_signal, target_length)
    modulator_signal_padded = zero_pad(modulator_signal, target_length)

    # Process each band
    output_signal = np.zeros_like(carrier_signal_padded)
    for i in range(num_bands):
        low = band_edges[i]
        high = band_edges[i + 1]
        
        # Filter the modulator signal
        modulator_band = np.int16(bandpass_filter(modulator_signal_padded, low, high, fs, order=order))
        
        # Filter the carrier signal
        carrier_band = np.int16(bandpass_filter(carrier_signal_padded, low, high, fs, order=order))
        
        # Modulate the carrier with the envelope of the modulator
        envelope = np.abs(modulator_band)
        output_signal += envelope * carrier_band

    t = np.linspace(0, 1, fs)
    input_signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
 

    # Plot the input and output signals
    plt.figure(figsize=(10, 6))
    plt.plot(t, input_signal, label="Input Signal")
    plt.plot(t, output_signal, label="Output Signal (Chorus)")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Chorus Effect")
    plt.grid()
    plt.show()
    
