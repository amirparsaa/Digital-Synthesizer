import numpy as np
import matplotlib.pyplot as plt

 # delaysList of delay times in seconds for each branch
  # weights  List of weights for each branch
def reverbration_effect(audio_data, delays, weights, sample_rate):
 
    delay_buffers = [np.zeros(int(delay * sample_rate)) for delay in delays]
    buffer_indices = [0] * len(delays)
    output_signal = np.zeros_like(audio_data)

    for i, sample in enumerate(audio_data):
        output_sample = sample * (1 - sum(weights))
        for i in range(len(delays)):
            delayed_sample = delay_buffers[i][buffer_indices[i]]
            output_sample += weights[i] * delayed_sample
            delay_buffers[i][buffer_indices[i]] = sample
            buffer_indices[i] = (buffer_indices[i] + 1) % len(delay_buffers[i])

        output_signal[i] = output_sample
    return output_signal

        
