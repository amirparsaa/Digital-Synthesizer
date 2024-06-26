import numpy as np
import matplotlib.pyplot as plt
#delay_times is a list of delays
def chorus_effect(audio_data,num_voices,delay_times,depth,rate,mix,sample_rate):

    lfo_phases = np.zeros(num_voices)
    delay_buffers = [np.zeros(sample_rate) for _ in range(num_voices)]
    buffer_indices = np.zeros(num_voices, dtype=int)
    output_signal = np.zeros_like(audio_data)
   
    for i, sample in enumerate(audio_data):
        output_sample = sample * (1 -mix)
        for i in range(num_voices):
            lfo = np.sin(2 * np.pi *lfo_phases[i]) * depth +delay_times[i]
            lfo_phases[i] += rate /sample_rate
            if lfo_phases[i] >= 1.0:
                lfo_phases[i] -= 1.0

            delay_samples = int(lfo *sample_rate)
            read_idx = (buffer_indices[i] - delay_samples +sample_rate) % sample_rate

            delayed_sample =delay_buffers[i][int(read_idx)]
            output_sample += delayed_sample * mix / num_voices
            delay_buffers[i][buffer_indices[i]] = sample

            buffer_indices[i] = (buffer_indices[i] + 1) % sample_rate
        output_signal[i] = output_sample
   
    return output_signal   
        

