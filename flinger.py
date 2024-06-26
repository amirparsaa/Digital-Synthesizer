import numpy as np
import matplotlib.pyplot as plt

def flinger_effect(audio_data,delay_time,depth,rate,feedback,mix,sample_rate):

    lfo_phase = 0
    delay_buffer = np.zeros(sample_rate)  # Initialize delay buffer
    buffer_idx = 0
    output_signal = np.zeros_like(audio_data)

    for i, sample in enumerate(audio_data):
        # Calculate current delay time with LFO modulation
        lfo = np.sin(2 * np.pi *lfo_phase) * depth + delay_time
        lfo_phase += rate / sample_rate
        if lfo_phase >= 1.0:
            lfo_phase -= 1.0

        delay_samples = int(lfo *sample_rate)
        read_idx = (buffer_idx - delay_samples + sample_rate) % sample_rate

        delayed_sample = delay_buffer[int(read_idx)]
        output_sample = (1 - mix) * sample +mix * delayed_sample
        delay_buffer[buffer_idx] = sample + feedback * delayed_sample

        buffer_idx = (buffer_idx + 1) % sample_rate
        output_signal[i] =output_sample

    return output_signal



    
    
        


