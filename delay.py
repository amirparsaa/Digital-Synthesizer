import numpy as np
import matplotlib.pyplot as plt


def delay_effect(audio_data,max_delay_samples, feedback,lfo_rate,lfo_depth,sample_rate):

    output_signal = np.zeros_like(audio_data)
    buffer_idx = 0
    lfo_phase = 0
    for i, sample in enumerate(audio_data):
        buffer = np.zeros(max_delay_samples)


        # Apply LFO modulation to delay time
        lfo_value = lfo_depth * np.sin(2 * np.pi * lfo_phase)
        modulated_delay = int(max_delay_samples * (1 + lfo_value) / 2)  # Divide by 2 to keep within bounds
        lfo_phase += lfo_rate / sample_rate
        if lfo_phase >= 1.0:
            lfo_phase -= 1.0
        
        # Calculate delay buffer index
        delay_idx = (buffer_idx - modulated_delay +max_delay_samples) % max_delay_samples
        
        # Get delayed sample
        delayed_sample = buffer[delay_idx]
        
        # Calculate output with feedback
        output = sample + feedback * delayed_sample
        
        # Store current sample in buffer
        buffer[buffer_idx] = output
        
        # Increment buffer index
        buffer_idx = (buffer_idx + 1) % max_delay_samples
        
        output_signal[i] =output
    return output_signal
