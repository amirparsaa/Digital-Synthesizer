import numpy as np
import matplotlib.pyplot as plt

def phaser_effect(audio_data,num_stages,rate,value,feedback,sample_rate,depth):
   
    lfo_phase = 0
    allpass_buffers = np.zeros((num_stages, 2))
    feedback_sample = 0

    output_signal = np.zeros_like(audio_data)


    for i, sample in enumerate(audio_data):
        # LFO for phase modulation
        lfo = value * np.sin(2 * np.pi * lfo_phase)
        lfo_phase += rate / sample_rate
        if lfo_phase >= 1.0:
            lfo_phase -= 1.0

        # All-pass filter coefficients
        g = np.sin(lfo * np.pi / 2)

        # Apply all-pass filters in series
        for stage in range(num_stages):
            ap_output = -g * sample + allpass_buffers[stage][0] + g *allpass_buffers[stage][1]
            allpass_buffers[stage][0] = sample
            allpass_buffers[stage][1] = ap_output
            sample = ap_output

        # Apply feedback
        output = sample + feedback * feedback_sample
        feedback_sample = output

        
        output_signal[i] = output
    
    return audio_data+depth*output_signal


   