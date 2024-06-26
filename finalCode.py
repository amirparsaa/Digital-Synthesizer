import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import mido
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter, hilbert
from scipy.signal import get_window
from scipy.fftpack import fft, ifft
# Convert MIDI note to frequency


def midi_to_freq(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

# Generate sine wave


def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=1.0, phase=0.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return wave

# Generate ADSR envelope


def generate_adsr_envelope(attack_time, decay_time, sustain_level, sustain_time, release_time, sample_rate=44100, alpha=5):
    def exponential_segment(start_level, end_level, duration, sample_rate, alpha):
        t = np.linspace(0, duration, int(
            sample_rate * duration), endpoint=False)
        if start_level < end_level:  # Exponential rise
            return start_level + (end_level - start_level) * (np.exp(alpha * t / duration) - 1) / (np.exp(alpha) - 1)
        else:  # Exponential decay
            return start_level - (start_level - end_level) * (1 - np.exp(-alpha * t / duration))

    attack_segment = exponential_segment(0, 1, attack_time, sample_rate, alpha)
    decay_segment = exponential_segment(
        1, sustain_level, decay_time, sample_rate, alpha)
    sustain_segment = np.ones(int(sample_rate * sustain_time)) * sustain_level
    release_segment = exponential_segment(
        sustain_level, 0, release_time, sample_rate, alpha)
    envelope = np.concatenate(
        [attack_segment, decay_segment, sustain_segment, release_segment])
    return envelope

# Function to convert MIDI to audio


def midi_to_audio(midi_file, output_wav_path):
    midi_data = mido.MidiFile(midi_file)
    sample_rate = 44100
    audio_data = np.zeros(0)
    current_time = 0.0
    tempo = 500000  # Default tempo in microseconds per beat

    for msg in midi_data.tracks[0]:
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            break

    note_count = 0  # Counter for the number of notes processed
    notes_to_plot = 3  # Number of notes to plot

    for track in midi_data.tracks:
        for msg in track:
            if not msg.is_meta:
                if msg.type == 'note_on' and msg.velocity > 0:
                    frequency = midi_to_freq(msg.note)
                    amplitude = msg.velocity / 127.0

                    attack_time = 0.01
                    decay_time = 0.1
                    sustain_level = 0.7
                    release_time = 0.2
                    note_duration = mido.tick2second(
                        msg.time, midi_data.ticks_per_beat, tempo)

                    sustain_time = max(0, note_duration -
                                       (attack_time + decay_time))

                    if sustain_time < 0:
                        sustain_time = 0

                    envelope = generate_adsr_envelope(
                        attack_time, decay_time, sustain_level, sustain_time, release_time, sample_rate)
                    wave1 = generate_sine_wave(
                        frequency, note_duration + release_time, sample_rate, amplitude)
                    wave = wave1[:len(envelope)] * envelope[:len(wave1)]

                    # Plotting the first 5 notes
                    if note_count < notes_to_plot:
                        plt.figure(figsize=(15, 4))

                        plt.subplot(1, 3, 1)
                        plt.plot(wave1[:len(envelope)])
                        plt.title(f'Sine Wave - Note {note_count + 1}')
                        plt.xlabel('Sample')
                        plt.ylabel('Amplitude')

                        plt.subplot(1, 3, 2)
                        plt.plot(envelope[:len(wave)])
                        plt.title(f'ADSR Envelope - Note {note_count + 1}')
                        plt.xlabel('Sample')
                        plt.ylabel('Amplitude')

                        plt.subplot(1, 3, 3)
                        plt.plot(wave[:len(envelope)] * envelope[:len(wave)])
                        plt.title(f'Final Wave - Note {note_count + 1}')
                        plt.xlabel('Sample')
                        plt.ylabel('Amplitude')

                        plt.tight_layout()
                        plt.show()

                    note_count += 1

                    padding_length = int(
                        current_time * sample_rate) - len(audio_data)
                    if padding_length > 0:
                        audio_data = np.concatenate(
                            (audio_data, np.zeros(padding_length)))

                    audio_data = np.concatenate((audio_data, wave))

                current_time += mido.tick2second(msg.time,
                                                 midi_data.ticks_per_beat, tempo)

    plt.figure(figsize=(6, 10))
    t = np.linspace(0, 1, sample_rate)
    plt.plot(t, audio_data[0:sample_rate], label='input signal')
    # audio_data = apply_saturation(audio_data, saturation_type='soft', threshold=0.5, drive=1.0,
    #                               bias=0.0, wet=1.0)
    # plt.plot(t, audio_data[0:sample_rate], label='saturated signal')
    # plt.legend()
    # plt.xlabel("Time [s]")
    # plt.ylabel("Amplitude")
    # plt.title("saturation_effect")

    # audio_data = apply_compression(
    #     audio_data, threshold=0.5, ratio=4.0, attack=0.01, release=0.1, sample_rate=44100, wet=1.0)
    # plt.plot(t, audio_data[0:sample_rate], label='compressed signal')
    # plt.legend()
    # plt.xlabel("Time [s]")
    # plt.ylabel("Amplitude")
    # plt.title("compression_effect")

    # audio_data = phaser_effect(
    # audio_data, 4, 5, 0.7, 0.6, sample_rate, 0.5, 'phaser_effect.wav')

    # audio_data=voice_encoder(audio_data,'output_vocoder.wav')

    # audio_data=delay_effect(audio_data,int(5* sample_rate), 0.98,50,1,sample_rate,'delay_effect.wav')

    # audio_data=flinger_effect(audio_data,0.005,0.002,1,0.5,0.5,sample_rate,'flinger_effect.wav')

    # audio_data=chorus_effect(audio_data,4,[0.1, 0.5, 0.8, 1.1], 0.002,0.3,0.5,sample_rate,'chorus_effect.wav')

    # audio_data = reverbration_effect(audio_data, [0.1, 0.5, 0.8, 1.1], [
    #                                 0.2, 0.15, 0.1, 0.05], sample_rate, 'reverbration_effect.wav')

    audio_data = apply_pitch_shift(audio_data, sample_rate, n_steps=4, wet=1.0)
    plt.plot(t, audio_data[0:sample_rate], label='pitch shifted signal')
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("pitch_shift_effect")

    # Normalize audio to prevent clipping
    audio_data = audio_data / np.max(np.abs(audio_data))

    plt.tight_layout()
    plt.show()
    # Convert to 16-bit PCM format
    audio_data = (audio_data * 32767).astype(np.int16)
    write(output_wav_path, sample_rate, audio_data)

    print(f"Audio saved to {output_wav_path}")


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


def voice_encoder(carrier_signal, output_wav_path):
    file = wav.read('Sports.wav')
    print(file[0])
    # Parameters
    fs = file[0]  # Sampling frequency
    num_bands = 40  # Number of frequency bands
    lowcut = 1.0  # Low cut for frequency band division
    highcut = 20000.0  # High cut for frequency band division
    order = 6
    # Generate frequency bands
    band_edges = np.logspace(
        np.log10(lowcut), np.log10(highcut), num_bands + 1)

    # Example carrier and modulator signals
    # Replace with actual carrier signal
    # Replace with actual modulator signal, different length
    modulator_signal = file[1]

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
        modulator_band = np.int16(bandpass_filter(
            modulator_signal_padded, low, high, fs, order=order))

        # Filter the carrier signal
        carrier_band = np.int16(bandpass_filter(
            carrier_signal_padded, low, high, fs, order=order))

        # Modulate the carrier with the envelope of the modulator
        envelope = np.abs(modulator_band)
        output_signal += envelope * carrier_band

    write(output_wav_path, fs, output_signal)
    print(f"Audio saved to {output_wav_path}")
    t = np.linspace(0, 1, fs)

    plt.figure(figsize=(10, 6))
    plt.plot(t, carrier_signal[0:fs], label="Input Signal")
    plt.plot(t, output_signal[0:fs], label="Output Signal (vocoder)")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("vocoder")
    plt.grid()
    plt.show()


def delay_effect(audio_data, max_delay_samples, feedback, lfo_rate, lfo_depth, sample_rate, output_wav_path):

    output_signal = np.zeros_like(audio_data)

    for i, sample in enumerate(audio_data):
        buffer = np.zeros(max_delay_samples)
        buffer_idx = 0
        lfo_phase = 0

        # Apply LFO modulation to delay time
        lfo_value = lfo_depth * np.sin(2 * np.pi * lfo_phase)
        # Divide by 2 to keep within bounds
        modulated_delay = int(max_delay_samples * (1 + lfo_value) / 2)
        lfo_phase += lfo_rate / sample_rate
        if lfo_phase >= 1.0:
            lfo_phase -= 1.0

        # Calculate delay buffer index
        delay_idx = (buffer_idx - modulated_delay +
                     max_delay_samples) % max_delay_samples

        # Get delayed sample
        delayed_sample = buffer[delay_idx]

        # Calculate output with feedback
        output = sample + feedback * delayed_sample

        # Store current sample in buffer
        buffer[buffer_idx] = output

        # Increment buffer index
        buffer_idx = (buffer_idx + 1) % max_delay_samples

        output_signal[i] = output
    write(output_wav_path, sample_rate, output_signal)
    print(f"Audio saved to {output_wav_path}")
    t = np.linspace(0, 1, sample_rate)

    plt.figure(figsize=(10, 6))
    plt.plot(t, audio_data[0:sample_rate], label="Input Signal")
    plt.plot(t, output_signal[0:sample_rate],
             label="Output Signal (delay_effect)")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("delay_effect")
    plt.grid()
    plt.show()

    return output_signal


# Saturation functions


def hard_clipping(x, threshold):
    return np.clip(x, -threshold, threshold)


def soft_clipping(x, threshold):
    return np.tanh(x / threshold) * threshold


def variable_clipping(x, threshold):
    return x / (1 + np.abs(x / threshold))


def apply_saturation(audio, saturation_type, threshold, drive, bias, wet):
    if saturation_type == 'hard':
        saturator = hard_clipping
    elif saturation_type == 'soft':
        saturator = soft_clipping
    elif saturation_type == 'variable':
        saturator = variable_clipping
    else:
        raise ValueError("Invalid saturation type")

    processed_audio = saturator(drive * audio + bias, threshold)

    return wet * processed_audio + (1 - wet) * audio


def apply_compression(audio, threshold=0.5, ratio=4.0, attack=0.01, release=0.1, sample_rate=44100, wet=1.0):
    compressed_audio = np.copy(audio)
    envelope = np.zeros_like(audio)
    gain = np.ones_like(audio)
    attack_coeff = np.exp(-1.0 / (attack * sample_rate))
    release_coeff = np.exp(-1.0 / (release * sample_rate))

    for i in range(1, len(audio)):
        envelope[i] = max(np.abs(audio[i]), envelope[i-1] *
                          (release_coeff if np.abs(audio[i]) < envelope[i-1] else attack_coeff))
        if envelope[i] > threshold:
            gain[i] = (1.0 + (envelope[i] - threshold) / (threshold * ratio))
        compressed_audio[i] = audio[i] / gain[i]

    return wet * compressed_audio + (1 - wet) * audio


def phaser_effect(audio_data, num_stages, rate, value, feedback, sample_rate, depth, output_wav_path):

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
            ap_output = -g * sample + \
                allpass_buffers[stage][0] + g * allpass_buffers[stage][1]
            allpass_buffers[stage][0] = sample
            allpass_buffers[stage][1] = ap_output
            sample = ap_output

        # Apply feedback
        output = sample + feedback * feedback_sample
        feedback_sample = output

        output_signal[i] = output
    output_signal = audio_data+depth*output_signal
    write(output_wav_path, sample_rate, output_signal)
    print(f"Audio saved to {output_wav_path}")
    t = np.linspace(0, 1, sample_rate)

    plt.figure(figsize=(10, 6))
    plt.plot(t, audio_data[0:sample_rate], label="Input Signal")
    plt.plot(t, output_signal[0:sample_rate],
             label="Output Signal (phaser_effect)")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("phaser_effect")
    plt.grid()
    plt.show()
    return audio_data+depth*output_signal


def flinger_effect(audio_data, delay_time, depth, rate, feedback, mix, sample_rate, output_wav_path):

    lfo_phase = 0
    delay_buffer = np.zeros(sample_rate)  # Initialize delay buffer
    buffer_idx = 0
    output_signal = np.zeros_like(audio_data)

    for i, sample in enumerate(audio_data):
        # Calculate current delay time with LFO modulation
        lfo = np.sin(2 * np.pi * lfo_phase) * depth + delay_time
        lfo_phase += rate / sample_rate
        if lfo_phase >= 1.0:
            lfo_phase -= 1.0

        delay_samples = int(lfo * sample_rate)
        read_idx = (buffer_idx - delay_samples + sample_rate) % sample_rate

        delayed_sample = delay_buffer[int(read_idx)]
        output_sample = (1 - mix) * sample + mix * delayed_sample
        delay_buffer[buffer_idx] = sample + feedback * delayed_sample

        buffer_idx = (buffer_idx + 1) % sample_rate
        output_signal[i] = output_sample
    t = np.linspace(0, 1, sample_rate)

    plt.figure(figsize=(10, 6))
    plt.plot(t, audio_data[0:sample_rate], label="Input Signal")
    plt.plot(t, output_signal[0:sample_rate],
             label="Output Signal (flinger_effect)")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("flinger_effect")
    plt.grid()
    plt.show()
    write(output_wav_path, sample_rate, output_signal)
    print(f"Audio saved to {output_wav_path}")

    return output_signal


# delay_times is a list of delays
def chorus_effect(audio_data, num_voices, delay_times, depth, rate, mix, sample_rate, output_wav_path):

    lfo_phases = np.zeros(num_voices)
    delay_buffers = [np.zeros(sample_rate) for _ in range(num_voices)]
    buffer_indices = np.zeros(num_voices, dtype=int)
    output_signal = np.zeros_like(audio_data)

    for j, sample in enumerate(audio_data):
        output_sample = sample * (1 - mix)
        for i in range(num_voices):
            lfo = np.sin(2 * np.pi * lfo_phases[i]) * depth + delay_times[i]
            lfo_phases[i] += rate / sample_rate
            if lfo_phases[i] >= 1.0:
                lfo_phases[i] -= 1.0

            delay_samples = int(lfo * sample_rate)
            read_idx = (buffer_indices[i] -
                        delay_samples + sample_rate) % sample_rate

            delayed_sample = delay_buffers[i][int(read_idx)]
            output_sample += delayed_sample * mix / num_voices
            delay_buffers[i][buffer_indices[i]] = sample

            buffer_indices[i] = (buffer_indices[i] + 1) % sample_rate
        output_signal[j] = output_sample
    write(output_wav_path, sample_rate, output_signal)
    print(f"Audio saved to {output_wav_path}")
    t = np.linspace(0, 1, sample_rate)
    plt.figure(figsize=(10, 6))
    plt.plot(t, audio_data[0:sample_rate], label="Input Signal")
    plt.plot(t, output_signal[0:sample_rate],
             label="Output Signal (chorus_effect)")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("chorus_effect")
    plt.grid()
    plt.show()
    return output_signal


# delaysList of delay times in seconds for each branch
# weights  List of weights for each branch
def reverbration_effect(audio_data, delays, weights, sample_rate, output_wav_path):

    delay_buffers = [np.zeros(int(delay * sample_rate)) for delay in delays]
    buffer_indices = [0] * len(delays)
    output_signal = np.zeros_like(audio_data)

    for j, sample in enumerate(audio_data):
        output_sample = sample * (1 - sum(weights))
        for i in range(len(delays)):
            delayed_sample = delay_buffers[i][buffer_indices[i]]
            output_sample += weights[i] * delayed_sample
            delay_buffers[i][buffer_indices[i]] = sample
            buffer_indices[i] = (buffer_indices[i] + 1) % len(delay_buffers[i])

        output_signal[j] = output_sample

    write(output_wav_path, sample_rate, output_signal)
    print(f"Audio saved to {output_wav_path}")
    t = np.linspace(0, 1, sample_rate)
    plt.figure(figsize=(10, 6))
    plt.plot(t, audio_data[0:sample_rate], label="Input Signal")
    plt.plot(t, output_signal[0:sample_rate],
             label="Output Signal (reverbration_effect)")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("reverbration_effect")
    plt.grid()
    plt.show()
    return output_signal

# Define the pitch shifter effect


def stft(x, fft_size=1024, hop_size=512, window='hann'):
    """ Short-time Fourier transform (STFT) """
    w = get_window(window, fft_size)
    return np.array([fft(w*x[i:i+fft_size]) for i in range(0, len(x)-fft_size, hop_size)])


def istft(X, fft_size=1024, hop_size=512, window='hann'):
    """ Inverse short-time Fourier transform (ISTFT) """
    w = get_window(window, fft_size)
    x = np.zeros((X.shape[0]-1) * hop_size + fft_size)
    for n, i in enumerate(range(0, len(x)-fft_size, hop_size)):
        x[i:i+fft_size] += np.real(ifft(X[n])) * w
    return x


def phase_vocoder(X, time_stretch):
    """ Phase vocoder for time-stretching """
    num_bins, num_frames = X.shape
    time_steps = np.arange(num_frames) * time_stretch
    phase_advances = np.linspace(0, np.pi * time_stretch, num_bins)
    X_stretch = np.zeros((num_bins, len(time_steps)), dtype=complex)
    for i, t in enumerate(time_steps):
        frame1 = int(np.floor(t))
        frame2 = min(frame1 + 1, num_frames - 1)
        interp_alpha = t - frame1
        X_stretch[:, i] = (1 - interp_alpha) * X[:, frame1] + \
            interp_alpha * X[:, frame2]
        X_stretch[:, i] *= np.exp(1j * phase_advances * i)
    return X_stretch


def pitch_shift(audio, sample_rate, n_steps, fft_size=1024, hop_size=512):
    """ Pitch shift using the phase vocoder """
    # Calculate the time-stretch factor
    time_stretch = 2.0 ** (-n_steps / 12.0)

    # Perform STFT
    X = stft(audio, fft_size, hop_size)

    # Time-stretch the STFT
    X_stretch = phase_vocoder(X, time_stretch)

    # Perform inverse STFT
    audio_stretch = istft(X_stretch, fft_size, hop_size)

    # Resample to adjust the pitch
    audio_shift = np.interp(np.arange(0, len(audio_stretch), time_stretch),
                            np.arange(len(audio_stretch)), audio_stretch)
    return audio_shift


def apply_pitch_shift(audio, sample_rate, n_steps=0, wet=1.0):
    """ Apply pitch shift effect """
    pitch_shifted_audio = pitch_shift(audio, sample_rate, n_steps)
    # Ensure pitch_shifted_audio and audio have the same length
    if len(pitch_shifted_audio) > len(audio):
        pitch_shifted_audio = pitch_shifted_audio[:len(audio)]
    else:
        pitch_shifted_audio = np.pad(pitch_shifted_audio, (0, len(
            audio) - len(pitch_shifted_audio)), 'constant')

    return wet * pitch_shifted_audio + (1 - wet) * audio


# Usage
midi_file = 'C:\\Users\\hh\\Desktop\\signal\\input.mid'
output_wav_path = 'output.wav'
midi_to_audio(midi_file, output_wav_path)
