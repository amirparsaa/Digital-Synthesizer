import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import mido

# Convert MIDI note to frequency


def midi_to_freq(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

# Generate sine wave


def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=1.0, phase=0.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return wave, t

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

# Saturation functions


def hard_clipping(x, T):
    return np.clip(x, -T, T)


def soft_clipping(x, T):
    return T * np.tanh(x / T)


def variable_saturation(x, T, a):
    return T * (np.abs(x / T) ** a) * np.sign(x)

# Function to apply saturation effect with bias and drive


def apply_saturation(wave, t, method='hard', T=0.5, a=1.0, bias=0.0, drive=1.0):
    # Apply drive and bias to the time values
    x = drive * t + bias
    # Apply the selected saturation method
    if method == 'hard':
        return hard_clipping(wave, T)
    elif method == 'soft':
        return soft_clipping(wave, T)
    elif method == 'variable':
        return variable_saturation(wave, T, a)
    else:
        return wave

# Function to convert MIDI to audio with saturation


def midi_to_audio_with_saturation(midi_file, output_wav_path, saturation_method='hard', T=0.5, a=1.0, bias=0.0, drive=1.0):
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
                    wave1, t = generate_sine_wave(
                        frequency, note_duration + release_time, sample_rate, amplitude)
                    wave = wave1[:len(envelope)] * envelope[:len(wave1)]

                    # Apply saturation
                    saturated_wave = apply_saturation(
                        wave, t[:len(wave)], method=saturation_method, T=T, a=a, bias=bias, drive=drive)

                    # Plotting the first few notes
                    if note_count < notes_to_plot:
                        plt.figure(figsize=(20, 5))

                        plt.subplot(1, 4, 1)
                        plt.plot(wave1[:len(envelope)])
                        plt.title(f'Sine Wave - Note {note_count + 1}')
                        plt.xlabel('Sample')
                        plt.ylabel('Amplitude')

                        plt.subplot(1, 4, 2)
                        plt.plot(envelope[:len(wave)])
                        plt.title(f'ADSR Envelope - Note {note_count + 1}')
                        plt.xlabel('Sample')
                        plt.ylabel('Amplitude')

                        plt.subplot(1, 4, 3)
                        plt.plot(wave[:len(envelope)])
                        plt.title(
                            f'Final Wave Before Saturation - Note {note_count + 1}')
                        plt.xlabel('Sample')
                        plt.ylabel('Amplitude')

                        plt.subplot(1, 4, 4)
                        plt.plot(saturated_wave[:len(envelope)])
                        plt.title(
                            f'Final Wave After Saturation - Note {note_count + 1}')
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

                    audio_data = np.concatenate((audio_data, saturated_wave))

                current_time += mido.tick2second(msg.time,
                                                 midi_data.ticks_per_beat, tempo)

    audio_data = (audio_data * 32767).astype(np.int16)
    write(output_wav_path, sample_rate, audio_data)
    print(f"Audio saved to {output_wav_path}")


# Usage
midi_file = 'C:\\Users\\hh\\Desktop\\signal\\input.mid'
output_wav_path = 'output_with_saturation.wav'
midi_to_audio_with_saturation(midi_file, output_wav_path,
                              saturation_method='variable', T=0.5, a=2.0, bias=0.1, drive=1.5)
