import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# تابع تولید سیگنال سینوسی


def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=1.0, phase=0.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return wave

# تابع تولید موج دندانه اره‌ای


def generate_sawtooth_wave(frequency, duration, sample_rate=44100, amplitude=1.0, phase=0.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * (2 * (t * frequency - np.floor(0.5 + t * frequency)))
    return wave

# تابع تولید موج مثلثی


def generate_triangle_wave(frequency, duration, sample_rate=44100, amplitude=1.0, phase=0.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * \
        (2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1)
    return wave

# تابع تولید موج مربعی


def generate_square_wave(frequency, duration, sample_rate=44100, amplitude=1.0, phase=0.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sign(np.sin(2 * np.pi * frequency * t + phase))
    return wave

# تابع طراحی فیلتر پایین گذر با استفاده از butterworth


def design_lowpass_filter(cutoff, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# تابع اعمال فیلتر


def apply_filter(data, b, a):
    y = lfilter(b, a, data)
    return y

# تابع تولید سیگنال LFO


def generate_lfo(frequency, duration, sample_rate=44100, amplitude=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    lfo = amplitude * np.sin(2 * np.pi * frequency * t)
    return lfo

# تابع اصلی برای تولید شکل موج، اعمال VCF و LFO


def generate_waveform_with_vcf_and_lfo(wave_type, frequency, duration, sample_rate=44100, amplitude=1.0, phase=0.0, cutoff_freq=1000, lfo_freq=5, lfo_amplitude=0.5):
    if wave_type == 'sine':
        wave = generate_sine_wave(
            frequency, duration, sample_rate, amplitude, phase)
    elif wave_type == 'sawtooth':
        wave = generate_sawtooth_wave(
            frequency, duration, sample_rate, amplitude, phase)
    elif wave_type == 'triangle':
        wave = generate_triangle_wave(
            frequency, duration, sample_rate, amplitude, phase)
    elif wave_type == 'square':
        wave = generate_square_wave(
            frequency, duration, sample_rate, amplitude, phase)
    else:
        raise ValueError(
            "Invalid wave type. Choose from 'sine', 'sawtooth', 'triangle', 'square'.")

    # طراحی و اعمال VCF
    b, a = design_lowpass_filter(cutoff_freq, sample_rate)
    filtered_wave = apply_filter(wave, b, a)

    # تولید و اعمال LFO
    lfo = generate_lfo(lfo_freq, duration, sample_rate, lfo_amplitude)
    modulated_wave = filtered_wave * (1 + lfo)

    return wave, filtered_wave, modulated_wave


# مثال برای تولید و نمایش شکل موج‌های مختلف با VCF و LFO
wave_types = ['sine', 'sawtooth', 'triangle', 'square']
frequency = 440  # فرکانس A4
duration = 2.0  # مدت زمان 2 ثانیه
sample_rate = 44100  # نرخ نمونه‌برداری 44100 هرتز
amplitude = 1.0  # دامنه
phase = 0.0  # فاز اولیه
cutoff_freq = 1000  # فرکانس قطع فیلتر پایین گذر
lfo_freq = 5  # فرکانس LFO
lfo_amplitude = 0.5  # دامنه LFO


def exponential_segment(start_level, end_level, duration, sample_rate, alpha):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if start_level < end_level:  # Exponential rise
        return start_level + (end_level - start_level) * (np.exp(alpha * t / duration)) / np.exp(alpha * (end_level - start_level))
    else:  # Exponential decay
        return start_level - (end_level - start_level) * np.exp(-alpha * t / duration) + (end_level - start_level)

# Generate ADSR envelope


def generate_adsr_envelope(attack_time, decay_time, sustain_level, sustain_time, release_time, sample_rate=44100, alpha=5):
    # Attack phase
    attack_segment = exponential_segment(0, 1, attack_time, sample_rate, alpha)

    # Decay phase
    decay_segment = exponential_segment(
        1, sustain_level, decay_time, sample_rate, alpha)

    # Sustain phase
    sustain_segment = np.ones(int(sample_rate * sustain_time)) * sustain_level

    # Release phase
    release_segment = exponential_segment(
        sustain_level, 0, release_time, sample_rate, alpha)

    # Full envelope
    envelope = np.concatenate(
        [attack_segment, decay_segment, sustain_segment, release_segment])
    return envelope


attack_time = 0.2  # Attack time in seconds
decay_time = 0.4   # Decay time in seconds
sustain_level = 0.3  # Sustain level (0 to 1)
sustain_time = 0.1  # Sustain time in seconds
release_time = 0.3  # Release time in seconds
sample_rate2 = 4000/(
    attack_time + decay_time + sustain_time + release_time)

# Generate ADSR envelope
envelope = generate_adsr_envelope(
    attack_time, decay_time, sustain_level, sustain_time, release_time, sample_rate2)

for i, wave_type in enumerate(wave_types):
    original_wave, filtered_wave, modulated_wave = generate_waveform_with_vcf_and_lfo(
        wave_type, frequency, duration, sample_rate, amplitude, phase, cutoff_freq, lfo_freq, lfo_amplitude)

    plt.subplot(len(wave_types), 3, 3 * i + 1)
    plt.plot(original_wave[:4000])
    plt.title(f"Original {wave_type.capitalize()} Wave - {frequency} Hz")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(len(wave_types), 3, 3 * i + 2)
    plt.plot(envelope)

    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(len(wave_types), 3, 3 * i + 3)
    plt.plot(original_wave[:4000]*envelope)

    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
