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

plt.figure(figsize=(12, 14))

for i, wave_type in enumerate(wave_types):
    original_wave, filtered_wave, modulated_wave = generate_waveform_with_vcf_and_lfo(
        wave_type, frequency, duration, sample_rate, amplitude, phase, cutoff_freq, lfo_freq, lfo_amplitude)

    plt.subplot(len(wave_types), 3, 3 * i + 1)
    plt.plot(original_wave[:1000])
    plt.title(f"Original {wave_type.capitalize()} Wave - {frequency} Hz")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(len(wave_types), 3, 3 * i + 2)
    plt.plot(filtered_wave[:1000])
    plt.title(f"Filtered {wave_type.capitalize()
                          } Wave with VCF - {cutoff_freq} Hz Cutoff")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(len(wave_types), 3, 3 * i + 3)
    plt.plot(modulated_wave[:1000])
    plt.title(f"Modulated {wave_type.capitalize()
                           } Wave with LFO - {lfo_freq} Hz")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
