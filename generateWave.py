import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
# تابع تولید سیگنال سینوسی با ناتنظیمی


def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=1.0, phase=0.0, detune=0.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    detuned_frequency = frequency * (1 + detune)
    wave = amplitude * np.sin(2 * np.pi * detuned_frequency * t + phase)
    return wave

# تابع تولید موج دندانه اره‌ای با ناتنظیمی


def generate_sawtooth_wave(frequency, duration, sample_rate=44100, amplitude=1.0, phase=0.0, detune=0.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    detuned_frequency = frequency * (1 + detune)
    wave = amplitude * (2 * (t * detuned_frequency -
                        np.floor(0.5 + t * detuned_frequency)))
    return wave

# تابع تولید موج مثلثی با ناتنظیمی


def generate_triangle_wave(frequency, duration, sample_rate=44100, amplitude=1.0, phase=0.0, detune=0.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    detuned_frequency = frequency * (1 + detune)
    wave = amplitude * (2 * np.abs(2 * (t * detuned_frequency -
                        np.floor(0.5 + t * detuned_frequency))) - 1)
    return wave

# تابع تولید موج مربعی با ناتنظیمی


def generate_square_wave(frequency, duration, sample_rate=44100, amplitude=1.0, phase=0.0, detune=0.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    detuned_frequency = frequency * (1 + detune)
    wave = amplitude * \
        np.sign(np.sin(2 * np.pi * detuned_frequency * t + phase))
    return wave

# تابع اصلی برای تولید شکل موج با پارامترهای مختلف و ناتنظیمی

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


def generate_waveform(wave_type, frequency, duration, sample_rate=44100, amplitude=1.0, phase=0.0, detune=0.0, num_detuned=1, cutoff_freq=1000, lfo_freq=5, lfo_amplitude=0.5):
    waves = []
    for i in range(num_detuned):
        detune_factor = detune * (i - num_detuned // 2)
        if wave_type == 'sine':
            wave = generate_sine_wave(
                frequency, duration, sample_rate, amplitude, phase, detune_factor)
        elif wave_type == 'sawtooth':
            wave = generate_sawtooth_wave(
                frequency, duration, sample_rate, amplitude, phase, detune_factor)
        elif wave_type == 'triangle':
            wave = generate_triangle_wave(
                frequency, duration, sample_rate, amplitude, phase, detune_factor)
        elif wave_type == 'square':
            wave = generate_square_wave(
                frequency, duration, sample_rate, amplitude, phase, detune_factor)
        else:
            raise ValueError(
                "Invalid wave type. Choose from 'sine', 'sawtooth', 'triangle', 'square'.")
        waves.append(wave)
    combined_wave = np.sum(waves, axis=0) / num_detuned
    b, a = design_lowpass_filter(cutoff_freq, sample_rate)
    filtered_wave = apply_filter(combined_wave, b, a)

    # تولید و اعمال LFO
    lfo = generate_lfo(lfo_freq, duration, sample_rate, lfo_amplitude)
    modulated_wave = filtered_wave * (1 + lfo)

    return combined_wave, filtered_wave, modulated_wave


# مثال برای تولید و نمایش شکل موج‌های مختلف با ناتنظیمی
# می‌تواند یکی از 'sine', 'sawtooth', 'triangle', 'square' باشد
wave_types = ['sine', 'sawtooth', 'triangle', 'square']
frequency = 440  # فرکانس A4
duration = 2.0  # مدت زمان 2 ثانیه
sample_rate = 44100  # نرخ نمونه‌برداری 44100 هرتز
amplitude = 1.0  # دامنه
phase = 0.0  # فاز اولیه
detune = 0.1  # ناتنظیمی (درصدی از فرکانس اصلی)
num_detuned = 2  # تعداد اسیلاتورهای ناتنظیم
cutoff_freq = 1000  # فرکانس قطع فیلتر پایین گذر
lfo_freq = 5  # فرکانس LFO
lfo_amplitude = 0.5  # دامنه LFO

plt.figure(figsize=(12, 14))

for i, wave_type in enumerate(wave_types):
    original_wave, filtered_wave, modulated_wave = generate_waveform(
        wave_type, frequency, duration, sample_rate, amplitude, phase, detune, num_detuned, cutoff_freq, lfo_freq, lfo_amplitude)

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
