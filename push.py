import numpy as np
import matplotlib.pyplot as plt

# Generate exponential segments


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


# Parameters
attack_time = 0.1  # Attack time in seconds
decay_time = 0.2   # Decay time in seconds
sustain_level = 0.5  # Sustain level (0 to 1)
sustain_time = 0.4  # Sustain time in seconds
release_time = 0.3  # Release time in seconds
sample_rate = 1000/(
    attack_time + decay_time + sustain_time + release_time)

# Generate ADSR envelope
envelope = generate_adsr_envelope(
    attack_time, decay_time, sustain_level, sustain_time, release_time, sample_rate)

# Plot ADSR envelope
plt.plot(envelope)
plt.title("ADSR Envelope")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()
