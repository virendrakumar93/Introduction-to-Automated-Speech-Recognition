
# 📒 Introduction to Audio Data (Summary Notes)

## 🔊 What is Audio?
- Audio is a **continuous analog signal** with infinite values.
- To process/store it digitally, we **sample** it into discrete values → **Digital Audio**.

## 🗂️ Audio File Formats
- `.wav`: Uncompressed waveform format.
- `.flac`: Lossless compression.
- `.mp3`: Lossy compression.
- All formats represent digitized versions of audio.

## 🎙️ Audio Capture: From Analog to Digital
1. **Microphone** → Converts sound to electrical signal.
2. **ADC (Analog-to-Digital Converter)** → Samples the signal at regular intervals.

## ⏱️ Sampling & Sampling Rate
- **Sampling**: Measuring signal at fixed intervals.
- **Sampling Rate (Hz)**: Samples per second.
  - CD audio: `44,100 Hz`
  - High-res audio: `192,000 Hz`
  - Common in speech ML: `16,000 Hz`

> **Nyquist Limit**: Max frequency captured = half the sampling rate.

### Example
- `5 sec audio @ 16 kHz` → `80,000 samples`
- `5 sec audio @ 8 kHz` → `40,000 samples`

## 🎚️ Amplitude & Bit Depth
- **Amplitude**: Loudness of sound (measured in dB).
- **Bit Depth**: Precision of amplitude values.
  - 16-bit → 65,536 levels
  - 24-bit → ~16 million levels
  - 32-bit → Stored as floating-point, range [-1.0, 1.0]

> Higher bit depth = less quantization noise

## 📈 Visualizing Audio: Waveform
- Waveform = Plot of amplitude vs time.

### Code: Plot waveform using Librosa
```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load example audio
array, sampling_rate = librosa.load(librosa.ex("trumpet"))

# Plot waveform
plt.figure(figsize=(12, 4))
librosa.display.waveshow(array, sr=sampling_rate)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()
```

## 🔁 Frequency Spectrum (Fourier Transform)
- **DFT (Discrete Fourier Transform)** shows frequency content at a moment.
- Use **log scale** on frequency axis for better visibility.
- Librosa provides conversion to dB.

### Code: Plot frequency spectrum
```python
import numpy as np

dft_input = array[:4096]
window = np.hanning(len(dft_input))
windowed_input = dft_input * window
dft = np.fft.rfft(windowed_input)

amplitude = np.abs(dft)
amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)
frequency = librosa.fft_frequencies(sr=sampling_rate, n_fft=len(dft_input))

plt.figure(figsize=(12, 4))
plt.plot(frequency, amplitude_db)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.xscale("log")
plt.title("Frequency Spectrum")
plt.show()
```

## 📊 Spectrogram (STFT)
- **Spectrogram** = Time vs Frequency vs Amplitude
- Uses **Short Time Fourier Transform (STFT)**
- Useful for detecting sound events over time

### Code: Plot spectrogram
```python
D = librosa.stft(array)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(12, 4))
librosa.display.specshow(S_db, x_axis="time", y_axis="hz", sr=sampling_rate)
plt.colorbar()
plt.title("Spectrogram (STFT)")
plt.show()
```

## 📉 Mel Spectrogram
- Human hearing is **non-linear** → more sensitive to low frequencies.
- **Mel scale** mimics human ear.
- **Mel Spectrogram** = STFT + Mel filterbanks

### Code: Plot mel spectrogram
```python
S = librosa.feature.melspectrogram(y=array, sr=sampling_rate, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(12, 4))
librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sampling_rate, fmax=8000)
plt.colorbar()
plt.title("Mel Spectrogram")
plt.show()
```

## ⚠️ Notes on Usage
- Match sampling rates across all dataset files.
- ML models often prefer **log-mel spectrograms** over raw waveforms.
- Spectrograms can be inverted to waveform using **Griffin-Lim** or **vocoder models**.

## ✅ Summary

| Concept            | Description                            |
|--------------------|----------------------------------------|
| Sampling Rate      | How often audio is measured (Hz)       |
| Bit Depth          | Precision of amplitude                 |
| Waveform           | Amplitude vs Time                      |
| Frequency Spectrum | Frequency vs Amplitude (snapshot)      |
| Spectrogram        | Frequency vs Time (changing over time) |
| Mel Spectrogram    | Perceptual frequency representation    |
