
# 🧹 Preprocessing an Audio Dataset with 🤗 Datasets

Once you’ve loaded an audio dataset, it must be preprocessed before training or inference. This involves:

- Resampling the audio data
- Filtering the dataset (e.g. by duration)
- Converting audio to the model’s expected input format

---

## 🔁 Resampling the Audio Data

📌 **Problem:** Sampling rate of dataset ≠ sampling rate expected by model.

👉 Most pretrained models (like Whisper) expect **16 kHz** audio. MINDS-14 is sampled at **8 kHz**.

📦 **Solution:** Use `cast_column()` to resample "on-the-fly":

```python
from datasets import Audio

minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
```

✅ Check new sampling rate:

```python
minds[0]["audio"]["sampling_rate"]  # should return 16000
```

📚 **Resampling Notes:**
- Upsampling: interpolate missing samples using curve approximation.
- Downsampling: must filter out higher frequencies before discarding samples (to avoid aliasing).

---

## 🔎 Filtering the Dataset

📌 **Use case:** Remove audio samples longer than 20 seconds.

### Step 1: Create a duration column

```python
import librosa

MAX_DURATION_IN_SECONDS = 20.0
new_column = [librosa.get_duration(path=x) for x in minds["path"]]
minds = minds.add_column("duration", new_column)
```

### Step 2: Define filter function

```python
def is_audio_length_in_range(input_length):
    return input_length < MAX_DURATION_IN_SECONDS
```

### Step 3: Apply filter and cleanup

```python
minds = minds.filter(is_audio_length_in_range, input_columns=["duration"])
minds = minds.remove_columns(["duration"])
```

✅ From 654 → 624 samples after filtering

---

## 🎛️ Preprocessing Audio Data

📌 **Goal:** Convert raw audio → input features suitable for models (e.g., Whisper).

🤗 Transformers provide a `FeatureExtractor` for each audio model.

### Example: Whisper

- Pads/truncates audio to **30s**
- Converts audio → **log-mel spectrograms**
- No attention mask needed (Whisper is trained to ignore silence)

### Load the feature extractor

```python
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
```

### Preprocess one sample

```python
def prepare_dataset(example):
    audio = example["audio"]
    features = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        padding=True
    )
    return features
```

### Apply to all examples

```python
minds = minds.map(prepare_dataset)
```

✅ New column `"input_features"` (log-mel spectrograms) added.

---

## 📊 Visualize Log-Mel Spectrogram

```python
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

example = minds[0]
input_features = example["input_features"]

plt.figure().set_figwidth(12)
librosa.display.specshow(
    np.asarray(input_features[0]),
    x_axis="time",
    y_axis="mel",
    sr=feature_extractor.sampling_rate,
    hop_length=feature_extractor.hop_length,
)
plt.colorbar()
plt.title("Log-Mel Spectrogram")
plt.show()
```

---

## 🧠 Preprocessing + Tokenization = Processor

For multimodal tasks (e.g., speech recognition), you need both:

- Feature extractor (for audio)
- Tokenizer (for text)

🤗 Transformers provides `AutoProcessor` to load both:

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("openai/whisper-small")
```

✍️ You can extend `prepare_dataset()` to include more custom processing if needed.

---

✅ **You’re now ready to preprocess audio datasets for any 🤗 model!**
