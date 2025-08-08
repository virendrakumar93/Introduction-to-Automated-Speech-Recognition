
# 📥 Load and Explore an Audio Dataset (MINDS-14)

## 📦 Installing Dependencies
```bash
pip install datasets[audio] gradio librosa matplotlib
```

## 🧠 What is 🤗 Datasets?
- Library for loading/preprocessing datasets (text, image, audio, etc).
- Integrates with Hugging Face Hub.
- `load_dataset()` downloads & prepares datasets in one line.

---

## 🔊 Loading the MINDS-14 Dataset

```python
from datasets import load_dataset

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds
```
**Output:**
- Dataset with 654 rows and 6 columns: `path`, `audio`, `transcription`, `english_transcription`, `intent_class`, `lang_id`.

---

## 🔍 Inspecting an Example

```python
example = minds[0]
example
```
**Important Fields:**
- `audio['array']`: 1D NumPy array of waveform.
- `audio['sampling_rate']`: Sampling rate (e.g., 8000 Hz).
- `transcription`: Spoken content.
- `intent_class`: Numeric label.

### Decode Class Label
```python
id2label = minds.features["intent_class"].int2str
id2label(example["intent_class"])  # Output: "pay_bill"
```

---

## 🧹 Cleaning the Dataset
Remove irrelevant features:
```python
columns_to_remove = ["lang_id", "english_transcription"]
minds = minds.remove_columns(columns_to_remove)
minds
```
**Output:** Dataset with columns: `path`, `audio`, `transcription`, `intent_class`.

---

## 🔊 Listening to Random Samples (Gradio UI)

```python
import gradio as gr

def generate_audio():
    example = minds.shuffle()[0]
    audio = example["audio"]
    return (audio["sampling_rate"], audio["array"]), id2label(example["intent_class"])

with gr.Blocks() as demo:
    with gr.Column():
        for _ in range(4):
            audio, label = generate_audio()
            output = gr.Audio(audio, label=label)

demo.launch(debug=True)
```

---

## 📈 Visualizing the Waveform

```python
import librosa
import matplotlib.pyplot as plt
import librosa.display

array = example["audio"]["array"]
sampling_rate = example["audio"]["sampling_rate"]

plt.figure(figsize=(12, 4))
librosa.display.waveshow(array, sr=sampling_rate)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()
```

---

## 🧪 Try This!
- Download another language/dialect from the `PolyAI/minds14` dataset.
- Listen and visualize some examples.
- Observe variation in accents, tempo, and audio quality.

🔗 [Browse available languages on the Hugging Face Hub](https://huggingface.co/datasets/PolyAI/minds14)
