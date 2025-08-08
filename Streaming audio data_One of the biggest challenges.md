
***

# 📡 Streaming Audio Data with 🤗 Datasets

## Key Points

- **Audio Datasets are Large**
    - 1 minute of uncompressed CD-quality audio ≈ 5 MB.
    - Example: GigaSpeech “xs” (10 hrs) ≈ 13 GB; “xl” (10,000 hrs) ≈ 1 TB.
    - Most personal computers can't store very large datasets locally.
- **Streaming Mode to the Rescue**
    - 🤗 Datasets library allows streaming audio data.
    - Only loads and processes examples as you iterate.
    - No huge disk space required—datasets can be arbitrarily large.


## Advantages of Streaming Mode

- **No Disk Space Needed:** Only the currently used example is in memory.
- **Quick Start:** Start training/processing immediately—no waiting for download.
- **Easy Experimentation:** Test code with first few samples before committing resources.


## Limitations

- **No Caching:** Data isn’t saved locally; each run re-downloads and processes samples.
- **No Indexing:** Can’t use `dataset["train"][i]`—must iterate.


## Using Streaming Mode

```python
from datasets import load_dataset

# Load GigaSpeech with streaming enabled
gigaspeech = load_dataset("speechcolab/gigaspeech", "xs", streaming=True)
```

- Preprocessing can be applied on streaming datasets as usual.
- Access samples using iteration (not indexing):

```python
# Get first training example
sample = next(iter(gigaspeech["train"]))
print(sample)
```

- To preview the first *n* examples, use `.take(n)` and convert to a list:

```python
# Get first 2 samples in the training split
head_samples = list(gigaspeech["train"].take(2))
print(head_samples)
```


## Example Streamed Sample Structure

```json
{
  "segment_id": "...",
  "speaker": "...",
  "text": "...",
  "audio": {
    "path": "...",
    "array": [...],
    "sampling_rate": 16000
  },
  "begin_time": ...,
  "end_time": ...,
  "audio_id": "...",
  "title": "...",
  "url": "...",
  "source": ...,
  "category": ...,
  "original_full_path": "..."
}
```


## When to Use Streaming vs. Downloading

- Use **streaming** for:
    - Exploring very large datasets
    - One-off experiments or evaluation across many datasets
- Use **download** for:
    - Repeated use of the same dataset (to cache processed data)

***

**Summary:**
Streaming datasets lets you work with gigantic audio corpora without worrying about your disk space or waiting hours for a download. Iterate to access samples, and preprocess just like with local datasets!

***

You can copy-paste this summary into a file named `streaming_audio_datasets.md` and upload it to your GitHub repository.

