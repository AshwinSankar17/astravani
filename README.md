# (FORGING) Astravaani: Streamlined Audio Processing for Synthesis and Enhancement

Loosely inspired by [audiotools](https://github.com/descriptinc/audiotools/) and [auraloss](https://github.com/csteinmetz1/auraloss/), Astravaani is a project that simplifies audio processing for synthesis and enhancement. It provides three main classes: `AudioSignal`, `AudioDataset`, `TTSDataset` which implement basic functions needed to build a TTS and/or Speech Enhancement Network while letting the user concentrate on the modelling part. Apart from this, I have also adapted some of the `LossFunctions` from auraloss.

This project is currently under development and offers minimal functionality as of now.

## Table of Contents

- [(FORGING) Astravaani: Streamlined Audio Processing for Synthesis and Enhancement](#forging-astravaani-streamlined-audio-processing-for-synthesis-and-enhancement)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [1. AudioSignal](#1-audiosignal)
    - [2. TTSDataset](#2-ttsdataset)
    - [3. CharTokenizer](#3-chartokenizer)

## Installation

To install Astravaani, you can use pip:

```bash
git clone https://github.com/iamunr4v31/astravani
cd astravani
conda env create --file=environment.yaml
pip install -e .
```

or

```bash
git clone https://github.com/iamunr4v31/astravani
cd astravani
conda create -n astravani python=3.11
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```

## Usage

### 1. AudioSignal
Here's a basic example of how to use the `AudioSignal` class to load and manipulate an audio signal:

```python
from astravani.core import AudioSignal

# Load an audio signal from file
signal = AudioSignal("path/to/audio.wav")

# Load an audio signal from an array
signal = AudioSignal(signal_array, sample_rate)

# Resample the audio signal to a new sample rate
signal.resample(new_sample_rate)

# Save the audio signal to a file
signal.write(output_path, "wav")

# Convert the audio signal to a mono signal
signal.downmix_mono()

# Convert the audio signal to a stereo signal
signal.upmix_stereo()

# Generate spectrogram
spectrogram = signal.get_spec(
  n_fft=1024,
  hop_length=256,
  win_length=1024,
  window="hann",
  center=False,
  normalized=False,
  pwr=2.0,
  eps=1e-9,
)
```

### 2. TTSDataset
Here's a basic example of how to use the `TTSDataset` class to load and preprocess a dataset for a text-to-speech model:

```python
from astravani.data.dataset import TTSDataset

dataset = TTSDataset(
  manifest_fpaths=["path/to/manifest.json"],
  sample_rate=22050,
  tokenizer=astravaani.data.tokenizer.Tokenizer(*args, **kwargs),
  sup_data_path="path/to/sup_data",
  sup_data_types=["speaker_id", "energy"],
  sort_batch_by="text",
)
```

### 3. CharTokenizer
Here's a basic example of how to use the `CharTokenizer` class to tokenize and detokenize text:

```python
from astravani.data.tokenizer import CharTokenizer

tokenizer = CharTokenizer(
  "path/to/vocab.txt",
  text_preprocessing_func=lambda x: x.lower(),
)

# Tokenize text
tokens = tokenizer.encode("Hello, world!")

# Detokenize tokens
text = tokenizer.decode(tokens)

# Batch tokenize text
batch_tokens = tokenizer.batch_encode(["Hello, world!", "How are you?"])

# Batch detokenize tokens
batch_text = tokenizer.batch_decode(batch_tokens)

# Get the vocabulary size
vocab_size = len(tokenizer)
```