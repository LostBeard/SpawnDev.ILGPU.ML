#!/usr/bin/env python3
"""Generate NLP and audio reference outputs for ILGPU.ML unit tests.

Runs DistilBERT, GPT-2, and Whisper encoder through onnxruntime (CPU)
and saves ground truth outputs as raw binary + JSON metadata.

Also generates tokenizer reference data and audio preprocessing references.

Usage:
    python gen_nlp_audio_references.py

Requires: pip install onnxruntime numpy transformers torch scipy
"""

import os
import sys
import json
import struct
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', 'SpawnDev.ILGPU.ML')
WWWROOT = os.path.join(PROJECT_ROOT, 'SpawnDev.ILGPU.ML.Demo', 'wwwroot')
MODELS_DIR = os.path.join(WWWROOT, 'models')
REFERENCES_DIR = os.path.join(WWWROOT, 'references')

def save_binary(path, data, dtype=np.float32):
    """Save numpy array as raw binary."""
    data.astype(dtype).tofile(path)
    return os.path.getsize(path)

def save_metadata(path, meta):
    """Save metadata as JSON."""
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj

    converted = {k: convert(v) for k, v in meta.items()}
    with open(path, 'w') as f:
        json.dump(converted, f, indent=2)


# ============================================================
# DISTILBERT-SST2 — Text Classification
# ============================================================
def generate_distilbert_references():
    print('=' * 60)
    print('DISTILBERT-SST2 — Text Classification')
    print('=' * 60)

    model_path = os.path.join(MODELS_DIR, 'distilbert-sst2', 'model.onnx')
    if not os.path.exists(model_path):
        print('  SKIP — model.onnx not found')
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(MODELS_DIR, 'distilbert-sst2'),
        local_files_only=True
    )

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model_path, opts)

    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    print(f'  Inputs: {input_names}')
    print(f'  Outputs: {output_names}')

    test_cases = [
        ("I love this movie, it is absolutely fantastic!", "positive"),
        ("This is terrible, worst experience ever.", "negative"),
        ("The cat sat on the mat.", "neutral"),
        ("SpawnDev.ILGPU makes GPU computing accessible to everyone.", "positive"),
        ("The food was okay, nothing special.", "neutral"),
    ]

    ref_dir = os.path.join(REFERENCES_DIR, 'distilbert-sst2-onnx')
    os.makedirs(ref_dir, exist_ok=True)

    all_cases = []
    for text, expected_sentiment in test_cases:
        encoded = tokenizer(text, return_tensors='np', padding=False, truncation=True, max_length=128)
        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)

        # Build feed dict matching model input names
        feed = {}
        for name in input_names:
            if 'input_ids' in name.lower() or name == 'input_ids':
                feed[name] = input_ids
            elif 'attention_mask' in name.lower() or name == 'attention_mask':
                feed[name] = attention_mask

        outputs = sess.run(output_names, feed)
        logits = outputs[0]  # [1, 2]

        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        predicted_class = int(np.argmax(logits, axis=-1)[0])
        confidence = float(probs[0, predicted_class])
        sentiment = "POSITIVE" if predicted_class == 1 else "NEGATIVE"

        # Generate safe filename prefix
        prefix = text[:20].lower().replace(' ', '_').replace(',', '').replace('.', '').replace('!', '')

        # Save binary files
        save_binary(os.path.join(ref_dir, f'{prefix}_ids.bin'), input_ids.flatten(), np.int64)
        save_binary(os.path.join(ref_dir, f'{prefix}_mask.bin'), attention_mask.flatten(), np.int64)
        save_binary(os.path.join(ref_dir, f'{prefix}_logits.bin'), logits.flatten(), np.float32)

        case = {
            'text': text,
            'expected_sentiment': expected_sentiment,
            'input_ids': input_ids.flatten().tolist(),
            'attention_mask': attention_mask.flatten().tolist(),
            'seq_length': int(input_ids.shape[1]),
            'logits': logits.flatten().tolist(),
            'probabilities': probs.flatten().tolist(),
            'predicted_class': predicted_class,
            'predicted_label': sentiment,
            'confidence': confidence,
            'file_prefix': prefix,
        }
        all_cases.append(case)
        print(f'  "{text[:40]}..." -> {sentiment} ({confidence:.4f})')

    metadata = {
        'model': 'distilbert-base-uncased-finetuned-sst-2-english',
        'task': 'text-classification',
        'labels': ['NEGATIVE', 'POSITIVE'],
        'input_names': input_names,
        'output_names': output_names,
        'test_cases': all_cases,
        'onnxruntime_version': ort.__version__,
        'numpy_version': np.__version__,
    }
    save_metadata(os.path.join(ref_dir, 'metadata.json'), metadata)
    print(f'  Saved {len(all_cases)} test cases to {ref_dir}')


# ============================================================
# GPT-2 — Text Generation
# ============================================================
def generate_gpt2_references():
    print()
    print('=' * 60)
    print('GPT-2 — Text Generation')
    print('=' * 60)

    model_path = os.path.join(MODELS_DIR, 'gpt2', 'model.onnx')
    if not os.path.exists(model_path):
        print('  SKIP — model.onnx not found')
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(MODELS_DIR, 'gpt2'),
        local_files_only=True
    )

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model_path, opts)

    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    print(f'  Inputs: {input_names}')
    print(f'  Outputs: {output_names}')

    test_prompts = [
        "The cat sat on the",
        "Once upon a time",
        "Hello, my name is",
        "The future of AI is",
        "In a galaxy far far",
    ]

    ref_dir = os.path.join(REFERENCES_DIR, 'gpt2-onnx')
    os.makedirs(ref_dir, exist_ok=True)

    all_cases = []
    for prompt in test_prompts:
        encoded = tokenizer(prompt, return_tensors='np')
        input_ids = encoded['input_ids'].astype(np.int64)

        # Build feed dict
        seq_len = input_ids.shape[1]
        feed = {}
        for name in input_names:
            if 'input_ids' in name.lower() or name == 'input_ids':
                feed[name] = input_ids
            elif 'attention_mask' in name.lower():
                feed[name] = np.ones_like(input_ids)
            elif 'position_ids' in name.lower():
                feed[name] = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)

        outputs = sess.run(output_names, feed)
        logits = outputs[0]  # [1, seq_len, vocab_size]

        # Next token prediction (last position)
        next_token_logits = logits[0, -1, :]  # [vocab_size]
        next_token_id = int(np.argmax(next_token_logits))
        next_token = tokenizer.decode([next_token_id])

        # Top 5 predictions
        top5_indices = np.argsort(next_token_logits)[-5:][::-1]
        top5_tokens = [tokenizer.decode([idx]) for idx in top5_indices]
        top5_logits = next_token_logits[top5_indices].tolist()

        prefix = prompt[:20].lower().replace(' ', '_').replace(',', '').replace('.', '')

        # Save binary files
        save_binary(os.path.join(ref_dir, f'{prefix}_ids.bin'), input_ids.flatten(), np.int64)
        save_binary(os.path.join(ref_dir, f'{prefix}_next_logits.bin'), next_token_logits, np.float32)

        case = {
            'prompt': prompt,
            'input_ids': input_ids.flatten().tolist(),
            'seq_length': int(input_ids.shape[1]),
            'next_token_id': next_token_id,
            'next_token': next_token,
            'top5_ids': top5_indices.tolist(),
            'top5_tokens': top5_tokens,
            'top5_logits': top5_logits,
            'logits_shape': list(logits.shape),
            'file_prefix': prefix,
        }
        all_cases.append(case)
        print(f'  "{prompt}" -> "{next_token}" (id={next_token_id}), top5={top5_tokens}')

    metadata = {
        'model': 'gpt2',
        'task': 'text-generation',
        'vocab_size': int(logits.shape[-1]),
        'input_names': input_names,
        'output_names': output_names,
        'test_cases': all_cases,
        'onnxruntime_version': ort.__version__,
        'numpy_version': np.__version__,
    }
    save_metadata(os.path.join(ref_dir, 'metadata.json'), metadata)
    print(f'  Saved {len(all_cases)} test cases to {ref_dir}')


# ============================================================
# WHISPER-TINY — Speech-to-Text (Encoder)
# ============================================================
def generate_whisper_references():
    print()
    print('=' * 60)
    print('WHISPER-TINY — Speech-to-Text (Encoder)')
    print('=' * 60)

    encoder_path = os.path.join(MODELS_DIR, 'whisper-tiny', 'encoder_model.onnx')
    if not os.path.exists(encoder_path):
        print('  SKIP — encoder_model.onnx not found')
        return

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(encoder_path, opts)

    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    print(f'  Inputs: {input_names}')
    print(f'  Outputs: {output_names}')

    ref_dir = os.path.join(REFERENCES_DIR, 'whisper-tiny-onnx')
    os.makedirs(ref_dir, exist_ok=True)

    # Test 1: Generate a synthetic tone and compute mel spectrogram
    print('  Generating synthetic audio test cases...')

    sample_rate = 16000
    duration = 3.0  # seconds

    test_cases = []

    # Test case 1: 440Hz sine tone
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    tone_440 = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Test case 2: Silence (zeros)
    silence = np.zeros(int(sample_rate * duration), dtype=np.float32)

    # Test case 3: White noise (seeded for reproducibility)
    rng = np.random.RandomState(42)
    noise = rng.randn(int(sample_rate * duration)).astype(np.float32) * 0.3

    audio_cases = [
        ('tone_440hz', tone_440, '440Hz sine tone'),
        ('silence', silence, 'silence'),
        ('white_noise', noise, 'white noise (seed=42)'),
    ]

    for name, audio, description in audio_cases:
        # Compute log-mel spectrogram (Whisper expects [1, 80, 3000])
        mel = compute_whisper_mel(audio, sample_rate)
        mel_input = mel[np.newaxis, :, :]  # [1, 80, 3000]

        # Run encoder
        feed = {input_names[0]: mel_input.astype(np.float32)}
        encoder_output = sess.run(output_names, feed)[0]  # [1, 1500, 384]

        # Save binary
        save_binary(os.path.join(ref_dir, f'{name}_mel.bin'), mel_input.flatten())
        save_binary(os.path.join(ref_dir, f'{name}_encoder_output.bin'), encoder_output.flatten())
        save_binary(os.path.join(ref_dir, f'{name}_audio.bin'), audio)

        case = {
            'name': name,
            'description': description,
            'audio_samples': len(audio),
            'sample_rate': sample_rate,
            'mel_shape': list(mel_input.shape),
            'encoder_output_shape': list(encoder_output.shape),
            'encoder_first_10': encoder_output.flatten()[:10].tolist(),
            'encoder_mean': float(encoder_output.mean()),
            'encoder_std': float(encoder_output.std()),
        }
        test_cases.append(case)
        print(f'    {name}: mel{list(mel_input.shape)} -> encoder{list(encoder_output.shape)}, '
              f'mean={encoder_output.mean():.4f}')

    metadata = {
        'model': 'whisper-tiny',
        'task': 'speech-to-text',
        'encoder_input_names': input_names,
        'encoder_output_names': output_names,
        'n_mels': 80,
        'n_audio_ctx': 1500,
        'n_audio_state': 384,
        'test_cases': test_cases,
        'onnxruntime_version': ort.__version__,
        'numpy_version': np.__version__,
    }
    save_metadata(os.path.join(ref_dir, 'metadata.json'), metadata)
    print(f'  Saved {len(test_cases)} test cases to {ref_dir}')


def compute_whisper_mel(audio, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160, max_frames=3000):
    """Compute Whisper-compatible log-mel spectrogram.

    Matches OpenAI Whisper's preprocessing:
    - STFT with Hann window, n_fft=400, hop=160
    - 80-bin mel filterbank
    - Log-mel with log10, clamped, normalized

    Returns: [n_mels, max_frames] float32
    """
    from scipy.signal import stft as scipy_stft

    # Pad or trim to 30 seconds
    target_length = sample_rate * 30
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    # STFT
    window = np.hanning(n_fft).astype(np.float32)
    # Use manual STFT for exact control
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    magnitudes = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        magnitudes[:, i] = np.abs(spectrum) ** 2

    # Mel filterbank
    filters = mel_filterbank(sample_rate, n_fft, n_mels)
    mel_spec = filters @ magnitudes  # [n_mels, num_frames]

    # Log-mel (Whisper uses log10, clamp, normalize)
    mel_spec = np.clip(mel_spec, a_min=1e-10, a_max=None)
    log_mel = np.log10(mel_spec)
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0

    # Pad to max_frames
    if log_mel.shape[1] < max_frames:
        log_mel = np.pad(log_mel, ((0, 0), (0, max_frames - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :max_frames]

    return log_mel.astype(np.float32)


def mel_filterbank(sample_rate, n_fft, n_mels, fmin=0, fmax=None):
    """Create mel filterbank matrix [n_mels, n_fft//2+1]."""
    if fmax is None:
        fmax = sample_rate / 2

    n_freqs = n_fft // 2 + 1

    # Mel scale conversion
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    freq_bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left = freq_bins[i]
        center = freq_bins[i + 1]
        right = freq_bins[i + 2]

        for j in range(left, center):
            if center > left:
                filters[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filters[i, j] = (right - j) / (right - center)

    # Normalize (slaney style)
    enorm = 2.0 / (hz_points[2:n_mels+2] - hz_points[:n_mels])
    filters *= enorm[:, np.newaxis]

    return filters


# ============================================================
# TOKENIZER REFERENCE DATA
# ============================================================
def generate_tokenizer_references():
    print()
    print('=' * 60)
    print('TOKENIZER REFERENCES')
    print('=' * 60)

    ref_dir = os.path.join(REFERENCES_DIR, 'tokenizer-tests')
    os.makedirs(ref_dir, exist_ok=True)

    all_results = {}

    # DistilBERT tokenizer
    distilbert_path = os.path.join(MODELS_DIR, 'distilbert-sst2')
    if os.path.exists(distilbert_path):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(distilbert_path, local_files_only=True)

        test_texts = [
            "Hello world",
            "I love this movie!",
            "The cat sat on the mat.",
            "",
            "SpawnDev",
            "GPU computing is the future of machine learning.",
            "a " * 200,  # long text (truncation test)
        ]

        cases = []
        for text in test_texts:
            encoded = tokenizer(text, padding=False, truncation=True, max_length=128)
            decoded = tokenizer.decode(encoded['input_ids'], skip_special_tokens=True)

            case = {
                'text': text[:100],  # truncate display for very long
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'decoded': decoded[:100],
                'num_tokens': len(encoded['input_ids']),
            }
            cases.append(case)
            print(f'  DistilBERT: "{text[:30]}..." -> {len(encoded["input_ids"])} tokens')

        all_results['distilbert'] = {
            'model': 'distilbert-base-uncased-finetuned-sst-2-english',
            'vocab_size': tokenizer.vocab_size,
            'pad_token_id': tokenizer.pad_token_id,
            'cls_token_id': tokenizer.cls_token_id,
            'sep_token_id': tokenizer.sep_token_id,
            'unk_token_id': tokenizer.unk_token_id,
            'test_cases': cases,
        }

    # GPT-2 tokenizer
    gpt2_path = os.path.join(MODELS_DIR, 'gpt2')
    if os.path.exists(gpt2_path):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(gpt2_path, local_files_only=True)

        test_texts = [
            "Hello world",
            "The cat sat on the",
            "Once upon a time",
            "SpawnDev.ILGPU.ML",
            "1 + 1 = 2",
            "",
        ]

        cases = []
        for text in test_texts:
            encoded = tokenizer(text)
            decoded = tokenizer.decode(encoded['input_ids'])

            case = {
                'text': text,
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'decoded': decoded,
                'num_tokens': len(encoded['input_ids']),
            }
            cases.append(case)
            print(f'  GPT-2: "{text}" -> {encoded["input_ids"]} ({len(encoded["input_ids"])} tokens)')

        all_results['gpt2'] = {
            'model': 'gpt2',
            'vocab_size': tokenizer.vocab_size,
            'eos_token_id': tokenizer.eos_token_id,
            'bos_token_id': tokenizer.bos_token_id,
            'test_cases': cases,
        }

    save_metadata(os.path.join(ref_dir, 'tokenizer_test_cases.json'), all_results)
    print(f'  Saved tokenizer references to {ref_dir}')


# ============================================================
# AUDIO PREPROCESSING REFERENCE DATA
# ============================================================
def generate_audio_preprocessing_references():
    print()
    print('=' * 60)
    print('AUDIO PREPROCESSING REFERENCES')
    print('=' * 60)

    ref_dir = os.path.join(REFERENCES_DIR, 'audio-preprocessing')
    os.makedirs(ref_dir, exist_ok=True)

    sample_rate = 16000

    # Test 1: Mel filterbank shape and values
    print('  Generating mel filterbank reference...')
    filters = mel_filterbank(sample_rate, n_fft=400, n_mels=80)
    save_binary(os.path.join(ref_dir, 'mel_filterbank_80x201.bin'), filters.flatten())

    # Test 2: Known audio -> mel spectrogram
    print('  Generating mel spectrogram references...')

    # 440Hz tone for 1 second
    t = np.linspace(0, 1.0, sample_rate, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Compute STFT magnitudes manually
    n_fft = 400
    hop = 160
    window = np.hanning(n_fft).astype(np.float32)
    num_frames = 1 + (len(tone) - n_fft) // hop
    magnitudes = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop
        frame = tone[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        magnitudes[:, i] = np.abs(spectrum) ** 2

    save_binary(os.path.join(ref_dir, 'tone_440hz_stft_magnitudes.bin'), magnitudes.flatten())

    mel_spec = filters @ magnitudes
    save_binary(os.path.join(ref_dir, 'tone_440hz_mel_spec.bin'), mel_spec.flatten())

    # Log-mel
    log_mel = np.log10(np.clip(mel_spec, 1e-10, None))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    save_binary(os.path.join(ref_dir, 'tone_440hz_log_mel.bin'), log_mel.flatten())

    # Test 3: Hann window
    hann = np.hanning(400).astype(np.float32)
    save_binary(os.path.join(ref_dir, 'hann_window_400.bin'), hann)

    # Test 4: Hz <-> Mel conversion
    hz_values = np.array([0, 100, 200, 440, 1000, 2000, 4000, 8000], dtype=np.float32)
    mel_values = 2595.0 * np.log10(1.0 + hz_values / 700.0)

    # Test 5: Resampling (44100 -> 16000)
    t_44k = np.linspace(0, 0.1, int(44100 * 0.1), dtype=np.float32)
    tone_44k = 0.5 * np.sin(2 * np.pi * 440 * t_44k)
    # Linear interpolation resample
    t_16k = np.linspace(0, 0.1, int(16000 * 0.1), dtype=np.float32)
    tone_16k = np.interp(t_16k, t_44k, tone_44k).astype(np.float32)

    save_binary(os.path.join(ref_dir, 'resample_44100_input.bin'), tone_44k)
    save_binary(os.path.join(ref_dir, 'resample_16000_output.bin'), tone_16k)

    metadata = {
        'mel_filterbank': {
            'shape': list(filters.shape),
            'sample_rate': sample_rate,
            'n_fft': 400,
            'n_mels': 80,
        },
        'stft': {
            'n_fft': 400,
            'hop_length': 160,
            'window': 'hann',
            'magnitudes_shape': list(magnitudes.shape),
        },
        'mel_spectrogram': {
            'shape': list(mel_spec.shape),
            'min': float(mel_spec.min()),
            'max': float(mel_spec.max()),
        },
        'log_mel': {
            'shape': list(log_mel.shape),
            'min': float(log_mel.min()),
            'max': float(log_mel.max()),
        },
        'hz_to_mel': {
            'hz_values': hz_values.tolist(),
            'mel_values': mel_values.tolist(),
        },
        'resample': {
            'input_rate': 44100,
            'output_rate': 16000,
            'input_samples': len(tone_44k),
            'output_samples': len(tone_16k),
            'frequency': 440,
            'duration': 0.1,
        },
        'hann_window': {
            'size': 400,
            'first_5': hann[:5].tolist(),
            'last_5': hann[-5:].tolist(),
        },
    }
    save_metadata(os.path.join(ref_dir, 'metadata.json'), metadata)
    print(f'  Saved audio preprocessing references to {ref_dir}')
    print(f'  Mel filterbank: {list(filters.shape)}')
    print(f'  STFT magnitudes: {list(magnitudes.shape)}')
    print(f'  Mel spectrogram: {list(mel_spec.shape)}')
    print(f'  Log-mel: {list(log_mel.shape)}')
    print(f'  Resample: {len(tone_44k)} -> {len(tone_16k)} samples')


# ============================================================
# TEXT PREPROCESSING REFERENCE DATA
# ============================================================
def generate_text_preprocessing_references():
    print()
    print('=' * 60)
    print('TEXT PREPROCESSING REFERENCES')
    print('=' * 60)

    ref_dir = os.path.join(REFERENCES_DIR, 'text-preprocessing')
    os.makedirs(ref_dir, exist_ok=True)

    test_cases = {}

    # Attention mask creation
    test_cases['attention_mask'] = [
        {'input_ids': [101, 2023, 2003, 1037, 3231, 102, 0, 0], 'pad_token': 0,
         'expected': [1, 1, 1, 1, 1, 1, 0, 0]},
        {'input_ids': [101, 7592, 102], 'pad_token': 0,
         'expected': [1, 1, 1]},
        {'input_ids': [0, 0, 0, 0], 'pad_token': 0,
         'expected': [0, 0, 0, 0]},
    ]

    # Pad or truncate
    test_cases['pad_or_truncate'] = [
        {'input_ids': [1, 2, 3], 'max_length': 6, 'pad_token': 0,
         'expected': [1, 2, 3, 0, 0, 0]},
        {'input_ids': [1, 2, 3, 4, 5, 6, 7], 'max_length': 4, 'pad_token': 0,
         'expected': [1, 2, 3, 4]},
        {'input_ids': [1, 2, 3], 'max_length': 3, 'pad_token': 0,
         'expected': [1, 2, 3]},
    ]

    # Token type IDs (BERT segment)
    test_cases['token_type_ids'] = [
        {'segment_a_length': 5, 'segment_b_length': 3, 'max_length': 10,
         'expected': [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]},
    ]

    # Position IDs
    test_cases['position_ids'] = [
        {'length': 6, 'expected': [0, 1, 2, 3, 4, 5]},
    ]

    # Softmax (stable)
    logits = [2.0, 1.0, 0.1]
    exp_logits = np.exp(np.array(logits) - max(logits))
    softmax_result = (exp_logits / exp_logits.sum()).tolist()
    test_cases['softmax'] = [
        {'logits': logits, 'expected': softmax_result},
        {'logits': [1000.0, 1000.0, 1000.0],
         'expected': [1/3, 1/3, 1/3]},  # numerical stability test
        {'logits': [-1000.0, 0.0, 1000.0],
         'expected': [0.0, 0.0, 1.0]},  # extreme values
    ]

    # TopK
    test_cases['topk'] = [
        {'scores': [0.1, 0.5, 0.2, 0.9, 0.3], 'k': 3,
         'expected_indices': [3, 1, 4], 'expected_scores': [0.9, 0.5, 0.3]},
    ]

    # Cosine similarity
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    c = [1.0, 0.0, 0.0]
    test_cases['cosine_similarity'] = [
        {'a': a, 'b': b, 'expected': 0.0},      # orthogonal
        {'a': a, 'b': c, 'expected': 1.0},      # identical
        {'a': a, 'b': [-1, 0, 0], 'expected': -1.0},  # opposite
        {'a': [1, 1, 0], 'b': [1, 0, 0], 'expected': float(1/np.sqrt(2))},  # 45 degrees
    ]

    save_metadata(os.path.join(ref_dir, 'text_preprocessing_tests.json'), test_cases)
    print(f'  Saved text preprocessing references to {ref_dir}')
    for category, cases in test_cases.items():
        print(f'    {category}: {len(cases)} test cases')


# ============================================================
# MAIN
# ============================================================
def main():
    print('ILGPU.ML NLP & Audio Reference Generator')
    print('=' * 60)
    print(f'onnxruntime {ort.__version__}, numpy {np.__version__}')
    print()

    generate_distilbert_references()
    generate_gpt2_references()
    generate_whisper_references()
    generate_tokenizer_references()
    generate_audio_preprocessing_references()
    generate_text_preprocessing_references()

    print()
    print('=' * 60)
    print('ALL REFERENCES GENERATED')
    print('=' * 60)


if __name__ == '__main__':
    main()
