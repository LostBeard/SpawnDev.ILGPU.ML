# The reference clip

`librivox-public-domain.wav` — 4.0 s, 16 kHz, mono. This is the voice every fixture in this harness is
cloned from.

## Where it comes from

| | |
|---|---|
| Source | LibriVox recording of *Jacko and Jumpo Kinkytail* by Howard R. Garis, chapter 1, read by "Shasta, Oakland, California" |
| Archive item | `jacko_and_jumpo_2007_librivox` on archive.org |
| Rights | **Creative Commons Public Domain Mark 1.0** (`creativecommons.org/publicdomain/mark/1.0/`), as recorded in the item's own metadata |
| Excerpt | 8.8 s to 12.8 s of chapter 1 |
| Processing | decoded to 16 kHz mono, then **peak-normalised by +8.9 dB** (the excerpt sat at −31 dBFS). No compression, no filtering, no editing of the speech itself. |

## What it says

```
All LibriVox recordings are in the public domain.
```

**Exactly that.** This sentence was chosen because LibriVox's spoken preamble is standardized wording, so
the transcript is KNOWN text rather than a transcription that might be wrong — which is the entire
requirement for a reference clip and the thing the previous one failed.

## Why this replaced the packaged sample clip

ZipVoice ships a `prompt.wav`, and every fixture here used to claim it says *"Some call me nature, others
call me mother nature."* It does not. Whisper transcribes it as *"Today, I'm so happy. So, today is the
first day."*, and the English-only model refuses it outright as *"(speaking in foreign language)"*.

Given a transcript containing words the audio does not, ZipVoice **speaks those words** at the start of
what it generates. That was the preamble on every render in this project — "Others call me Mother
Nature" ahead of the actual sentence — and it cost roughly six points of word error to both frontends
equally:

| | packaged clip | this clip |
|---|---|---|
| reference frontend (espeak-ng) | 9.1% | 6.7% |
| our MIT phonemizer | 7.2% | 4.7% |
| preamble | ~2.3 seconds of it | none |

## Replacing this clip was MEASURED AND REJECTED (2026-08-29)

An open item claimed this clip "has room tone" and that a studio clip measured 2.3% where this one
measures 4.7%. **Both halves of that failed to survive measurement. Do not re-baseline the fixtures on
it.**

**It is not a noisy clip.** Measured with 20 ms frames, comparing the loud decile against the quiet decile:

| clip | length | noise floor | SNR |
|---|---|---|---|
| **this one** | 4.0 s | -55.9 dBFS | **37.2 dB** |
| LJ Speech LJ001-0009 | 7.6 s | -55.9 dBFS | 40.2 dB |
| LJ Speech LJ001-0084 | 4.2 s | -57.8 dBFS | 39.5 dB |
| LJ Speech LJ001-0117 | 6.6 s | -56.5 dBFS | 38.3 dB |

It sits in the same band as a purpose-recorded TTS corpus. (It is also moot: `ComputePromptFeatures`
boosts a quiet reference to `TargetRms` before the mel, so absolute level cannot matter much anyway.)

**And swapping it changes nothing measurable.** Two LJ Speech clips - one length-matched at 4.2 s, one
long at 7.6 s - were built into parallel fixture sets over the same 9 phase1 sentences and rendered at
three seeds, paired against this clip on the same sentence and seed:

| | mean word error, 27 pairs |
|---|---|
| this clip | 4.57% |
| LJ001-0084 (4.2 s) | 3.96% |
| LJ001-0009 (7.6 s) | 4.44% |

Every pairwise gap (0.13-0.61%) is **below the set's own 1.70-1.95% resolution**, and **20-22 of the 27
pairs transcribe IDENTICALLY**. The claimed 2.4-point win would have been obvious here; it is not there.

⚠️ **The direction flipped with the seed.** At one seed LJ001-0084 looked WORSE (4.62% vs 4.16%); at three
seeds it looked better (3.96% vs 4.57%). Neither is a result - both are noise, and a single-seed
comparison of reference clips is worthless. This is the same trap that once made a phonemizer rebuild look
like a win.

Resolving a sub-1% difference would need roughly 40x the renders (resolution improves as sqrt(n)). There is
no reason to believe a real effect is hiding there, so this stays.

**If someone still wants to replace it,** LJ Speech is the best-licensed candidate found: genuinely public
domain (texts 1884-1964, audio recorded 2016-17 by LibriVox, no restrictions), and every clip ships an
exact transcript, which removes the wrong-transcript failure below by construction. Individual wavs and
`metadata.csv` are fetchable per-file from the `flexthink/ljspeech` HuggingFace repo - no 2.6 GB download.
⛔ LibriTTS and LibriTTS-R are **CC BY 4.0, not CC0** - usable with attribution, but not public domain.
⛔ Government/NASA archive audio is public domain but radio-band-limited and compressed - acoustically the
wrong direction for a reference clip.

## Choosing another clip

Two properties matter, in this order:

1. **You must know exactly what it says.** Not approximately. A transcript that omits or adds a word
   makes the model speak the difference.
2. **Licensing you can point at.** The Public Domain Mark or an explicit dedication. "Freely usable" is
   not a licence — that is why the Open Speech Repository recordings, which were the first candidate, are
   not used here.

Then build fixtures against it:

```
dotnet run --project tools/zipvoice-fixture -c Release -- --file sentences.txt --out <dir> \
    --prompt-wav <your.wav> --prompt-text "<exactly what it says>"
```

The prompt's token ids are derived as the common prefix of two probe runs, so nothing has to be assumed
about how many debug lines the reference frontend splits it into.
