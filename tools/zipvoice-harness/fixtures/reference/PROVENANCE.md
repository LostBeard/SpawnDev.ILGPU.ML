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
