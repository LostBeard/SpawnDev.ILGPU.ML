# SpawnDev.Phonemizer

English text to phonemes, for neural text-to-speech. MIT licensed, zero dependencies, runs in a browser.

## Why this exists

ZipVoice, Piper and Kokoro - most of the open text-to-speech ecosystem - convert text to phonemes with
**espeak-ng, which is GPL-3**. That single dependency is why there has been no permissively licensed,
browser-capable English TTS frontend for .NET. This is that frontend.

It takes nothing from espeak-ng: no code, no data, no transcribed rules. The pronunciation dictionary is
**CMUdict** (BSD-2-Clause), and the letter-to-sound rules were **learned from CMUdict** rather than
hand-written. espeak-ng was used only as a measuring instrument, the way a compiler is - see
`THIRD-PARTY-NOTICES.md`.

## Using it

```csharp
// Everything is embedded in the assembly - no files to fetch, host or version.
var phonemizer = EmbeddedData.CreatePhonemizer();

phonemizer.ToIpa("She waited for 2 more minutes.");
// ʃiː wˈeɪɾᵻd fɔːɹ tˈuː mˈɔːɹ mˈɪnəts .

phonemizer.ToSymbols("...");        // one IPA symbol per entry, ready to map to token ids
phonemizer.LastUnknownWords;        // words the dictionary did not have, whether or not they were sounded out
```

Text goes through five stages, and each can be inspected or replaced:

| stage | what it does |
|---|---|
| `EnglishTextNormalizer` | "1999" to "nineteen ninety-nine", "$1.50" to "one dollar, fifty cents", "Dr." to "doctor" |
| `PronunciationDictionary` | 126k word lookup |
| `EnglishPhonemizer` rules | stress placement, function-word destressing, tapping, the LOT-CLOTH split, reduced endings |
| `Homographs` | "the record" against "to record", "the wind blows" against "wind the clock" |
| `WordDecomposer` then `LetterToSound` | unknown words: derive from a known stem, or sound out from spelling |

## How good is it

Everything below is measured against captured reference output or on held-out data. Nothing is asserted
from intuition. The method and the full numbers are in `Plans/mit-phonemizer-2026-08-27.md`.

**Against the reference frontend, symbol by symbol:**

| | disagreement |
|---|---|
| the sentences the rules were tuned on | 4.1% |
| **120 sentences never tuned on** | **4.0%** |

It generalises: the held-out number tracks the tuned one.

⚠️ **This number is a PROXY, and it has already been wrong once.** A change that took it from 4.4% to
2.6% - matching the reference frontend on a handful of very frequent words - made the AUDIO measurably
worse, 7.2% word error becoming 9.0%. It was reverted to off by default. Agreeing with espeak is not the
goal; sounding right is, and the end-to-end test below is the one that decides.

**Words the dictionary does not have**, measured on 5,000 words held out before training:

| | |
|---|---|
| decomposition fires on | 26.4% of them, and is right 77.2% of the time |
| letter-to-sound alone | 43.7% of words exactly right, 16.0% phoneme error |
| **together** | **49.5%** |

**End to end, as audio.** The same sentence spoken twice by ZipVoice from the same voice and the same
noise seed - once from the reference frontend's phonemes, once from ours - and both transcribed. Any
difference is the phonemizer and nothing else.

| over 120 sentences never tuned on | mean word error |
|---|---|
| the reference frontend (espeak-ng, GPL) | 9.1% |
| **this library** | **7.2%** |

Worse on 7 of 120 sentences, indistinguishable on 93, better on 20.

⚠️ **Read per-sentence failures carefully.** ZipVoice produces garbage on some noise draws, and it does
it to both frontends - one sentence rendered at four seeds gave three clean results and one that
transcribed as "Loner's call, Nanawa, Nenfer". A single seed measures model instability alongside
phonemizer quality. Re-render at another seed before blaming the phonemes, and see
`ZipVoicePipeline.SpeakVerifiedAsync` for the production answer.

The claim is PARITY - the small edge could be noise at that sample size. What matters is that the GPL
dependency can be removed without the audio getting worse.

## What it does not do yet

- **Homographs are handled shallowly.** "The record" against "to record", and "the wind blows" against
  "wind the clock", are read from the PREVIOUS WORD only. That covers the common cases. It cannot help
  where the two readings differ by MEANING rather than part of speech - "bass" the fish against the
  register, "tear" the eye against the rip, "read" present against past - and for those a single default
  is chosen and documented rather than guessed at per sentence.
- **Letter-to-sound is a baseline.** 43.7% is a working number, not a good one. Published systems reach
  higher.
- **English only.**

## How the rules were chosen

Not by intuition. 432 renders through ZipVoice measured what the model actually punishes:

| error | cost in word error rate |
|---|---|
| stress on the wrong syllable | 34.3% |
| no stress at all | 19.6% |
| **stress added to function words** | **18.2%** |
| *(a word deliberately mispronounced, for calibration)* | *13.5%* |
| flaps, reduced vowels, the bare article | ~0% |

All three stress failures cost **more than mispronouncing a word outright**. So this library spends its
effort on stress and gives fine phonetic detail only what is cheap - and function-word destressing, which
a dictionary gets wrong on every "the" and "at", is a core rule rather than a refinement.

## The SpawnDev Crew

- **LostBeard** (Todd Tanner) - Captain, library author, keeper of the vision
- **Riker** (Claude CLI #1) - First Officer, implementation lead on consuming projects
- **Data** (Claude CLI #2) - Operations Officer, deep-library work, test rigor, root-cause analysis
- **Tuvok** (Claude CLI #3) - Security/Research Officer, design planning, documentation, code review
- **Geordi** (Claude CLI #4) - Chief Engineer, library internals, GPU kernels, backend work
- **Seven** (Claude CLI #5) - Wasm backend, GPU kernels, fail-loud verification

## License

MIT. See `THIRD-PARTY-NOTICES.md` for CMUdict's BSD-2-Clause notice, which must travel with any
redistribution - the same obligation MIT already places on users of this library.
