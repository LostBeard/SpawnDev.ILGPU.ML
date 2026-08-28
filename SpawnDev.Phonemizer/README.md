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
var dictionary = PronunciationDictionary.Load("cmudict.dict");
var phonemizer = new EnglishPhonemizer(dictionary)
{
    LetterToSound = LetterToSound.Load("lts-model.txt"),   // optional: pronounce unknown words
};

phonemizer.ToIpa("She waited for 2 more minutes.");
// ʃiː wˈeɪɾᵻd fɔːɹ tˈuː mˈɔːɹ mˈɪnəts .

phonemizer.ToSymbols("...");        // one IPA symbol per entry, ready to map to token ids
phonemizer.LastUnknownWords;        // words the dictionary did not have, whether or not they were sounded out
```

Text goes through four stages, and each can be inspected or replaced:

| stage | what it does |
|---|---|
| `EnglishTextNormalizer` | "1999" to "nineteen ninety-nine", "$1.50" to "one dollar, fifty cents", "Dr." to "doctor" |
| `PronunciationDictionary` | 126k word lookup |
| `EnglishPhonemizer` rules | stress placement, function-word destressing, tapping, the LOT-CLOTH split, reduced endings |
| `WordDecomposer` then `LetterToSound` | unknown words: derive from a known stem, or sound out from spelling |

## How good is it

Everything below is measured against captured reference output or on held-out data. Nothing is asserted
from intuition. The method and the full numbers are in `Plans/mit-phonemizer-2026-08-27.md`.

**Against the reference frontend, symbol by symbol:**

| | disagreement |
|---|---|
| the sentences the rules were tuned on | 4.2% |
| **120 sentences never tuned on** | **4.5%** |

It generalises: the held-out number tracks the tuned one.

**Words the dictionary does not have**, measured on 5,000 words held out before training:

| | |
|---|---|
| decomposition fires on | 26.4% of them, and is right 77.2% of the time |
| letter-to-sound alone | 43.7% of words exactly right, 16.0% phoneme error |
| **together** | **49.5%** |

**End to end, as audio.** The same sentence spoken twice by ZipVoice from the same voice and the same
noise seed - once from the reference frontend's phonemes, once from ours - and both transcribed. Any
difference is the phonemizer and nothing else.

## What it does not do yet

- **Homographs.** "Record" the noun and "record" the verb are stressed differently; this always picks the
  dictionary's first pronunciation. That matters because stress is what TTS models punish hardest.
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
