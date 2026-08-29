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

### Teaching it a word

The words an application says most often are exactly the ones a general dictionary lacks - character
names, brands, jargon, people. Letter-to-sound guesses those, and it is right about half the time. When
you know how a word is said, say so:

```csharp
phonemizer.Define("Aubriella", "AO2 B R IY0 EH1 L AH0");   // ARPAbet, as CMUdict writes it
```

| | |
|---|---|
| before | `ˈɔːbɹiɛlə` - stress on the "au", because that is where the rules put it |
| after | `ˌɔːbɹiˈɛlə` - stress on the "ell", like every other "-ella" name |

A definition **replaces** anything held for that word and sits ahead of both decomposition and
letter-to-sound, so a defined word is never guessed at and never re-resolved by homograph context.
`Remove` puts it back. Phones are validated on the way in - an unrecognised phone, or a vowel missing
its stress digit, throws rather than travelling silently into the output as a wrong sound.

Define words at setup: the dictionary is a plain map with no locking.

### Turning symbols into token ids

A model wants integers, not symbols:

```csharp
var vocabulary = PhonemeVocabulary.Load("tokens.txt");
long[] ids = vocabulary.Encode(phonemizer.ToSymbols("Hello there."));
```

⚠️ **A symbol can be whitespace** - a ZipVoice vocabulary really does list a space as a token, and it is
the one that separates words - so the file is split on its LAST tab rather than on whitespace. `Encode`
throws and names a symbol the model has no token for, because dropping it renders the sentence missing a
sound with nothing to explain the gap; `TryEncode` is there when you would rather decide yourself, and
`Decode` reads ids back for debugging.

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
| letter-to-sound alone | 49.9% of words exactly right, 13.8% phoneme error |
| **together** | **53.1%** |

**End to end, as audio.** The same sentence spoken twice by ZipVoice from the same voice and the same
noise seed - once from the reference frontend's phonemes, once from ours - and both transcribed. Any
difference is the phonemizer and nothing else.

| sentences never tuned on | reference (espeak-ng, GPL) | **this library** |
|---|---|---|
| 120 sentences, one noise seed | 9.1% | **7.2%** |
| 40 sentences, three noise seeds | 10.8% | **7.3%** |
| 30 sentences, **properly paired reference clip** | 3.1% | **2.3%** |
| **80 sentences x 3 seeds, all carrying a NAME the dictionary lacks** | 15.0% | **13.0%** |

That last row is the one that exercises this library's hardest path - letter-to-sound, on words no
dictionary contains - and it is the only row whose difference has been shown to be **larger than the
measurement could have produced by chance**: over its 240 paired renders the set resolves about 1.23% at
95% confidence, and the gap is 2.0%. Ours is better on 58 renders, worse on 25, and indistinguishable on
the remaining 157.

It has now replicated twice. At 40 sentences the same comparison read 13.5% against 12.1%, a gap of 1.4%
against a resolution of 1.27% - true, but only barely resolvable. Doubling the set to 80, with 40 fresh
sentences written afterwards, moved the gap to 2.0% and left it comfortably clear of the noise. ⚠️ The
ABSOLUTE numbers rose because the added sentences are harder; the GAP is the stable quantity, which is
the whole reason this is measured paired.

⚠️ Read the other rows as point estimates. They are real measurements on real audio, but a difference
smaller than a set can resolve is not evidence of a difference - `tools/zipvoice-harness endtoend` now
prints its own resolution beside every result so this is visible rather than assumed.

That last row is the one to look at for absolute quality. The packaged sample clip is paired with a
transcript that is not what it says, so the model speaks those words at the start of every render - worth
about six points of word error to BOTH frontends. Give ZipVoice a reference clip you have an accurate
transcript for.

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
- **Letter-to-sound is a baseline.** 49.9% is a working number, not a good one. Published systems reach
  higher.
- **English only.**

## How the rules were chosen

Not by intuition. 432 renders through ZipVoice measured what the model actually punishes:

| error | word error added | how far it moved the audio |
|---|---|---|
| stress on the wrong syllable | **17.1%** | 0.89 |
| *(a word deliberately mispronounced, for calibration)* | *12.6%* | *0.51* |
| length marks dropped | 7.8% | 0.75 |
| no stress at all | 5.2% | 0.95 |
| stress added to function words | 2.8% | 0.75 |
| flaps, reduced vowels, the bare article | ~0% | 0.29-0.50 |

Stress on the wrong syllable is the only failure that costs more than mispronouncing a word outright, and
the stress classes move the AUDIO furthest even where the words survive. So this library spends its
effort on stress and gives fine phonetic detail only what is cheap.

⚠️ An earlier run of this study, through a reference clip whose transcript was wrong, put function-word
stress at 18.2% and made it the headline. On a properly paired clip it is 2.8%. The rule is still applied
- it is correct English and free - but the number is corrected here rather than left overstating it.

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
