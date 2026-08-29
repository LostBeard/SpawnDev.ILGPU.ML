# SpawnDev.Phonemizer Changelog

## 1.0.0 (2026-08-28)

First release. MIT English grapheme-to-phoneme (text to phoneme) with **no dependencies and no native
binaries** - a permissive replacement for GPL-3 espeak-ng as a text-to-speech frontend.

Every model worth running in the open TTS ecosystem - ZipVoice, Piper, Kokoro - phonemizes through
espeak-ng, which is GPL-3. That single dependency is why .NET has had no cleanly licensed, browser-capable
English TTS frontend. This is that frontend.

### What it does

- **CMUdict lookup** for the 126,052 words it knows, with homograph resolution ("record" the noun against
  the verb, which differ in vowel as well as stress).
- **Morphological decomposition** for unknown words that are a known stem plus an ending, with the
  allomorphy English actually demands: the -s of "cats" is an S, of "dogs" a Z, of "boxes" a syllable. It
  finds the stems spelling hides - "hoped" is hope+d, "running" is run+ning.
- **Letter-to-sound** for everything else, LEARNED from CMUdict and measured on words held out of
  training. A hand-written ruleset encodes one person's recollection of English spelling and can never
  honestly claim an accuracy figure.
- Output as **ARPAbet or IPA**, with stress, punctuation written the way these models were trained on.
- **Text normalization** (numbers, currency, ordinals, abbreviations) so real text can be spoken.

### Measured, on words held out before training

| | |
|---|---|
| letter-to-sound alone | **49.9%** of unknown words exactly right, 13.8% phoneme error |
| with decomposition first | **53.1%** |
| decomposition | fires on 26.4% of unknown words, right 77.2% of the time |
| CMUdict coverage | 99.5% of the top 1,000 English words |

**End to end as AUDIO**, which is the number that actually matters: rendered through ZipVoice from the same
voice and the same noise seed, ours measures **9.69% word error against espeak-ng's 11.30%** on sentences
carrying names - and **4.7% against 6.7%** on read-aloud declaratives. Parity or better with the GPL tool
it replaces, which is what removing that dependency required.

### Deliberately

- **Zero package dependencies.** The dictionary (915 KB) and the learned rules (482 KB) are embedded in the
  assembly, gzipped, and expanded on first use. A phonemizer that needs two data files fetched from
  somewhere is not really dependency-free, and in a browser it is one more thing to host and version.
- **Browser-capable.** Nothing outside what WASM already carries.

### Attribution

The CMU Pronouncing Dictionary is BSD-2-Clause, and the letter-to-sound model is derived from it, so it
carries the same notice. See `THIRD-PARTY-NOTICES.md`, which ships inside this package.
