# SpawnDev.Phonemizer Changelog

## Unreleased

### Added

- **`Define` / `Remove` on `PronunciationDictionary`, and `Define` on `EnglishPhonemizer`** - teach it how
  a word is said at runtime, without rebuilding the embedded data. The words an application says most are
  the ones CMUdict lacks (names, brands, jargon), and letter-to-sound is right about half the time on
  those; a word you KNOW should never be guessed at. Aubriella goes from `ˈɔːbɹiɛlə` to `ˌɔːbɹiˈɛlə`,
  which is where every other "-ella" name in the dictionary puts its stress.
  - Replaces rather than appends, so a definition is authoritative: `Homographs` only chooses between
    alternates when two or more exist, and leaving the originals would let context pick one of them.
  - Sits ahead of decomposition and letter-to-sound, so a defined word is never sounded out.
  - Phones are validated against the same tables that convert them - an unknown phone, or a vowel with no
    stress digit, throws naming the offender instead of emitting a wrong sound.
  - `EnglishPhonemizer.Dictionary` is now exposed.
- Behaviour is unchanged for anyone who defines nothing: the accuracy probe reads an identical
  24 differences over 592 reference symbols (4.1%) before and after.

### Fixed

- Three missing XML `param` tags (CS1573). The package builds with zero warnings.

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
voice and the same noise seed, ours measures **9.7% word error against espeak-ng's 11.3%** on sentences
carrying names, and **4.7% against 6.7%** on read-aloud declaratives.

On the out-of-vocabulary set - every sentence carrying a name the dictionary lacks, which is this library's
hardest path - **12.1% against espeak-ng's 13.5%** over 120 paired renders.

That one is established rather than merely observed. At the original 60 renders the paired standard
deviation was 7.6%, so the set resolved only ~1.9% at 95% confidence and a 1.6% gap could not be told from
no difference at all. Doubling the sentences took the resolution to **1.3%** against a **1.4%** gap, and
the effect **replicated** on the 20 fresh sentences (-1.6% -> -1.4%). `tools/zipvoice-harness endtoend` now
prints its own resolution beside every result, so a difference smaller than the set can see is visible as
such instead of being read as a win.

### Deliberately

- **Zero package dependencies.** The dictionary (915 KB) and the learned rules (482 KB) are embedded in the
  assembly, gzipped, and expanded on first use. A phonemizer that needs two data files fetched from
  somewhere is not really dependency-free, and in a browser it is one more thing to host and version.
- **Browser-capable.** Nothing outside what WASM already carries.

### Attribution

The CMU Pronouncing Dictionary is BSD-2-Clause, and the letter-to-sound model is derived from it, so it
carries the same notice. See `THIRD-PARTY-NOTICES.md`, which ships inside this package.
