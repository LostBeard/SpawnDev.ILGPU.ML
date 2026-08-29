# SpawnDev.Phonemizer Changelog

## 1.1.0 (2026-08-29)

### Added

- **`Define` / `Remove` on `PronunciationDictionary`, and `Define` on `EnglishPhonemizer`** - teach it how
  a word is said at runtime, without rebuilding the embedded data. The words an application says most are
  the ones CMUdict lacks (names, brands, jargon), and letter-to-sound is right about half the time on
  those; a word you KNOW should never be guessed at. Aubriella goes from `ËÉËbÉ¹iÉlÉ` to `ËÉËbÉ¹iËÉlÉ`,
  which is where every other "-ella" name in the dictionary puts its stress.
  - Replaces rather than appends, so a definition is authoritative: `Homographs` only chooses between
    alternates when two or more exist, and leaving the originals would let context pick one of them.
  - Sits ahead of decomposition and letter-to-sound, so a defined word is never sounded out.
  - Phones are validated against the same tables that convert them - an unknown phone, or a vowel with no
    stress digit, throws naming the offender instead of emitting a wrong sound.
  - `EnglishPhonemizer.Dictionary` is now exposed.
- Behaviour is unchanged for anyone who defines nothing: the accuracy probe reads an identical
  24 differences over 592 reference symbols (4.1%) before and after.

- **`PhonemeVocabulary`** - maps phoneme symbols to a model's token ids and back, the last mile between
  `ToSymbols` and a neural input tensor. The same parsing loop had been written out in four of this
  repository's own tools, and every copy is a chance to get the same detail wrong: **a symbol can be
  whitespace** (a ZipVoice vocabulary really does list a space as a token), so the file must be split on
  its LAST tab. Splitting the obvious way silently drops the token that separates words. `Encode` throws
  and names a symbol the model has no token for, rather than dropping a sound and leaving nothing to
  explain the gap; `TryEncode` is there for callers that would rather decide.
  ℹ `ZipVoiceTokenizer.LoadSymbolTable` is NOT yet collapsed onto this - it is the correct copy the type
  was extracted FROM, and rewriting library code after the release sweep had started would have meant
  shipping something the gate never ran. It is behaviourally identical; folding it in is the next change.

- **Acronyms are spelled out instead of guessed at.** The dictionary's own notes said its misses are
  "almost entirely abbreviations and acronyms... which want expanding or spelling out rather than
  guessing" - and nothing did it, so "RSS" went to letter-to-sound and came back as an invented word. An
  ALL-CAPS token the dictionary does not have is now read out as its letters.
  - The filter does the work: only a word the DICTIONARY LACKS is spelled. That leaves "NASA" alone
    (it is in there as a word), and the exception list maintains itself.
  - Letter names come from the dictionary rather than a new table - it already holds them ("r" is AA1 R,
    "w" is D AH1 B AH0 L Y UW0). â ï¸ Except "a", whose first entry is the ARTICLE; its letter name is the
    alternate.
  - Stress follows CMUdict's own treatment of the acronyms it DOES hold - secondary on every letter but
    the last - which is verified against five of them (html, url, api, dvd, pdf, cpu).

- **Possessives resolve their STEM instead of being guessed whole.** "Aubriella's" is a different string
  from "Aubriella", so it missed the dictionary and went to letter-to-sound as one long unknown word -
  which defeated `Define` outright, since you could teach it a name and still have the possessive
  guessed. The stem now goes back through the same order (dictionary, acronym, decomposition, guessing)
  and the ending is the regular English rule: a reduced vowel plus /z/ after a sibilant, /s/ after a
  voiceless consonant, /z/ otherwise. A plural possessive ("the dogs' bowls") adds an apostrophe on the
  page and no sound. ð´ This also removed an embarrassment: "FAQ's" was being guessed as an obscenity.

### Fixed

- **A grouped number was read as a year.** "I have 1,234 of them" came out as "twelve thirty-four": the
  commas were stripped before the year heuristic ran, so a quantity arrived indistinguishable from a
  year - and the comma, the one thing that tells them apart, was what got discarded. Grouped numbers are
  now expanded where the grouping is still visible. â ï¸ Round ones keep the year-style reading, because
  that is what English says: "fifteen hundred apples", never "one thousand five hundred apples".
  Roundness is the discriminator, not the comma - reading every grouped number as a cardinal broke that,
  and the existing test caught it.
- **A clock time left its colon in.** "3:30" reached the phonemizer as "three : thirty", where the colon
  is punctuation and is spoken as a pause. Times now read "three thirty", "nine oh five", "two o'clock".
- **"St." was always a saint.** "123 Main St." was read as "Main saint". A saint's name FOLLOWS the
  abbreviation and a street's PRECEDES it, so what comes before now decides. The sentence-ending period
  is preserved, since the phonemizer reads punctuation as prosody.
- **`&` and `#` were silent.** They reached the phonemizer as punctuation, so "Mr. & Mrs." simply lost
  the "and". Now "and" and "number".
- **An abbreviation that is also an ordinary word no longer fires without its period.** "co-op" was read
  as "COMPANY op" - a hyphen is a word boundary, and `co` was matched on a boundary alone. The same route
  turns "rev the engine" into "reverend the engine", and "gen"/"hon" are a generation and a term of
  endearment as often as a general and an honorable. Those four now require their full stop, and are
  expanded BEFORE the step that strips it.
- Three missing XML `param` tags (CS1573), and an ambiguous `cref`. The package builds with zero warnings.

- **Units were left as letters, and "ft" was read as "fort".** "5km" was spoken "five km"; "6 ft tall"
  came out "six fort tall" because the abbreviation table owns "ft". Units now expand when a NUMBER
  precedes them, agreeing in number ("1 km" singular, "1.5 km" plural). Requiring the number is what
  makes it safe: single letters that are ordinary English words ("in", "m", "s") are deliberately not
  units at all, so "I live in Ohio" is untouched. Every expansion was checked against the dictionary
  first - only "gigahertz" was missing, so it is not offered.
- **Arithmetic, a leading minus, and word/number hyphens.** "5 + 3 = 8" reads "plus"/"equals" (only
  BETWEEN numbers - elsewhere they are punctuation), "-5" reads "minus five", and "COVID-19" no longer
  carries a pause through the middle of one word. An ordinary hyphenated word and a dash between
  clauses are untouched.

### Known gaps (measured, not fixed)

URLs are not spoken ("www.example.com"), and an attributive unit keeps its plural ("a 500 mb file" reads
"megabytes file").

â **Joining hyphenated words was measured and rejected.** "re-read" reads as "ray read" because CMUdict's
entry for "re" is the musical note, and the joined form "reread" is correct - but the same rule turns
"co-op" into "kuËp", a chicken coop. Every part of all 15 hyphenated words tested is already IN the
dictionary, so this is not a guessing failure at all: it is one prefix whose dictionary entry is the wrong
sense, the same shape as "a" being the article rather than the letter name. A blanket join loses more than
it wins.

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
