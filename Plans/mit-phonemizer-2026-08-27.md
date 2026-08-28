# MIT English phonemizer (grapheme to phoneme) - plan and progress tracker

Tuvok, started 2026-08-27. **This is the living tracking doc.** Update the status markers and the
"Session log" at the bottom as work lands, so any session can resume from this file alone.

Status key: `[ ]` not started, `[~]` in progress, `[x]` done and verified, `[!]` blocked.

---

## Why this exists

ZipVoice cannot speak English on our stack. The shipped `lexicon.txt` is Chinese-only (68,037 CJK
entries, ZERO English), so every English word reaches the model through espeak-ng grapheme to phoneme.
espeak-ng is GPL-3. TJ's standing decision: **our code is MIT; permissively licensed third-party data
with attribution is acceptable; GPL is not.** So espeak-ng cannot be ported, linked, or shipped.

The gap is bigger than ZipVoice. Piper, Kokoro, and most of the open TTS ecosystem phonemize through
espeak-ng, which is why there is no clean, permissively licensed, browser-capable English TTS frontend
for .NET. **The deliverable is that frontend.** ZipVoice is its first consumer and its test rig.

## The target, stated precisely

NOT "reproduce espeak". The target is **phoneme sequences that ZipVoice renders correctly**.

Exact token agreement with espeak is sufficient but not necessary. Where our sequence differs on a
token the model treats as interchangeable, the audio is fine and the disagreement is only noise in the
metric. Where it differs on a token the model is sensitive to, the audio is wrong. Which differences
fall in which bucket is the single most important unknown, and Phase 1 measures it.

---

## Verified facts (do not re-derive)

### The model was trained on espeak en-us output

`k2-fsa/ZipVoice` (Apache-2.0), `zipvoice/tokenizer/tokenizer.py`, `EmiliaTokenizer.tokenize_EN`:

```python
text = self.english_normalizer.normalize(text)
tokens = phonemize_espeak(text, "en-us")   # piper_phonemize -> espeak-ng
tokens = reduce(lambda x, y: x + y, tokens)
```

So the token stream carries espeak's quirks: flapping, the reduced vowels, and explicit stress marks.
This is the unfavourable case, and it is why Phase 1 comes before any frontend code.

### The token inventory is fixed and small

`tokens.txt` in the model dir, 360 lines. Ids 0-158 are the espeak/piper IPA symbol set plus
punctuation; 159+ are pinyin syllables (tone3 style, initials suffixed with a zero) and are not on the
English path. Relevant ids: primary stress = 120, secondary stress = 121, length mark = 122, the
barred-i reduced vowel = 128, space = 3.

### Text normalization is already permissive and portable

`zipvoice/tokenizer/normalizer.py` `EnglishTextNormalizer` is Apache-2.0 and uses `inflect` (MIT).
Covers commas in numbers, dollars, fractions, decimals, percent, ordinals, cardinals, abbreviations.
`EmiliaTokenizer.map_punctuations` normalizes CJK punctuation to ASCII and three dots to the ellipsis
character. Both port to C# with no licensing question.

### Licenses, fetched not recalled (2026-08-27)

- **CMUdict**: BSD-2-Clause (`cmusphinx/cmudict/LICENSE`). Obligation is retain copyright notice plus
  disclaimer, in source and in binary distribution documentation. Same shape as MIT's obligation.
  Compatible with shipping an MIT-licensed library. **Attribution goes in `THIRD-PARTY-NOTICES.md`.**
- **Flite**: BSD-like with an added no-endorsement clause; its `COPYING` states GPL code appears only
  in the build process, and that a few files carry different licenses.
  **[ ] Before any Flite LTS data enters the tree, verify the license header of those specific files.**
- **espeak-ng**: GPL-3. Usable as a measuring instrument only (same standing as a compiler). No code,
  no data, and no derived rules may enter our tree. Our ground truth is read through sherpa-onnx,
  which is already an external process in the gate loop.

### Tooling built for this work (all in the repo, all reusable)

- **`tools/zipvoice-fixture`** - sentences in, ground-truth fixtures out. Drives the oracle as a child
  process (sherpa logs from native code straight to fd 2, which `Console.SetError` cannot intercept),
  parses the ids, and peels the fixed prompt off the front **by matching ids against a known-good
  fixture** rather than counting debug lines. Fails loudly if the split does not line up, because a
  silently mis-split capture would poison every measurement built on it.
  `dotnet run --project tools/zipvoice-fixture -c Release -- --file tools/zipvoice-fixture/sentences-phase1.txt --out tools/zipvoice-harness/fixtures/phase1`
- **`tools/zipvoice-harness sensitivity`** - the perturbation experiment. Paired, replicated, resumable,
  with a positive control. Writes `results.json` after every graded clip so a long run never loses work.
  `dotnet run --project tools/zipvoice-harness -c Release -- sensitivity fixtures/phase1 <outDir>`
  Env: `SENSITIVITY_SEEDS=a,b,c`, `SENSITIVITY_ONLY=<variant>`, `SENSITIVITY_NO_GRADE=1`,
  `SENSITIVITY_REGRADE=1`, `ZIPVOICE_TAIL_PAD=<sec>`, `ZIPVOICE_INT8=1`, `WHISPER_MODEL_DIR=<dir>`.
- **`tools/zipvoice-g2p-probe`** - the loop-closer. Builds what CMUdict actually produces and aligns it
  WORD BY WORD against what espeak actually produced, then classifies every difference, flagging classes
  that have no perturbation testing them as `UNTESTED-*`. Run it whenever the mapping changes; anything
  it reports as UNTESTED is an unmeasured risk.
  `dotnet run --project tools/zipvoice-g2p-probe -c Release`
- **`tools/zipvoice-listen`** - builds a self-contained HTML listening page from a run: every clip with
  what was changed, what to listen for in plain English, and what the grader heard. Audio embedded as
  data URIs, so it is one file that works anywhere. Exists because the person judging the audio should
  not have to already know what a flap is.
  `dotnet run --project tools/zipvoice-listen -c Release -- <resultsDir> [out.html] [--fixture n] [--seed n]`

### What makes the measurement trustworthy rather than an anecdote

- **Paired.** Every perturbation is scored against the control rendered from the SAME sentence with the
  SAME noise seed, so sentence difficulty and the noise draw cancel instead of acting as confounds.
- **Replicated.** Nine sentences, three noise seeds. Flow matching starts from fresh noise every call,
  so one render is a sample, not a measurement.
- **Positive control.** `wrong-vowel-last-word` deliberately mispronounces one word. If that row does
  not show damage, the grader cannot see damage at all and every clean row is meaningless - the tool
  declares the run VOID rather than reporting a comfortable result.
- **No-op rows excluded.** A sentence with no flap in it cannot say anything about flaps. Those rows
  have zero token edits and are dropped from the aggregate instead of diluting it toward zero.

### Threats to validity that REMAIN (read before trusting any number here)

- **Whisper repairs damage.** It is a strong language model and will happily correct a mispronounced
  word back to the one the sentence made likely. Its bias is therefore toward UNDER-reporting damage,
  which makes the model look more tolerant than it is. **Answered two ways**: `tools/zipvoice-listen`
  puts a human ear on it, and `AcousticDistance.cs` adds a language-free axis - DTW-aligned log-mel
  distance from the paired control, reported as a multiple of the NOISE FLOOR (the distance between two
  control renders of the same sentence at different seeds, which is what flow matching does on its own).
  Read together they separate three cases WER cannot: low WER and ~1x sound means the change was
  genuinely ignored; low WER and well above 1x means it sounds different but stays intelligible, which
  is accent and rhythm damage; high WER means the words themselves broke.
- **The perturbations are my MODEL of CMUdict's errors, not CMUdict's actual output.** They were chosen
  from what ARPAbet cannot express. **Phase 4 must close this loop**: once real CMUdict-derived
  sequences exist, diff them against the oracle ids, enumerate the error classes that ACTUALLY occur,
  and confirm they are the ones measured here. Any class that shows up in the diff and not in this
  table has never been tested.
- **One voice.** All renders use the same reference clip. Voice-dependent effects would not show up.
- **WER is word-level.** It cannot see an accent, an odd rhythm, or a mechanical delivery.
- **The graders cannot see disfluency at all.** Whisper tidies stutters away and the acoustic metric is
  relative to a control that may itself stutter. Confirmed by ear, then measured at a floor of 3.2% of
  renders. A human listening pass is not optional at any phase.

### Instruments that already exist in the repo

- `tools/zipvoice-oracle` - sherpa-onnx with `Debug=1` prints the per-word and per-sentence token ids
  espeak produced. Ground truth for token sequences.
  `dotnet run --project tools/zipvoice-oracle -c Release -- "text" out.wav 2> debug.log`
  then `grep "new sentence" debug.log`.
- `tools/zipvoice-harness synth <fixture.json>` - synthesises from **arbitrary token ids** through
  `ZipVoicePipeline` on onnxruntime, with `NoiseSeed = 1234` pinned. This is what makes a controlled
  perturbation experiment possible: token ids are the only variable between runs.
- `tools/zipvoice-harness/fixtures/paint-the-sockets.json` - a captured ground-truth token set.
- `../tools/whisper-harness` (outer `tools/`, NOT in any git repo - worth fixing) transcribes any wav:
  `dotnet run --project tools/whisper-harness -c Release -- <file.wav>`. This is the audio grader.

---

## Phases

### Phase 1 - Measure the model's sensitivity BEFORE building anything `[x]` REPLICATED

Perturb the oracle's own correct token stream in exactly the ways a CMUdict-based mapping will get it
wrong, synthesise each, and grade. Establishes how precise the mapping has to be, per error class.

Perturbations (each applied to the ground-truth sequence, one at a time):

- [x] flap reversal: the alveolar tap back to t and d
- [x] reduced vowel: barred-i to small-capital-i
- [x] reduced vowel: turned-a to schwa
- [x] rhotic: r-coloured schwa to schwa plus r
- [x] secondary stress dropped
- [x] all stress marks dropped
- [x] primary stress moved one vowel later
- [x] length mark dropped
- [x] control: the unmodified sequence (proves the rig, not the perturbation, is what is being read)

Grade each by Whisper transcript against the known text, plus a listen. Deliverable is a table of
error class against damage.

#### RESULT (replicated), 2026-08-27

Nine sentences x three noise seeds x sixteen variants = 432 renders, graded by **whisper-base.en** and
by acoustic distance. Superseded the first single-sentence run, which is kept below only as a lesson.

```
dotnet run --project tools/zipvoice-harness -c Release -- sensitivity fixtures/phase1 <outDir>
SENSITIVITY_SEEDS=1234,20260827,31337   WHISPER_MODEL_DIR=<...>/whisper-base.en
```

Control floor **8.6% mean** over 27 clean renders. **One cell excluded** by a rule declared in advance -
`about-a-hundred-packages` seed 31337, whose control transcribed as *"I'm not a human, I'm a human."*
That is the TTS drawing bad noise, not the grader and not the perturbation; a cell whose baseline is
broken cannot measure anything. Cells are dropped on the CONTROL's own score alone, never on how the
perturbations inside them turned out, so the exclusion cannot bend the result.

**The yardstick: the positive control - one word deliberately mispronounced - cost 13.5%.** Read every
row against that, not against zero.

| variant | WER+ | hurt | sound | verdict |
|---|---|---|---|---|
| stress-moved-later | **34.3%** | 21/26 | 0.57 | worse than mispronouncing a word outright |
| no-stress-at-all | **19.6%** | 17/26 | 0.59 | worse than a wrong word |
| **stress-added-function-words** | **18.2%** | 15/26 | 0.52 | worse than a wrong word |
| *(positive control: one word wrong)* | *13.5%* | *21/26* | *0.26* | *the calibration* |
| turned-a-to-schwa | 5.9% | 3/8 | 0.19 | below the yardstick, n=8 |
| no-length-marks | 5.6% | 10/23 | 0.41 | below on words, MOVES THE SOUND |
| glottal-and-syllabic-to-plain | 5.1% | 1/3 | 0.18 | n=3, says nothing |
| barred-i-to-schwa | 4.6% | 5/11 | 0.25 | below |
| r-schwa-split | 4.3% | 6/20 | 0.40 | below on words, MOVES THE SOUND |
| open-o-to-open-a | 4.2% | 7/18 | 0.27 | below |
| no-secondary-stress | 0.9% | 2/9 | 0.37 | below on words, MOVES THE SOUND |
| flap-to-d | 0.8% | 3/18 | 0.21 | free |
| flap-to-t | 0.4% | 2/18 | 0.18 | free |
| barred-i-to-small-i | 0.3% | 3/11 | 0.20 | free |
| article-a-to-schwa | -0.6% | 4/11 | 0.22 | free |

#### What the frontend must do, in priority order

1. **Get stress right.** All three stress failures cost MORE than mispronouncing a word outright. This
   is the whole ballgame.
2. **Destress function words.** 18.2%, and it is what CMUdict will hand us on every `the`, `at`, `in`,
   `and` unless we stop it. Closed word list, so the rule is simple - but it is not optional.
3. **Emit length marks, keep r-coloured vowels whole, keep secondary stress.** These cost almost nothing
   in *words* (0.9% to 5.6%) but move the AUDIO as much as 0.41 against a segmental baseline of ~0.20.
   That is the "still intelligible, no longer sounds the same" band the acoustic axis was built to
   expose, and for a voice Aubs will listen to, naturalness is the product.
4. **Spend nothing on flaps, the reduced vowels, or the bare article.** Measured free on both axes.

#### Corrections this run forced on the earlier one

- **RETRACTED: "length marks are damaging at 14%".** That came from one sentence and one seed. Across
  nine sentences it is 5.6% on words - below the positive control. It survives in the list above only
  for its acoustic effect, which is a different and weaker claim.
- **The first grader was not fit for purpose.** whisper-tiny averaged 16.2% WER on UNDAMAGED audio and
  produced negative paired deltas (perturbed clips scoring better than clean ones), which is noise, not
  signal. whisper-base.en halved that floor. Using it required fixing a real library defect first - see
  below.
- **My own reporting of the acoustic scale was wrong** in the first pass: same-seed comparisons are
  bounded well below the seed-to-seed noise floor by construction, so "near 1x means nothing happened"
  was nonsense. It now reads as a relative scale where 1.00 is "as different as an independent take".

#### Library defects found and fixed on the way

`SpeechRecognitionPipeline` could not run ANY English-only Whisper checkpoint - it returned an empty
string with no error, indistinguishable from silent audio. Two causes:

1. `BPETokenizer.LoadFromTokenizerJson` ignored the tokenizer's `added_tokens` block, where every
   Whisper special token lives, so callers had to hard-code ids. Now parsed, plus `TryGetTokenId`.
2. The pipeline hard-coded the MULTILINGUAL ids and a fixed four-token prompt. The `.en` checkpoints
   carry a byte-level BPE vocabulary one entry smaller, so every special id shifts down by one. Special
   ids now resolve from the model's own tokenizer, and the prompt adapts to the family
   (`IsEnglishOnlyModel`, detected from the end-of-text id).

Guarded by `MLTestBase.WhisperTokenizerTests` - both families, the detection signal, and proof that
adding special tokens does not change how ordinary text encodes (which would have silently corrupted
GPT-2 and CLIP).

Also fixed: `ZipVoicePipeline.Dispose()` disposed the graphs it was HANDED, so a second pipeline over the
same graphs threw a NullReferenceException from inside onnxruntime.

#### The FIRST run, kept as a lesson

One sentence, one seed, ten variants, graded by whisper-tiny. It got the headline right (stress matters,
segmental detail does not) and got a detail wrong (length marks), had no positive control, and no idea
its grader was failing 16% of the time on clean audio. **A single-condition result is a hypothesis.**

#### A HUMAN EAR FOUND WHAT THREE INSTRUMENTS MISSED

TJ played the CONTROL clip - the undamaged baseline everything else is measured against - and heard
*"my mother wou-would rather"*. A repeated half-word. Nothing in the rig had reported it:

- **Whisper hid it.** It is a language model trained on real speech, and real speech is full of
  disfluencies, so it transcribes "would" once and moves on. Every stutter in this study was invisible
  to the grader by design.
- **The acoustic distance could not see it.** That metric compares a clip to its own control. When the
  CONTROL is the thing that stutters, the perturbed clips merely look "different" for a reason that has
  nothing to do with the perturbation.
- **A mel self-similarity detector written to catch it FAILED VALIDATION and was deleted.** Over these
  renders it scored stuttering clips at median 0.402 and clean ones at 0.413 - no separation whatsoever.
  It is recorded here so nobody rebuilds it believing it works.

What does work: an immediately repeated word in the transcript. **14 of 432 renders (3.2%)**, costing
**7.2% of WER** where they occur (20.3% against 13.1%). That is a FLOOR, not a count - the render TJ
caught by ear transcribed perfectly cleanly, so the true rate is higher.

**Does it change the conclusions? No.** The stutters are spread across variants (1-3 each out of 27)
rather than concentrated, so they widen every error bar without favouring any row, and the effect being
measured - 34%, 20% and 18% for the stress classes against 6% or less for everything segmental - is far
larger than a 7% noise source that lands on 3% of renders.

**Is it ours? No.** sherpa-onnx rendering the same sentence with the same prompt produced
*"understand that make me more their wood, rather water the better city garden"* - the same region,
mangled worse than ours. Same attribution as the prompt bleed below.

**The lesson for the rest of this project:** the automatic graders measure whether the words survive.
They do not measure whether it sounds like a person. Keep a human listening to the audio at every phase;
`tools/zipvoice-listen` exists for exactly that, and it earned its cost the first time it was used.

#### Artifact discovered while building the rig, worth not rediscovering

ZipVoice regenerates the reference clip's own speech ahead of the line it is asked to speak, and the cut
at the prompt boundary does not land cleanly, so the audio opens with a few words of the reference.
**This is not our pipeline.** sherpa-onnx does it too and worse: graded here, sherpa's own output for
this sentence opens "In guh, others call me mother nature. But the road is understand that..." while
ours opens "So me mother nature about the roses understand that...". Raising
`ReferenceTailSilenceSeconds` makes it worse, not better (0.25s -> 29% plain WER, 0.5s -> 43%,
1.0s -> 57%). Hence the infix scoring. Root-causing the bleed itself is separate work and is not a
phonemizer problem.

**Interpretation, decided in advance so the result cannot be rationalised afterwards:**

- Tolerant across most classes -> CMUdict maps in comfortably, proceed with Phase 3 onward.
- Brittle only on stress and segment identity -> proceed, and spend the effort on stress, not on
  allophonic detail.
- Brittle across the board -> the frontend cannot be approximate. Re-open the model choice before
  investing further (a model that tokenizes text directly, for example the XTTS family, sidesteps g2p
  entirely; its cost is size and licensing, both of which were open questions last session).

### Phase 1b - What CMUdict ACTUALLY gets wrong (loop closed) `[x]`

`tools/zipvoice-g2p-probe` builds the sequence CMUdict actually produces (first-cut ARPAbet-to-espeak
mapping, no flapping, no context rules, first pronunciation only) and aligns it WORD BY WORD against the
ids espeak actually produced, then classifies every difference. This exists because the Phase 1
perturbations were my PREDICTION of CMUdict's errors, and a prediction can be incomplete.

`dotnet run --project tools/zipvoice-g2p-probe -c Release`

**Result over the nine Phase 1 sentences: 62 differences across 592 espeak symbols (10.5%), and ZERO
out-of-vocabulary words.** So a naive dictionary mapping already agrees with espeak on ~90% of symbols.

| difference class | share | status |
|---|---|---|
| flap (water, better, city) | 24.2% | tested, harmless |
| **stress ADDED on function words** (at, for, in, and, a, is, it, of, than) | **19.4%** | **was UNTESTED** |
| barred-i vs schwa (packages, collected, boxes) | 6.5% | was UNTESTED |
| the article "a" (espeak emits a bare a) | 6.5% | was UNTESTED |
| turned-a (about) | 4.8% | tested, harmless |
| barred-i vs small-cap-i (roses, waited) | 4.8% | tested, harmless |
| length mark (often, on) | 4.8% | tested, DAMAGING |
| stress DROPPED (in) | 4.8% | tested, DAMAGING |
| small-cap-i vs schwa (hundred, packages) | 3.2% | was UNTESTED |
| turned-a vs ae (address, than) | 3.2% | was UNTESTED |
| primary/secondary swapped (address, seventeen) | 3.2% | tested, DAMAGING |
| open-o vs open-a (on) | 3.2% | was UNTESTED |
| glottal stop + syllabic n (kitten) | 4.8% | was UNTESTED |
| pre-vocalic ER0 (arrived: espeak turned-a + r, not r-schwa) | 1.6% | UNTESTED, rare |
| voicing (photographs) | 1.6% | UNTESTED, rare |
| small-cap-i vs close-i (pretending) | 1.6% | UNTESTED, rare |

**The find that justifies the whole exercise: stress ADDED on function words, 19.4% of all real
differences.** CMUdict stores citation forms, so "for", "at" and "in" carry a stress they lose in running
speech; espeak leaves them unstressed. That lands on the ONE axis Phase 1 showed the model is most
sensitive to, and the original perturbation set did not contain it - the first table only ever removed or
moved stress, never added it. Five perturbations were added to the rig to measure these classes:
`stress-added-function-words`, `barred-i-to-schwa`, `article-a-to-schwa`,
`glottal-and-syllabic-to-plain`, `open-o-to-open-a`.

**Consequence for the frontend regardless of how those score:** a function-word destressing pass is not
an optimisation, it is a core rule, and it is cheap to implement from a closed word list.

### Phase 1c - How much does CMUdict actually cover? `[x]`

Measured 2026-08-27 against the `google-10000-english-usa` frequency list (MIT) and CMUdict's 126,052
headwords. This decides how much weight Phase 6 (letter-to-sound) has to carry.

| word rank | not in CMUdict |
|---|---|
| top 1,000 | **0.5%** (5 words) |
| 1,000 - 3,000 | 3.3% |
| 3,000 - 10,000 | 7.2% |
| **overall top 10,000** | **5.7%** |

**But the misses are not words.** In frequency order they are: `ii, rss, faq, apr, jul, pics, ny, eur,
usr, fri, eg, thu, tx, ie, iii, gmt, fl, mb, prev, int, ...` - abbreviations, month and day
abbreviations, US state codes, units, roman numerals, and web-crawl noise.

**This reprioritises the plan.** The real gap at the top of the frequency distribution is
**abbreviations and acronyms, which is a TEXT NORMALIZATION problem (Phase 3), not a letter-to-sound
problem (Phase 6)**. "NY" wants expanding to "New York" or spelling out as letters; guessing it
phonetically is the wrong answer however good the guesser is. Phase 3 therefore covers more of the real
gap than Phase 6 does, and should come first.

⚠️ **Caveat: this list is web-crawl derived**, so it overstates web junk and understates PROPER NAMES.
Names are the true letter-to-sound case, and a spot check shows exactly where the line falls:

| in CMUdict | NOT in CMUdict |
|---|---|
| todd, nikki, tanner, claude, riker, spock, picard, nvidia, github | **aubriella**, aubs, tuvok, geordi, blazor, onnx, anthropic |

So Rose can say "Todd" and "Nikki" straight from the dictionary and **cannot say "Aubriella" without the
out-of-vocabulary path**. That is the requirement Phase 6 has to meet, stated concretely.
**Still owed:** a real name-coverage number against a census surname list rather than a spot check.

### Phase 2 - Validate against sentences the rules were NOT tuned on `[x]`

The Phase 4 rules were tuned against nine sentences, which is few enough to overfit. **120 Harvard
sentences** (IEEE Recommended Practice for Speech Quality Measurements, 1969 - public domain,
phonetically balanced, everyday vocabulary, every sixth sentence of the standard lists so the sample
spans all 72) were run through the oracle and used as a held-out set.

`dotnet run --project tools/zipvoice-fixture -c Release -- --file tools/zipvoice-fixture/sentences-phase2.txt --out tools/zipvoice-harness/fixtures/phase2`
`dotnet run --project tools/zipvoice-g2p-probe -c Release -- tools/zipvoice-harness/fixtures/phase2`

**Result: 4.5% symbol disagreement on the 120 unseen sentences against 4.2% on the nine that were tuned
against. The rules generalise; they are not overfitted.** And **0 of 940 words** fell outside CMUdict,
which is a stronger coverage result than the frequency-list estimate suggested.

The scale also exposed a systematic bug the nine sentences could not: **destressing was reducing the
VOWEL as well as removing the mark**, turning "of" into "uhv" and "and" into "uhnd" where the reference
keeps the fuller vowel and simply leaves it unmarked. That was 17% of all remaining differences.
Fixing it took the held-out set from 5.7% to 4.6%.

The original word-level batch oracle idea is superseded: sentences are a better instrument than isolated
words, because the function-word rule only exists in connected speech. Two of its sub-tasks are answered:
- [x] Words per run: 120 sentences in one command via `zipvoice-fixture`.
- [x] Abort synthesis early - **NOT POSSIBLE, measured.** ZipVoice does not stream, so sherpa's progress
      callback fires once after the audio already exists: 2076ms without the trick, 2192ms with it.

### Phase 2b - Batch word-level oracle (superseded, kept for reference) `[ ]`

- [ ] Extend `zipvoice-oracle` with a batch mode: feed many words in one run, capture the per-word
      token ids from the debug stream, emit a JSON word to ids map.
- [ ] Abort synthesis after the frontend has logged (sherpa's generate callback returning 0 stops
      generation), so the run cost is phonemization and not audio.
- [ ] Produce a scored word list: CMUdict headwords intersected with a frequency list, so the
      agreement number is weighted by what real text actually contains.

### Phase 3 - Text normalization in C# `[ ]` - PROMOTED, see Phase 1c

- [ ] Port `EnglishTextNormalizer` semantics (Apache-2.0 source, our own implementation).
- [ ] Cardinals, ordinals, years, money, decimals, fractions, percent, time, abbreviations, units.
- [ ] **Abbreviations, acronyms and initialisms** - Phase 1c showed these, not exotic words, are what
      CMUdict actually misses at the top of the frequency distribution. Expansion where the expansion is
      unambiguous (Mr, Dr, St, month and day names, US state codes, units), letter-by-letter spelling
      where it is not (RSS, FAQ, GMT). Both are normalization, not pronunciation guessing.
- [ ] The `map_punctuations` equivalent.
- [ ] Unit tests per category. This is the part users notice first and it has no licensing question,
      so it can proceed in parallel with anything else.

### Phase 4 - Dictionary lookup and ARPAbet to espeak-IPA mapping `[~]` STARTED

`SpawnDev.Phonemizer` exists: MIT, **zero dependencies**, browser-capable, no reference to ILGPU or ML.
`Arpabet` (symbol mapping), `FunctionWords` (the destressing list), `PronunciationDictionary` (CMUdict
loader), `EnglishPhonemizer` (the rules).

**Gate: `tools/zipvoice-g2p-probe`, which now runs the REAL library.** Symbol disagreement against
captured reference output over the nine Phase 1 sentences:

| state | differences | rate |
|---|---|---|
| throwaway inline table (Phase 1b) | 62 | 10.5% |
| library, first run | 66 | 11.1% |
| + punctuation written the reference's way, narrowed weak-form list | 50 | 8.4% |
| + tap T only (never D), drop that/my/would, add with | 42 | 7.1% |
| + stressless content words get their stress | 39 | 6.6% |
| + reduced vowel in real -es/-ed endings (stem checked in the dictionary) | 32 | 5.4% |
| + destressing removes the MARK only, not the vowel quality | 26 | 4.4% |
| + the LOT-CLOTH split (short o before a voiceless fricative) | **25** | **4.2%** |

Held-out check on 120 sentences the rules were never tuned against: **4.5%**, against 4.2% on the tuned
nine. It generalises.

**Tried and REVERTED, recorded so it is not retried:** closing a stressed AO to o before R. Right for
"boards" and "port", wrong for "quart", "or" and "for", which keep the opener vowel under stress too -
twenty new differences against four fixed. A rule that looks clean is not always a rule that measures.

**Every stress-ADDED and stress-DROPPED difference is now gone.** The two stress differences left are
primary/secondary swaps in "address" and "Seventeen" - genuine dictionary disagreements, and "address"
is a homograph, so both belong to Phase 5. Of the 39 remaining, 14 fall in classes the sensitivity
experiment measured as HARMLESS (the bare article, the two reduced vowels).

Rules implemented, in the order the measurement said they matter:
- [x] Stress mark placed before the VOWEL, not the syllable onset.
- [x] **Function-word destressing** - the 18.2% class. Narrow weak-form list; `in`, `on`, `about` and
      the polysyllabic prepositions deliberately excluded because the reference stresses them.
- [x] A content word with no stressed vowel in the dictionary (`in` is `IH0 N`) gets primary stress.
- [x] Length marks, secondary stress, r-coloured vowels preserved (the naturalness band).
- [x] Flapping, T only.
- [x] Reduced barred-i in real -es/-ed endings, with the stem checked in the dictionary so "hundred"
      (which is not "hundr" + ed) is correctly excluded.
- [x] Destressing removes the mark only, never the vowel quality.
- [x] The LOT-CLOTH split.
- [x] **Validated on 120 held-out sentences: 4.5% against 4.2% tuned. NOT overfitted.**
- [ ] Remaining top classes, in order: length marks on "on/onto/logs" (41), stress dropped from "A",
      "will", "such" where the reference keeps or downgrades it (33), the article "a" (21, measured
      harmless), unstressed IH vs AH in "jacket"/"acid" (18), "and" (12, the dictionary stores the weak
      form where the reference uses the strong one).
- [ ] The reference gives some weak words SECONDARY stress rather than none ("but" is bˌʌt, "such" is
      sˌʌtʃ) - a two-tier list would close part of the 33, but secondary stress measured nearly harmless
      so this is polish, not priority.

Guarded by `MLTestBase.PhonemizerTests` - 44/44 across cpu, cuda, opencl, WebGPU, WebGL and Wasm.

### Phase 4 (original scope) `[ ]`

- [ ] Ingest CMUdict into a compact lookup (browser-friendly; the raw file is a few MB of text).
- [ ] ARPAbet plus stress digit to the `tokens.txt` symbol set.
- [ ] Context rules for the classes Phase 1 says the model actually cares about.
- [ ] Score exact-sequence agreement against the Phase 2 oracle. **This number is the go/no-go.**

### Phase 5 - Homograph resolution `[ ]`

CMUdict lists multiple pronunciations and gives no way to choose. "I read the book" against "I will
read the book"; record, bass, live, wind, lead, tear, close, use, minute.

- [ ] Homograph list with part-of-speech conditions.
- [ ] A part-of-speech signal: a rule-based tagger, or a small model on our own ONNX engine.
- [ ] Test set of sentences containing both readings of each homograph.

### Phase 6 - Out-of-vocabulary letter to sound `[~]` WORKING BASELINE

`tools/lts-train` learns letter-to-sound FROM CMUdict and measures itself on 5,000 words held out
before training. Learned rather than hand-written because a hand-written ruleset encodes one person's
recollection of English spelling and can never honestly claim an accuracy figure.

`dotnet run --project tools/lts-train -c Release -- --out SpawnDev.Phonemizer/lts-model.txt`

**Held out: 43.7% of words exactly right guessing alone, 49.5% with decomposition first, 16.0% phoneme error rate.** Three designs were measured on the
same 5,000 words, and the intuitive one lost:

| design | words exactly right |
|---|---|
| stress digits baked into the sound emissions | 39.5% |
| word-level stress model (word ending + syllable count) | 37.2% |
| **separate sound and stress models over letter context** | 42.6% |
| **+ exactly one primary stress per word** | **43.7%** |

Stress genuinely IS a property of the word rather than of a letter, and modelling it that way still
lost - letter context carries more of the signal than a word ending does. Keeping stress out of the
SOUND emissions was worth it regardless: stress-blind accuracy rose 50.7% -> 52.1% by making that model
less sparse. Stress-only errors fell 14.8% -> 8.4%.

**It says the name:** "Aubriella" -> `ˈɔːbɹiɛlə`. Also sounded out: Tuvok, Geordi, Blazor, Anthropic.

⚠️ **Honest limits.** 43.7% means most unknown words come out with something wrong somewhere, and 16%
of phonemes are wrong. Published systems reach higher. This is a working baseline that unblocks names,
not a finished component:
- [ ] "Aubs" comes out with a doubled vowel (`ˈɔːaʊbz`) - the "au" digraph emits two vowels.
- [ ] Stress placement on "Aubriella" is first-syllable; a speaker would say aw-bree-EL-uh.
- [ ] The model file is **1.4 MB**, which is heavy for a browser. Pruning is packaging work (Phase 7).
- [x] **DONE and measured: decompose before guessing.** `WordDecomposer` builds an unknown word from a
      known stem plus its ending, with the allomorphy English actually demands - the -s of "cats" is an
      S, of "dogs" a Z, of "boxes" a whole syllable; the -ed of "walked" a T, of "wanted" a syllable.
      It also finds stems English spelling hides: "hoped" is hope+d with the e swallowed, "running" is
      run+ing with the n doubled.

      | on the same 5,000 held-out words | |
      |---|---|
      | fires on | 26.4% of them |
      | right when it fires | **77.2%** |
      | letter-to-sound alone, on those same words | 54.9% |
      | **overall, decompose-then-guess** | **49.5%** against 43.7% guessing alone |

      It declines rather than inventing: "aubriella" is not anything plus an ending, so it hands the
      word to letter-to-sound, which can at least try.

A silent-failure guard is locked by `MLTestBase.LetterToSoundTests`: when the runtime skipped the stress
model, every vowel came back with no stress digit, all of them were then discarded downstream as
unrecognised phones, and "Aubriella" was pronounced **"bɹl"** - the consonants alone, with no error
raised anywhere. The tests assert the output contract, not the accuracy.

### Phase 6 (original scope) `[ ]`

Names, brands, coinages. "Aubriella" is not in CMUdict.

- [ ] Decide between Flite LTS trees (license check first, see above) and a small model trained on
      CMUdict and exported to ONNX. The ONNX route runs on SpawnDev.ILGPU.ML in the browser and on the
      desktop, which would leave the phonemizer with no dependency outside our own stack.
- [ ] Score OOV accuracy against the oracle on a held-out word list.

### Phase 7 - Packaging `[ ]`

- [ ] `THIRD-PARTY-NOTICES.md` with the CMUdict notice and disclaimer verbatim.
- [ ] Unit tests in the repo's test project, so this is covered by PMT and not by a private script.
- [ ] README, including the crew credits section.

---

## Decisions locked

- **Placement**: starts as its own project in this repo, with no dependency on ILGPU or ML, so lifting
  it into its own repo and package later is a move rather than a rewrite. Working name
  `SpawnDev.Phonemizer`. The Phase 6 ONNX option, if chosen, goes behind an interface so the core
  stays dependency-free.
- **espeak-ng is a scoreboard, never a source.** No porting, no transcribing its rules, no training on
  its output. Ground truth is read through sherpa-onnx as an external process.
- **Grading is audio-first.** Token agreement is the fast proxy; Whisper transcription of synthesised
  audio is the real gate, because the target is what the model renders, not what espeak emits.

## Open questions

- [ ] Does the model's tolerance differ between the prompt token stream and the text token stream?
      (The prompt tokens condition the clone; an error there may matter more or less.)
- [ ] Does `piper_phonemize` post-process espeak's raw output in ways the CLI would not show? Relevant
      only if we ever compare against something other than sherpa.
- [ ] `AudioPreprocessor.Resample` is linear interpolation; sherpa uses band-limited (kaldi
      LinearResample). Pre-existing, unrelated to g2p, still open, matters when the reference clip is
      not already 24 kHz.

## Session log

- **2026-08-27 (Tuvok)**: Phase 1 REPLICATED at 432 renders with a stronger grader, a positive control
  and an acoustic second axis. Stress is everything: all three stress failures cost more than
  mispronouncing a word outright, and function-word destressing (the class the probe found) is confirmed
  at 18.2%. Retracted the earlier "length marks are damaging" claim. Fixed two library defects that made
  every English-only Whisper model decode to an empty string, with tests. Phase 1c measured CMUdict
  coverage: 99.5% of the top 1,000 words, and the misses are abbreviations, which promotes Phase 3.
- **2026-08-27 (Tuvok)**: Phase 1 first run - the model is tolerant of segmental error and
  brittle about stress, which is the branch that favours CMUdict. Built `tools/zipvoice-harness
  sensitivity` (`Sensitivity.cs`) and `fixtures/loaded-classes.json`. Fixed a real defect found on the
  way: `ZipVoicePipeline.Dispose()` disposed the graphs it was HANDED, so a second pipeline over the
  same graphs threw NRE inside onnxruntime - the same ownership rule the accelerator has in this repo.
- **2026-08-27 (Tuvok)**: Direction set. espeak-ng port abandoned on licensing (GPL-3) per TJ. Verified
  ZipVoice trains on `phonemize_espeak` en-us, so the model carries espeak's quirks. Verified CMUdict
  is BSD-2-Clause and Flite is BSD-like with per-file exceptions. Confirmed
  `zipvoice-harness synth` accepts arbitrary token ids with a pinned noise seed, which makes Phase 1 a
  controlled experiment on instruments already in the tree. Plan written. Phase 1 started.
