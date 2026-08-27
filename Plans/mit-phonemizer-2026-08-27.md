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

### Phase 1 - Measure the model's sensitivity BEFORE building anything `[x]` (replication still owed)

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

#### RESULT, 2026-08-27

Run it with:
```
dotnet run --project tools/zipvoice-harness -c Release -- sensitivity
SENSITIVITY_REUSE_WAVS=1   # grade audio already rendered, for iterating on the grader
SENSITIVITY_ONLY=<variant> # render one variant plus the control, for probing the rig
ZIPVOICE_TAIL_PAD=<sec>    # reference tail silence
```

Fixture `fixtures/loaded-classes.json`, prompt `prompt.wav`, noise seed 1234, fp32 graphs on
onnxruntime, graded by whisper-tiny. `infix` is WER with free skips at the head and tail of the
transcript; the plain column is charged for those. **The control scores 0% infix, so the rig is sound.**

| variant | token edits | WER | infix WER | reading |
|---|---|---|---|---|
| control | 0 | 29% | **0%** | rig is valid |
| flap-to-t | 3 | 29% | **0%** | no damage |
| barred-i-to-small-i | 1 | 29% | **0%** | no damage |
| r-schwa-split | 10 | 29% | **0%** | no damage |
| no-secondary-stress | 1 | 29% | **0%** | no damage |
| flap-to-d | 3 | 29% | 7% | one inserted word |
| turned-a-to-schwa | 1 | 36% | 7% | one inserted word |
| no-length-marks | 2 | 43% | **14%** | damage |
| no-stress-at-all | 13 | 79% | **29%** | damage |
| stress-moved-later | 20 | 100% | **100%** | catastrophic: output collapsed to "It's a little bit of a problem." |

**The model is tolerant of segmental detail and brittle about prosody.** Every allophonic difference a
CMUdict frontend will produce - flaps, the reduced vowels, splitting the r-coloured vowel into schwa
plus r - cost nothing. What the model cares about is stress, and after that length.

That is the favourable branch, because **CMUdict carries lexical stress** (0/1/2 per vowel), which is
exactly the information the model is most sensitive to. Length is deterministic from vowel identity in
en-us, so the mapping table can emit it by rule rather than having to model it.

**Consequences for the frontend:**
- Stress placement is the top priority, and homograph resolution matters mostly because it moves stress
  (record the noun against record the verb). Phase 5 is not optional polish.
- The ARPAbet-to-IPA table MUST emit length marks; dropping them is measurable damage.
- Do NOT spend effort chasing espeak's flapping or its choice of reduced vowel symbol. Free.

**Caveats, stated so a later session does not over-read this:**
- ONE sentence, ONE reference voice, ONE noise seed. Replication across sentences, prompts and seeds is
  owed before this is treated as settled. The two 7% rows are single-word insertions well inside the
  range a different noise draw might produce on its own.
- whisper-tiny is a coarse grader: 0% means the words are recoverable, NOT that it sounds native. The
  wavs are in the out dir and still want a listen.

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

### Phase 2 - Batch word-level oracle `[ ]`

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

### Phase 4 - Dictionary lookup and ARPAbet to espeak-IPA mapping `[ ]`

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

### Phase 6 - Out-of-vocabulary letter to sound `[ ]`

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

- **2026-08-27 (Tuvok)**: Phase 1 RUN and answered - the model is tolerant of segmental error and
  brittle about stress, which is the branch that favours CMUdict. Built `tools/zipvoice-harness
  sensitivity` (`Sensitivity.cs`) and `fixtures/loaded-classes.json`. Fixed a real defect found on the
  way: `ZipVoicePipeline.Dispose()` disposed the graphs it was HANDED, so a second pipeline over the
  same graphs threw NRE inside onnxruntime - the same ownership rule the accelerator has in this repo.
- **2026-08-27 (Tuvok)**: Direction set. espeak-ng port abandoned on licensing (GPL-3) per TJ. Verified
  ZipVoice trains on `phonemize_espeak` en-us, so the model carries espeak's quirks. Verified CMUdict
  is BSD-2-Clause and Flite is BSD-like with per-file exceptions. Confirmed
  `zipvoice-harness synth` accepts arbitrary token ids with a pinned noise seed, which makes Phase 1 a
  controlled experiment on instruments already in the tree. Plan written. Phase 1 started.
