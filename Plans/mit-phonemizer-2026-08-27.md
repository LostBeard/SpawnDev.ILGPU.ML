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

### Phase 2 - Batch word-level oracle `[ ]`

- [ ] Extend `zipvoice-oracle` with a batch mode: feed many words in one run, capture the per-word
      token ids from the debug stream, emit a JSON word to ids map.
- [ ] Abort synthesis after the frontend has logged (sherpa's generate callback returning 0 stops
      generation), so the run cost is phonemization and not audio.
- [ ] Produce a scored word list: CMUdict headwords intersected with a frequency list, so the
      agreement number is weighted by what real text actually contains.

### Phase 3 - Text normalization in C# `[ ]`

- [ ] Port `EnglishTextNormalizer` semantics (Apache-2.0 source, our own implementation).
- [ ] Cardinals, ordinals, years, money, decimals, fractions, percent, time, abbreviations, units.
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
