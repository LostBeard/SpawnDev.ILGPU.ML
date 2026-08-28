# Third-party notices

`SpawnDev.Phonemizer` is MIT licensed. It ships two things derived from third-party data, both under
permissive licences, and both are acknowledged here as those licences require.

---

## CMU Pronouncing Dictionary (CMUdict)

Used as the pronunciation dictionary, and as the training data from which `lts-model.txt` (the
letter-to-sound rules) was learned. Anything derived from CMUdict carries the same notice, so this
covers both.

Source: https://github.com/cmusphinx/cmudict

```
Copyright (C) 1993-2015 Carnegie Mellon University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   The contents of this file are deemed to be source code.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the
   distribution.

This work was supported in part by funding from the Defense Advanced
Research Projects Agency, the Office of Naval Research and the National
Science Foundation of the United States of America, and by member
companies of the Carnegie Mellon Sphinx Speech Consortium. We acknowledge
the contributions of many volunteers to the expansion and improvement of
this dictionary.

THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## Harvard sentences

Used only as TEST DATA, in `tools/zipvoice-fixture/sentences-phase2.txt`. Not shipped in the library.

The sentences are from *IEEE Recommended Practice for Speech Quality Measurements* (IEEE Transactions on
Audio and Electroacoustics, 1969), and are in the public domain. They are phonetically balanced, which is
why they make a fair held-out set.

---

## What is NOT here, deliberately

**espeak-ng is not used.** It is the frontend that every comparable open text-to-speech system depends on,
and it is GPL-3, which is why this library exists at all. No espeak-ng code, data, or rules are present in
this project or were transcribed into it.

espeak-ng was used as a *measuring instrument* during development, in the same sense a compiler is: its
output, read through sherpa-onnx, is the reference that this library's accuracy is scored against. Reading
a system's output to check your own work creates no derivative of it.

---

## Notes for redistributors

The obligation CMUdict places on you is the same one MIT places on users of this library: keep the
copyright notice and the disclaimer with the distribution. Shipping this file alongside the assembly
satisfies it. There is no restriction on commercial use, and no copyleft.
