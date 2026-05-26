# Benchmark fixtures

Inputs the benchmark harness feeds to models. Each task in `tasks.json` references a fixture in this directory by `<task_id>.<ext>` convention.

## Text fixtures (`*.txt`)

Plain prompt text. The harness reads the file and uses it as the user message for task-mode runs. Mostly excerpted-and-anonymized snippets shaped to resemble production prompts so the resulting wall-clock numbers are predictive.

## Audio fixtures (`*.wav`)

### `audio_30s.wav`

- **Format:** 30 s, mono, 16 kHz, PCM s16le WAV (~938 KB).
- **Source:** LibriVox recording of *The Adventures of Tom Sawyer* by Mark Twain, chapter 01 — narrator: John Greenman, public domain.
  - Item: <https://archive.org/details/tom_sawyer_librivox>
  - Original file: `TSawyer_01-02_twain_64kb.mp3` (chapters 01–02 combined).
- **License:** Public domain. LibriVox recordings are explicitly placed in the public domain by their narrators; the source text (Twain, 1876) is also long out of copyright.
- **Cut:** First ~13 s of the chapter is the standard LibriVox boilerplate intro ("This is a LibriVox recording…"); we skip past it and take 30 s of actual book content beginning with the dedication and preface. Re-derivable with:

  ```bash
  ffmpeg -y -i TSawyer_01-02_twain_64kb.mp3 \
      -ss 13 -t 30 -ac 1 -ar 16000 -c:a pcm_s16le audio_30s.wav
  ```

- **Why this clip:** single male narrator, clean studio recording, standard English, well-known PD text so any model's transcription can be sanity-checked against the actual preface. Real prose with sentence boundaries and proper names — better signal than synthetic tones or read sentence lists.

### Why WAV instead of FLAC

WAV is universally accepted by every audio backend we benchmark (Whisper, Parakeet) without extra codec deps. The ~1 MB-per-fixture cost is acceptable for the small number of audio fixtures we expect to accumulate. If we ever hit a repo-size threshold we'll re-encode to FLAC and update both the fixtures and the harness's `audio_format` argument together.
