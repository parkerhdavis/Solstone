# Field Journal Test Content

This fork replaces the live `journal/` with content from [solpbc/field_journal](https://github.com/solpbc/field_journal) — a curated set of public-domain audio and screen recordings — so the instance runs on real, reproducible test material instead of personal capture.

This is a **fork-only** workflow. Upstream solstone leaves `journal/` to the user.


## One-time setup

### 1. Clone field_journal

```sh
git clone https://github.com/solpbc/field_journal ~/Field_Journal
```

`~/Field_Journal/` is read-only from this fork's perspective. Never commit, push, or let solstone write into it — the setup script exists specifically to avoid that by copying rather than symlinking.

### 2. Scaffold the journal

If you don't already have a configured `journal/` (identity, providers, convey secret, facets), bootstrap one the normal way first — `make install` plus whatever initial `sol` setup brings the journal to a usable state. `setup_field_journal.sh` only populates `chronicle/`; it expects the rest of the journal scaffolding to already exist.

If you're replacing an existing journal, back it up first:

```sh
mv journal journal.bak-$(date +%Y%m%d)
```

Then recreate the structural parts (config, identity, facets skeleton, tokens, link state, routines) in a fresh `journal/`, either by copying from the backup or by re-running setup. Do not carry over `chronicle/`, `indexer/`, `entities/`, or `health/` — those are derived and will be regenerated from field_journal media.

### 3. Populate chronicle from field_journal

```sh
./setup_field_journal.sh
```

Copies each `YYYYMMDD` day directory from `~/Field_Journal/journal/` into `journal/chronicle/`. Copies (not symlinks) — solstone writes derived artifacts (`audio.jsonl`, `audio.npz`, screen descriptions, etc.) as siblings of source media, and symlinking would dirty the field_journal clone.

Options:

- `--source PATH` — field_journal clone location (default: `~/Field_Journal`).
- `--force` — overwrite chronicle days that already exist.


## Refreshing after upstream updates

```sh
git -C ~/Field_Journal pull
./setup_field_journal.sh --force
```

`--force` replaces each day wholesale, including any derived artifacts solstone wrote under it. After a force-refresh, rerun the pipeline to rebuild derived state from the new media.


## Running the pipeline

Field_journal provides **media only** — transcripts, descriptions, entities, facets, and indexer state are not pre-built. After populating chronicle, run the stack (`sol up` / `make dev`) to have the think-side produce derived artifacts from the new media.


## Stream naming

Field_journal uses `field.audio` and `field.screen` as stream names. These are compatible with solstone as-is: the stream-name validator accepts dotted names, and downstream processing (sense, transcribe, describe) dispatches on file extension rather than stream name. No rename step is required.
