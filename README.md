# Mixr

Your playlist. DJ'd by AI. Built at DubHacks 2026 by Shaan, Aarav, and Sahil.

Pipeline: Upload (2 to 5 MP3/WAV) → Analyze → Score → Order → optional A/B Tune → Mix → Play/Download.

## What it does

- **Analyze.** For every song: tempo, a kick-locked beat grid, musical key
  (left blank when detection isn't confident), loudness over time, melody
  activity, 8-bar phrase starts, sections (intro, chorus, drop, break, outro)
  and how long the song fades in and out.
- **Score and order.** Every pair of songs gets a compatibility score: 40%
  harmony (Camelot wheel), 35% tempo (half and double time count as a match),
  15% energy, 10% mood. A missing score (no confident key, no mood) counts as
  neutral. The Claude mood step is not wired in yet, so every pair gets a
  neutral mood and today's scores are fully deterministic. The best order is found by checking every ordering. Users can
  drag songs into their own order instead, and Mixr explains how it scores
  against its own pick.
- **Pick the played part of each song.** Full songs mode trims as little as
  possible. DJ set mode keeps each song's strongest stretch to fit a target
  length. Songs may be trimmed in either mode; the UI shows exactly which part
  of each song plays and why.
- **Render real transitions.** Each seam lines up on the beat and on a phrase,
  glides the tempo with Rubber Band when the songs differ, swaps the bass
  between songs, and matches loudness to -14 LUFS. When both songs have a lead
  melody at the seam, Mixr does a vocal swap or an echo-out instead of stacking
  two melodies. Blend length is chosen per pair; the transition length setting
  (Auto, Longer, Extended) leans it shorter or longer.
- **Tune (optional A/B).** Two previews of the same seam that differ only in
  requested blend length, played at equal loudness. Each vote moves a
  preference value by 0.5 within -1 to 1, and that value leans every blend in
  the final mix. This is a simple deterministic rule, not a trained model, and
  it only learns blend length. Skipping it gives a normal default mix.
- **One renderer, honest manifest.** The A/B previews and the final mix use
  the same `render_transition`. Each transition in the mix manifest records
  requested and actual blend length, beat adjustment, tempo glide, EQ, gains
  and any safety change.

## Before/after demo

`scripts/before_after.py` renders the same song pair two ways: a plain
crossfade like a normal music player, and Mixr's transition. Both use the same
decoded audio and the same loudness, so the only difference is the transition.

```bash
python scripts/before_after.py "outgoing.mp3" "incoming.mp3" --before-overlap 6 --out out/before_after/
```

It writes `before_plain_crossfade.wav`, `after_mixr.wav`,
`before_then_after.wav` (both back to back) and `applied.json` (what Mixr
actually did).

Other scripts: `scripts/analyze_files.py` (print analysis for files),
`scripts/render_pair.py` (render a short and a long blend of one pair),
`scripts/make_mix.py` (render a full mix from the command line).

## Setup

Requirements: Python 3.11 or 3.12 (not 3.13), Node 20+, FFmpeg, Rubber Band
(see below). Without Rubber Band the app still works, but seams between
tracks with different tempos fall back to a short 2-beat blend.

### Rubber Band

The `rubberband` command-line tool powers tempo glides between tracks. The
app finds it on your `PATH`; v3 or newer is needed for the higher-quality
`--fine` engine.

**macOS**

```bash
brew install rubberband
```

**Debian / Ubuntu**

```bash
sudo apt-get install rubberband-cli
```

**Windows** (PowerShell). There's no winget package, so install the official
prebuilt executable from [breakfastquay.com/rubberband](https://breakfastquay.com/rubberband/):

```powershell
$dest = "$env:LOCALAPPDATA\Programs\rubberband"
Invoke-WebRequest https://breakfastquay.com/files/releases/rubberband-4.0.0-gpl-executable-windows.zip -OutFile "$env:TEMP\rubberband.zip"
Expand-Archive "$env:TEMP\rubberband.zip" -DestinationPath "$env:TEMP\rubberband" -Force
New-Item -ItemType Directory -Force $dest | Out-Null
Copy-Item "$env:TEMP\rubberband\rubberband-4.0.0-gpl-executable-windows\*" $dest
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";$dest", "User")
```

Keep `sndfile.dll` in the same folder as `rubberband.exe`; it needs it to read
audio. The last line adds the folder to your user `PATH`, so open a **new**
terminal afterwards.

**Check it** (any OS, new terminal, from the repo root with the venv active):

```bash
rubberband --version             # 3.x or newer
python -c "from backend.audio.stretch import rubberband_available; print(rubberband_available())"   # True
```

Without it, the tests that need it (mostly `tests/test_glide.py`) are skipped.

### Backend

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # optional: add ANTHROPIC_API_KEY
uvicorn backend.main:app --reload --port 8000
```

Check: http://localhost:8000/api/health returns `{"ok": true}`. Interactive API docs: http://localhost:8000/docs

Frontend (in a second terminal, with the backend running):

```bash
cd frontend
npm install
npm run dev                      # http://localhost:3000
```

Set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` if the backend isn't on `http://localhost:8000`.

Tests: `pytest`

## Ownership

| Area | Owner |
| --- | --- |
| API, uploads, frontend, deployment | Shaan |
| Audio analysis, transition renderer, Claude integration | Aarav |
| Schemas, scoring, ordering, calibration, manifest | Sahil |

`backend/api/schemas.py` is shared: changes need all three to approve.

`docs/integration.md` defines the functions the API calls in `backend/audio` and `backend/logic`.

## Rules

- All audio time values are in seconds.
- One `render_transition` powers both A/B previews and the final mix.
- Songs may be trimmed to make a better transition or fit a set length; the
  manifest and UI always show which part of each song actually plays.
- Calibration and Claude are optional; the default mix must always work.
- Report what the renderer actually did, never only what was requested.
- Work on branches, merge through PRs.

Planning documents in `docs/` were written before the event.
