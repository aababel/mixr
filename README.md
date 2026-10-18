# Mixr

Your playlist. DJ'd by AI. Built at DubHacks 2026 by Shaan, Aarav, and Sahil.

Pipeline: Upload (3–5 MP3/WAV) → Analyze → Score → Order → optional A/B calibration → Mix → Play/Download.

> Status: upload works end to end. Analyze and order return placeholder data; calibration and mix return 501 until the renderer lands.

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

Without it, the glide tests in `tests/test_glide.py` are skipped, and two
tests in `tests/test_mix.py` currently fail because they assume a tempo glide.

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
- Final mix keeps every track complete; only overlaps at the seams.
- Calibration and Claude are optional; the default mix must always work.
- Work on branches, merge through PRs.

Planning documents in `docs/` were written before the event.
