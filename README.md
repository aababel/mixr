# Mixr

Your playlist. DJ'd by AI. Built at DubHacks 2026 by Shaan, Aarav, and Sahil.

Pipeline: Upload (3–5 MP3/WAV) → Analyze → Score → Order → optional A/B calibration → Mix → Play/Download.

> Status: initial skeleton. Only the health route and shared schemas exist so far.

## Setup

Requirements: Python 3.11 or 3.12 (not 3.13), Node 20+, FFmpeg.

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # optional: add ANTHROPIC_API_KEY
uvicorn backend.main:app --reload --port 8000
```

Check: http://localhost:8000/api/health returns `{"ok": true}`.

Tests: `pytest`

## Ownership

| Area | Owner |
| --- | --- |
| API, uploads, frontend, deployment | Shaan |
| Audio analysis, transition renderer, Claude integration | Aarav |
| Schemas, scoring, ordering, calibration, manifest | Sahil |

`backend/api/schemas.py` is shared: changes need all three to approve.

## Rules

- All audio time values are in seconds.
- One `render_transition` powers both A/B previews and the final mix.
- Final mix keeps every track complete; only overlaps at the seams.
- Calibration and Claude are optional; the default mix must always work.
- Work on branches, merge through PRs.

Planning documents in `docs/` were written before the event.
