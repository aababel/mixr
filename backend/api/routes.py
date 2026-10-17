"""HTTP routes (owner: Shaan).

Upload is real. Analyze and order return placeholder data in the agreed
schema until backend/audio and backend/logic land. Mix calls
backend.audio.mix.render_mix once it exists (see docs/integration.md) and
returns 501 until then. Calibration returns 501.
"""
import importlib
import os
import uuid
from pathlib import Path

import soundfile as sf
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.api.schemas import MixManifest, PairScore, Track

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp3", ".wav"}
MIN_TRACKS = 3
MAX_TRACKS = 5
MAX_FILE_BYTES = 50 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
DEFAULT_OVERLAP_SECONDS = 8.0

_tracks: dict[str, Track] = {}
_track_paths: dict[str, Path] = {}
_mixes: dict[str, MixManifest] = {}
_mix_paths: dict[str, Path] = {}


def storage_dir() -> Path:
    path = Path(os.getenv("MIXR_STORAGE_DIR", "backend/storage"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_track(track_id: str) -> Track:
    track = _tracks.get(track_id)
    if track is None:
        raise HTTPException(404, f"Unknown track id: {track_id}")
    return track


def get_track_path(track_id: str) -> Path:
    get_track(track_id)
    return _track_paths[track_id]


class TrackIdsRequest(BaseModel):
    track_ids: list[str]


class UploadResponse(BaseModel):
    track_ids: list[str]
    tracks: list[Track]


class AnalyzeResponse(BaseModel):
    tracks: list[Track]


class OrderResponse(BaseModel):
    ordered_track_ids: list[str]
    scores: list[PairScore]


class MixRequest(BaseModel):
    ordered_track_ids: list[str]
    overlap_preference: float = Field(0.0, ge=-1.0, le=1.0)


class MixResponse(BaseModel):
    mix_id: str
    audio_url: str
    manifest: MixManifest


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    written = 0
    with dest.open("wb") as out:
        while chunk := await upload.read(CHUNK_BYTES):
            written += len(chunk)
            if written > MAX_FILE_BYTES:
                raise HTTPException(
                    413, f"{upload.filename} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB"
                )
            out.write(chunk)
    if written == 0:
        raise HTTPException(400, f"{upload.filename} is empty")


@router.post("/tracks", response_model=UploadResponse)
async def upload_tracks(files: list[UploadFile] = File(...)):
    if not MIN_TRACKS <= len(files) <= MAX_TRACKS:
        raise HTTPException(400, f"Upload between {MIN_TRACKS} and {MAX_TRACKS} files")

    for upload in files:
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"{upload.filename}: only MP3 and WAV are supported")

    paths: list[Path] = []
    tracks: list[Track] = []
    try:
        for upload in files:
            track_id = uuid.uuid4().hex[:12]
            dest = storage_dir() / f"{track_id}{Path(upload.filename).suffix.lower()}"
            paths.append(dest)
            await _save_upload(upload, dest)
            try:
                info = sf.info(str(dest))
            except Exception:
                raise HTTPException(400, f"{upload.filename} could not be decoded as audio")
            if info.frames <= 0:
                raise HTTPException(400, f"{upload.filename} contains no audio")
            tracks.append(
                Track(
                    id=track_id,
                    filename=upload.filename,
                    duration_seconds=info.frames / info.samplerate,
                    sample_rate=info.samplerate,
                )
            )
    except HTTPException:
        for path in paths:
            path.unlink(missing_ok=True)
        raise

    for track, path in zip(tracks, paths):
        _tracks[track.id] = track
        _track_paths[track.id] = path

    return UploadResponse(track_ids=[t.id for t in tracks], tracks=tracks)


@router.get("/tracks/{track_id}", response_model=Track)
def read_track(track_id: str):
    return get_track(track_id)


@router.get("/tracks/{track_id}/audio")
def read_track_audio(track_id: str):
    path = get_track_path(track_id)
    media_type = "audio/mpeg" if path.suffix == ".mp3" else "audio/wav"
    return FileResponse(path, media_type=media_type, filename=get_track(track_id).filename)


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: TrackIdsRequest):
    return AnalyzeResponse(tracks=[get_track(tid) for tid in request.track_ids])


@router.post("/order", response_model=OrderResponse)
def order(request: TrackIdsRequest):
    for tid in request.track_ids:
        get_track(tid)
    return OrderResponse(ordered_track_ids=list(request.track_ids), scores=[])


@router.post("/calibration/preview")
def calibration_preview():
    raise HTTPException(501, "Calibration previews need render_transition (backend/audio)")


@router.post("/calibration/vote")
def calibration_vote():
    raise HTTPException(501, "Calibration voting is not implemented yet (backend/logic)")


def _optional(module: str, name: str):
    """Return module.name, or None if the module or function doesn't exist yet."""
    try:
        return getattr(importlib.import_module(module), name, None)
    except ModuleNotFoundError as e:
        if e.name != module:
            raise
        return None


@router.post("/mix", response_model=MixResponse)
def mix(request: MixRequest):
    tracks = [get_track(tid) for tid in request.ordered_track_ids]
    if len(tracks) < 2:
        raise HTTPException(400, "A mix needs at least two tracks")
    if len(set(request.ordered_track_ids)) != len(tracks):
        raise HTTPException(400, "Each track can appear in the mix only once")

    render_mix = _optional("backend.audio.mix", "render_mix")
    if render_mix is None:
        raise HTTPException(501, "Mixing needs render_mix (backend/audio/mix.py)")
    recommend = _optional("backend.logic.calibration", "recommended_overlap_seconds")
    overlap = (
        recommend(request.overlap_preference) if recommend else DEFAULT_OVERLAP_SECONDS
    )

    mix_id = uuid.uuid4().hex[:12]
    mixes_dir = storage_dir() / "mixes"
    mixes_dir.mkdir(exist_ok=True)
    out_path = mixes_dir / f"{mix_id}.wav"
    try:
        manifest = render_mix(
            mix_id=mix_id,
            tracks=tracks,
            paths=[get_track_path(t.id) for t in tracks],
            overlap_seconds=[overlap] * (len(tracks) - 1),
            out_path=out_path,
        )
    except Exception as e:
        out_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Mix failed: {e}")
    if not out_path.exists():
        raise HTTPException(500, "Mix failed: renderer did not write an audio file")

    _mixes[mix_id] = manifest
    _mix_paths[mix_id] = out_path
    return _mix_response(mix_id)


def _mix_response(mix_id: str) -> MixResponse:
    manifest = _mixes.get(mix_id)
    if manifest is None:
        raise HTTPException(404, f"Unknown mix id: {mix_id}")
    return MixResponse(mix_id=mix_id, audio_url=f"/api/mixes/{mix_id}/audio", manifest=manifest)


@router.get("/mixes/{mix_id}", response_model=MixResponse)
def read_mix(mix_id: str):
    return _mix_response(mix_id)


@router.get("/mixes/{mix_id}/audio")
def read_mix_audio(mix_id: str):
    _mix_response(mix_id)
    return FileResponse(_mix_paths[mix_id], media_type="audio/wav", filename=f"mixr-{mix_id}.wav")
