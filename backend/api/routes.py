"""HTTP routes (owner: Shaan).

Upload is real. Analyze and order return placeholder data in the agreed
schema until backend/audio and backend/logic land. Calibration and mix
return 501 rather than fake audio.
"""
import os
import uuid
from pathlib import Path

import soundfile as sf
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.api.schemas import PairScore, Track

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp3", ".wav"}
MIN_TRACKS = 3
MAX_TRACKS = 5
MAX_FILE_BYTES = 50 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024

_tracks: dict[str, Track] = {}
_track_paths: dict[str, Path] = {}


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


@router.post("/mix")
def mix():
    raise HTTPException(501, "Mixing needs render_transition (backend/audio)")


@router.get("/mixes/{mix_id}")
def read_mix(mix_id: str):
    raise HTTPException(404, f"Unknown mix id: {mix_id}")
