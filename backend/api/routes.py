"""HTTP routes (owner: Shaan).

Thin layer over backend/audio (Aarav) and backend/logic (Sahil); see
docs/integration.md. All state is in memory for the session.
"""
import os
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional

import soundfile as sf
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.api.schemas import (
    AppliedTransition,
    MixManifest,
    PairScore,
    Preference,
    Track,
    TrackWindow,
    TransitionRequest,
    Vote,
)
from backend.audio.analysis import analyze_track
from backend.audio.harmony import chroma_for
from backend.audio.mix import render_mix
from backend.audio.mood import mood_scores
from backend.audio.preview import render_preview
from backend.logic.calibration import calibration_overlaps, update_preference
from backend.logic.explain import OrderExplanation, explain_order, explain_user_order
from backend.logic.ordering import order_tracks
from backend.logic.scoring import score_pairs
from backend.logic.windows import (
    DEFAULT_SET_SECONDS_PER_TRACK,
    TransitionLength,
    overlap_limits,
    plan_windows,
)

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp3", ".wav"}
MIN_TRACKS = 2
MAX_TRACKS = 5
MAX_FILE_BYTES = 50 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
MIN_AUDIBLE_OVERLAP_GAP_SECONDS = 2.0
MIN_SET_SECONDS = 120.0
MAX_SET_SECONDS = 1200.0

_tracks: dict[str, Track] = {}
_track_paths: dict[str, Path] = {}
_mixes: dict[str, MixManifest] = {}
_mix_paths: dict[str, Path] = {}
_trials: dict[str, "Trial"] = {}


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


class OrderRequest(TrackIdsRequest):
    # True: keep track_ids as the user arranged them and explain that order.
    keep_order: bool = False


class OrderResponse(BaseModel):
    ordered_track_ids: list[str]
    suggested_track_ids: list[str]  # Mixr's best order, even when the user kept their own
    scores: list[PairScore]
    explanation: OrderExplanation


class MixModeOptions(BaseModel):
    """Full mode plays every song; set mode trims each to its best part (docs/integration.md)."""

    mode: Literal["full", "set"] = "full"
    # Set mode only; defaults to DEFAULT_SET_SECONDS_PER_TRACK per song.
    target_seconds: Optional[float] = Field(None, ge=MIN_SET_SECONDS, le=MAX_SET_SECONDS)
    # Blend lengths: "auto" lets each pair of songs decide, "longer" leans every
    # seam long, "extended" allows blends up to 60 s (windows.BLEND_SIZES).
    transition_length: TransitionLength = "auto"


class MixRequest(MixModeOptions):
    ordered_track_ids: list[str]
    overlap_preference: float = Field(0.0, ge=-1.0, le=1.0)
    calibrated: bool = False


class PreviewRequest(MixModeOptions):
    ordered_track_ids: list[str]
    trial: int = Field(0, ge=0)
    # (from_id, to_id) seams already compared; they are never offered again.
    exclude_pairs: list[tuple[str, str]] = Field(default_factory=list)


class PreviewVariant(BaseModel):
    audio_url: str


class PreviewResponse(BaseModel):
    trial_id: str
    from_id: str
    to_id: str
    a: PreviewVariant
    b: PreviewVariant


class VoteRequest(BaseModel):
    trial_id: str
    chosen: Literal["A", "B"]
    preference: Preference = Field(default_factory=Preference)


class VoteResponse(BaseModel):
    preference: Preference
    chosen_overlap_seconds: float
    other_overlap_seconds: float
    chose_longer: bool


class Trial(BaseModel):
    trial_id: str
    vote: Vote
    applied: dict[str, AppliedTransition]
    paths: dict[str, Path]


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
    tracks = [get_track(tid) for tid in request.track_ids]
    pending = [t for t in tracks if t.analysis_status != "ok"]
    if pending:
        with ThreadPoolExecutor(max_workers=MAX_TRACKS) as pool:
            for analyzed in pool.map(lambda t: analyze_track(_track_paths[t.id], t), pending):
                _tracks[analyzed.id] = analyzed
    return AnalyzeResponse(tracks=[_tracks[tid] for tid in request.track_ids])


def _distinct_tracks(track_ids: list[str], minimum: int) -> list[Track]:
    tracks = [get_track(tid) for tid in track_ids]
    if len(tracks) < minimum:
        raise HTTPException(400, f"Need at least {minimum} tracks")
    if len(set(track_ids)) != len(tracks):
        raise HTTPException(400, "Each track can appear only once")
    return tracks


@router.post("/order", response_model=OrderResponse)
def order(request: OrderRequest):
    tracks = _distinct_tracks(request.track_ids, 1)
    scores = score_pairs(tracks, mood_scores(tracks))
    suggested = order_tracks(tracks, scores)
    if request.keep_order:
        ordered = list(request.track_ids)
        explanation = explain_user_order(tracks, scores, suggested, ordered)
    else:
        ordered = suggested
        explanation = explain_order(tracks, scores, request.track_ids, ordered)
    return OrderResponse(
        ordered_track_ids=ordered,
        suggested_track_ids=suggested,
        scores=scores,
        explanation=explanation,
    )


def _plan_windows(
    tracks: list[Track], overlap_preference: float, options: MixModeOptions
) -> tuple[list[TrackWindow], list[float]]:
    """Cue points for every song and a blend length for every seam, picked per
    pair of songs, in both modes (full mode = no target length)."""
    target = None
    if options.mode == "set":
        target = options.target_seconds or DEFAULT_SET_SECONDS_PER_TRACK * len(tracks)
        target = min(MAX_SET_SECONDS, max(MIN_SET_SECONDS, target))
    try:
        chroma = [chroma_for(get_track_path(t.id)) for t in tracks]
        return plan_windows(tracks, overlap_preference, target, chroma, options.transition_length)
    except Exception as e:
        raise HTTPException(500, f"Choosing cue points failed: {e}")


def _render_trial(
    outgoing: Track,
    incoming: Track,
    outgoing_window: TrackWindow | None = None,
    incoming_window: TrackWindow | None = None,
    transition_length: TransitionLength = "auto",
) -> Trial:
    trial_id = uuid.uuid4().hex[:12]
    previews_dir = storage_dir() / "previews"
    previews_dir.mkdir(exist_ok=True)
    short, long = calibration_overlaps(transition_length)
    a_seconds, b_seconds = (short, long) if random.random() < 0.5 else (long, short)
    applied, paths = {}, {}
    for variant, seconds in (("A", a_seconds), ("B", b_seconds)):
        paths[variant] = previews_dir / f"{trial_id}_{variant.lower()}.wav"
        applied[variant] = render_preview(
            _track_paths[outgoing.id],
            _track_paths[incoming.id],
            outgoing,
            incoming,
            TransitionRequest(
                from_id=outgoing.id, to_id=incoming.id, requested_overlap_seconds=seconds
            ),
            paths[variant],
            outgoing_window=outgoing_window,
            incoming_window=incoming_window,
            safety=overlap_limits(transition_length),
        )
    # Votes record what was actually heard, so a clamped pair can't fake a preference.
    vote = Vote(
        from_id=outgoing.id,
        to_id=incoming.id,
        a_requested_seconds=applied["A"].actual_overlap_seconds,
        b_requested_seconds=applied["B"].actual_overlap_seconds,
        chosen="A",
    )
    return Trial(trial_id=trial_id, vote=vote, applied=applied, paths=paths)


def _audible_gap(trial: Trial) -> float:
    return abs(trial.applied["A"].actual_overlap_seconds - trial.applied["B"].actual_overlap_seconds)


@router.post("/calibration/preview", response_model=PreviewResponse)
def calibration_preview(request: PreviewRequest):
    tracks = _distinct_tracks(request.ordered_track_ids, 2)
    # The preference isn't known yet, so cue points use the neutral one: the
    # windows the mix will play unless the votes lean its blends.
    windows, _ = _plan_windows(tracks, 0.0, request)
    excluded = set(request.exclude_pairs)
    seams = list(zip(tracks, tracks[1:], windows, windows[1:]))
    start = request.trial % len(seams)
    for outgoing, incoming, out_window, in_window in seams[start:] + seams[:start]:
        if (outgoing.id, incoming.id) in excluded:
            continue
        try:
            trial = _render_trial(outgoing, incoming, out_window, in_window, request.transition_length)
        except Exception as e:
            raise HTTPException(500, f"Preview failed: {e}")
        if _audible_gap(trial) >= MIN_AUDIBLE_OVERLAP_GAP_SECONDS:
            break
        for path in trial.paths.values():
            path.unlink(missing_ok=True)
    else:
        if excluded:
            raise HTTPException(422, "No more song pairs left to compare")
        raise HTTPException(
            422,
            "Blend length wouldn't make an audible difference for these songs "
            "(tempos too far apart for a long blend, or songs too short)",
        )

    trial_id = trial.trial_id
    _trials[trial_id] = trial
    return PreviewResponse(
        trial_id=trial_id,
        from_id=trial.vote.from_id,
        to_id=trial.vote.to_id,
        a=PreviewVariant(audio_url=f"/api/calibration/previews/{trial_id}/a"),
        b=PreviewVariant(audio_url=f"/api/calibration/previews/{trial_id}/b"),
    )


def _get_trial(trial_id: str) -> Trial:
    trial = _trials.get(trial_id)
    if trial is None:
        raise HTTPException(404, f"Unknown calibration trial: {trial_id}")
    return trial


@router.get("/calibration/previews/{trial_id}/{variant}")
def read_preview_audio(trial_id: str, variant: Literal["a", "b"]):
    path = _get_trial(trial_id).paths[variant.upper()]
    return FileResponse(path, media_type="audio/wav")


@router.post("/calibration/vote", response_model=VoteResponse)
def calibration_vote(request: VoteRequest):
    trial = _get_trial(request.trial_id)
    vote = trial.vote.model_copy(update={"chosen": request.chosen})
    other = "B" if request.chosen == "A" else "A"
    chosen_seconds = trial.applied[request.chosen].actual_overlap_seconds
    other_seconds = trial.applied[other].actual_overlap_seconds
    return VoteResponse(
        preference=update_preference(request.preference, vote),
        chosen_overlap_seconds=chosen_seconds,
        other_overlap_seconds=other_seconds,
        chose_longer=chosen_seconds > other_seconds,
    )


@router.post("/mix", response_model=MixResponse)
def mix(request: MixRequest):
    tracks = _distinct_tracks(request.ordered_track_ids, 2)
    windows, overlaps = _plan_windows(tracks, request.overlap_preference, request)

    mix_id = uuid.uuid4().hex[:12]
    mixes_dir = storage_dir() / "mixes"
    mixes_dir.mkdir(exist_ok=True)
    out_path = mixes_dir / f"{mix_id}.wav"
    try:
        manifest = render_mix(
            mix_id=mix_id,
            tracks=tracks,
            paths=[get_track_path(t.id) for t in tracks],
            overlap_seconds=overlaps,
            out_path=out_path,
            calibrated=request.calibrated,
            overlap_preference=request.overlap_preference,
            windows=windows,
            mode=request.mode,
            safety=overlap_limits(request.transition_length),
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
