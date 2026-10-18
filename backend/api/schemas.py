"""Shared data contracts for Mixr.

All audio time values are in SECONDS. Changes to this file need approval
from all three developers (see CODEOWNERS).
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

FadeStyle = Literal["equal_power", "linear"]
RenderMode = Literal["preview", "final"]
MixMode = Literal["full", "set"]


class Section(BaseModel):
    start_seconds: float  # on a phrase start (the first section starts at 0)
    end_seconds: float
    energy: float  # mean loudness of the section, 0-1 relative to the track's loudest section
    label: Optional[str] = None  # "intro" | "verse" | "build" | "chorus" | "drop" | "break" | "outro"; None unless confident


class Track(BaseModel):
    id: str
    filename: str
    duration_seconds: float
    sample_rate: int
    bpm: Optional[float] = None
    beats_seconds: list[float] = Field(default_factory=list)
    key_camelot: Optional[str] = None  # null when key detection is unreliable
    energy_rms: list[float] = Field(default_factory=list)
    energy_hop_seconds: Optional[float] = None
    phrase_starts_seconds: list[float] = Field(default_factory=list)  # every 8 bars on the beat grid; [] without beats
    sections: list[Section] = Field(default_factory=list)  # contiguous, cover the whole track
    melody_activity: list[float] = Field(default_factory=list)  # 0-1 per energy_hop_seconds: lead melody/vocal present
    analysis_status: Literal["pending", "ok", "error"] = "pending"
    analysis_error: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class PairScore(BaseModel):
    from_id: str
    to_id: str
    total: float
    harmonic: Optional[float] = None
    tempo: Optional[float] = None
    energy: Optional[float] = None
    mood: Optional[float] = None  # null when Claude is unused or failed


class TransitionRequest(BaseModel):
    from_id: str
    to_id: str
    requested_overlap_seconds: float = Field(gt=0)
    fade_style: FadeStyle = "equal_power"
    beat_align: bool = True


class AppliedTransition(BaseModel):
    """What the renderer ACTUALLY did, not what was requested."""

    from_id: str
    to_id: str
    mode: RenderMode
    fade_style: FadeStyle
    requested_overlap_seconds: float
    actual_overlap_seconds: float
    overlap_changed_for_safety: bool
    outgoing_start_seconds: float  # where the overlap begins in the outgoing track
    incoming_start_seconds: float  # offset into the incoming track (0.0 in final mode)
    beat_alignment_adjustments_seconds: float = 0.0
    applied_gain_db: dict = Field(default_factory=dict)
    applied_eq: Optional[str] = None  # None means not applied
    tempo_change: Optional[float] = None  # None means not applied
    safety_adjustments: list[str] = Field(default_factory=list)
    output_duration_seconds: float


class Vote(BaseModel):
    from_id: str
    to_id: str
    a_requested_seconds: float
    b_requested_seconds: float
    chosen: Literal["A", "B"]


class Preference(BaseModel):
    overlap_preference: float = Field(0.0, ge=-1.0, le=1.0)
    votes: list[Vote] = Field(default_factory=list)
    completed: bool = False


class TrackWindow(BaseModel):
    """The part of a track that plays in the mix, in source-track seconds."""

    track_id: str
    start_seconds: float
    end_seconds: float
    reason: Optional[str] = None  # e.g. "the drop at 1:12 (loudest section)"


class MixManifest(BaseModel):
    mix_id: str
    ordered_track_ids: list[str]
    transitions: list[AppliedTransition]
    source_durations_seconds: list[float]
    output_duration_seconds: float  # approx sum(sources) - sum(actual overlaps)
    calibrated: bool
    overlap_preference: float
    limitations: list[str] = Field(default_factory=list)
    mode: MixMode = "full"
    windows: list[TrackWindow] = Field(default_factory=list)  # windows actually played, same order as ordered_track_ids
