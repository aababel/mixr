"""Track analysis (owner: Aarav).

analyze_track(path, track) fills the musical fields of a Track using librosa.
It never raises for bad audio: failures set analysis_status="error" with a
readable analysis_error. All times are seconds.

Fields filled:
  bpm, beats_seconds      librosa beat tracker on the mono signal; bpm is
                          60 / mean regular beat gap (more precise than librosa's tempo),
                          then folded into TEMPO_RANGE (half/double-time fix);
                          metadata.bpm_detected keeps the unfolded value
  key_camelot             Krumhansl-Schmuckler key estimate on chroma; None
                          when the estimate is weak (see KEY_MIN_CORRELATION)
  energy_rms              RMS per ENERGY_HOP_SECONDS block (0..1 float)
  energy_hop_seconds      ENERGY_HOP_SECONDS
  metadata                key_name, key_confidence, intro/outro mean energy,
                          analysis_sample_rate, notes
"""
from pathlib import Path

import librosa
import numpy as np

from backend.api.schemas import Track

ANALYSIS_SR = 22050
ENERGY_HOP_SECONDS = 0.5
EDGE_SECONDS = 15.0  # window used for intro/outro energy
KEY_MIN_CORRELATION = 0.6  # below this the key is reported as None
TEMPO_RANGE = (90.0, 180.0)  # detected tempos are folded into [lo, hi) BPM

# Krumhansl-Kessler key profiles, index 0 = tonic.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Camelot wheel number by tonic pitch class.
_CAMELOT_MAJOR = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]
_CAMELOT_MINOR = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10]


def estimate_key(y: np.ndarray, sr: int) -> tuple[str | None, str | None, float]:
    """Return (camelot, key_name, correlation). camelot is None if weak."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    profile = chroma.mean(axis=1)
    if not np.any(profile > 0) or np.std(profile) < 1e-6:
        return None, None, 0.0
    best = (-2.0, 0, "major")
    for tonic in range(12):
        for mode, template in (("major", _MAJOR), ("minor", _MINOR)):
            r = float(np.corrcoef(profile, np.roll(template, tonic))[0, 1])
            if r > best[0]:
                best = (r, tonic, mode)
    r, tonic, mode = best
    if mode == "major":
        camelot, name = f"{_CAMELOT_MAJOR[tonic]}B", f"{_NAMES[tonic]} major"
    else:
        camelot, name = f"{_CAMELOT_MINOR[tonic]}A", f"{_NAMES[tonic]} minor"
    if r < KEY_MIN_CORRELATION:
        return None, name, r
    return camelot, name, r


def fold_tempo(
    bpm: float, beats: list[float], lo: float = TEMPO_RANGE[0], hi: float = TEMPO_RANGE[1]
) -> tuple[float, list[float], str | None]:
    """Fold half/double-time detections into [lo, hi) BPM.

    The beat tracker often locks onto every other beat of dance music
    (e.g. 64 instead of 128). Doubling inserts the midpoint between each
    pair of detected beats; halving keeps every other beat.
    """
    original = bpm
    while bpm < lo and len(beats) >= 2:
        mids = [(a + b) / 2 for a, b in zip(beats, beats[1:])]
        beats = sorted(beats + mids)
        bpm *= 2
    while bpm >= hi and len(beats) >= 2:
        beats = beats[::2]
        bpm /= 2
    beats = [round(b, 4) for b in beats]
    if bpm == original:
        return bpm, beats, None
    return bpm, beats, f"tempo folded from {original:.2f} to {bpm:.2f} BPM (half/double-time detection)"


def block_rms(y: np.ndarray, sr: int, hop_seconds: float) -> list[float]:
    hop = max(1, int(round(hop_seconds * sr)))
    n_blocks = int(np.ceil(len(y) / hop))
    padded = np.zeros(n_blocks * hop, dtype=np.float32)
    padded[: len(y)] = y
    blocks = padded.reshape(n_blocks, hop)
    return [round(float(v), 5) for v in np.sqrt(np.mean(blocks**2, axis=1))]


def analyze_track(path: Path, track: Track) -> Track:
    try:
        y, sr = librosa.load(str(path), sr=ANALYSIS_SR, mono=True)
    except Exception as exc:  # unreadable or undecodable file
        return track.model_copy(
            update={"analysis_status": "error", "analysis_error": f"could not decode audio: {exc}"}
        )
    if y.size == 0:
        return track.model_copy(
            update={"analysis_status": "error", "analysis_error": "file contains no audio"}
        )

    try:
        notes: list[str] = []
        decoded_seconds = len(y) / sr

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        beats_seconds = [round(float(b), 4) for b in beats]
        # Beat times sit on a ~23 ms frame grid, so single gaps (and their median)
        # are quantized. Averaging the regular gaps (within 25% of the median,
        # which drops missed or doubled beats) cancels that rounding.
        if len(beats_seconds) >= 2:
            gaps = np.diff(beats_seconds)
            med = float(np.median(gaps))
            regular = gaps[np.abs(gaps - med) <= 0.25 * med]
            bpm = 60.0 / float(np.mean(regular if regular.size else gaps))
        else:
            bpm = float(np.atleast_1d(tempo)[0])
        if bpm <= 0 or len(beats_seconds) < 2:
            bpm = None
            beats_seconds = []
            notes.append("no reliable beat found; transitions will not beat-snap")
        bpm_detected = bpm
        if bpm is not None:
            bpm, beats_seconds, fold_note = fold_tempo(bpm, beats_seconds)
            if fold_note:
                notes.append(fold_note)

        camelot, key_name, key_r = estimate_key(y, sr)
        if camelot is None:
            notes.append("key estimate too weak; key_camelot left empty")

        energy = block_rms(y, sr, ENERGY_HOP_SECONDS)
        edge_blocks = max(1, int(EDGE_SECONDS / ENERGY_HOP_SECONDS))

        metadata = dict(track.metadata)
        metadata.update(
            {
                "bpm_detected": round(bpm_detected, 2) if bpm_detected else None,
                "key_name": key_name,
                "key_confidence": round(key_r, 3),
                "intro_energy": round(float(np.mean(energy[:edge_blocks])), 5),
                "outro_energy": round(float(np.mean(energy[-edge_blocks:])), 5),
                "analysis_sample_rate": sr,
                "decoded_duration_seconds": round(decoded_seconds, 3),
                "analysis_notes": notes,
            }
        )

        return track.model_copy(
            update={
                "bpm": round(bpm, 2) if bpm else None,
                "beats_seconds": beats_seconds,
                "key_camelot": camelot,
                "energy_rms": energy,
                "energy_hop_seconds": ENERGY_HOP_SECONDS,
                "analysis_status": "ok",
                "analysis_error": None,
                "metadata": metadata,
            }
        )
    except Exception as exc:
        return track.model_copy(
            update={"analysis_status": "error", "analysis_error": f"analysis failed: {exc}"}
        )
