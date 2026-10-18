"""Musical cue analysis: melody activity, phrase starts and sections (owner: Aarav).

Used by analysis.py; choose_windows (backend/logic) reads the results to pick
where each song is left and entered. All times are seconds.

melody_activity
    How much lead melody or vocal is present, 0-1, one value per
    ENERGY_HOP_SECONDS block (same grid as energy_rms). Harmonic part of an
    HPSS split (drums removed), energy in MELODY_BAND_HZ, as amplitude
    relative to the track's own MELODY_REF_PERCENTILE (so 1.0 = as melodic as
    this song gets, 0.5 = about 6 dB quieter). Sustained pads and chords in
    that band count too: this is a cheap proxy, not vocal detection.

phrase_starts_seconds
    Every PHRASE_BARS bars on the beat grid (beats_seconds). Bars are 4
    beats; the downbeat is the beat phase where chords change and low-end
    onsets hit hardest. The phrase phase is the bar offset where the music
    changes most (chroma, loudness and melody before vs after).

sections
    Consecutive phrases merged while they sound alike; a boundary is kept
    where the feature change is clearly above this track's typical change.
    Contiguous and covering the whole track (first starts at 0, last ends at
    the duration). Labels only where the energy shape makes them obvious:
    a quiet first section is "intro", a quiet last one "outro", a quiet one
    in between "break". Everything else stays None.
"""
import librosa
import numpy as np

from backend.api.schemas import Section

MELODY_BAND_HZ = (300.0, 3400.0)
MELODY_REF_PERCENTILE = 95
N_FFT, HOP = 2048, 512
HPSS_MAX_HZ = 4500.0  # HPSS only on bins below this (melody band plus margin): 2-3x faster
LOW_ONSET_HZ = 150.0
BEATS_PER_BAR = 4
PHRASE_BARS = 8
NOVELTY_BARS = 4  # compare this many bars before vs after a boundary
SECTION_SPLIT = 1.25  # keep a phrase boundary if its change > this x the median change
QUIET_SECTION = 0.6  # energy below this (relative to loudest section) can be intro/outro
BREAK_SECTION = 0.5


def _spectrum(y: np.ndarray, sr: int) -> np.ndarray:
    return np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2


def melody_activity(power: np.ndarray, sr: int, hop_seconds: float, n_blocks: int) -> list[float]:
    """0-1 per hop_seconds block from an STFT power spectrogram (bins, frames)."""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    top = int(np.searchsorted(freqs, HPSS_MAX_HZ)) + 1
    harmonic, _ = librosa.decompose.hpss(power[:top], margin=1.0)
    band = (freqs[:top] >= MELODY_BAND_HZ[0]) & (freqs[:top] <= MELODY_BAND_HZ[1])
    frame_energy = harmonic[band].sum(axis=0)

    frame_t = librosa.frames_to_time(np.arange(frame_energy.size), sr=sr, hop_length=HOP)
    idx = np.minimum((frame_t / hop_seconds).astype(int), n_blocks - 1)
    sums = np.bincount(idx, weights=frame_energy, minlength=n_blocks)
    counts = np.bincount(idx, minlength=n_blocks)
    amp = np.sqrt(np.divide(sums, counts, out=np.zeros(n_blocks), where=counts > 0))
    ref = float(np.percentile(amp, MELODY_REF_PERCENTILE)) if amp.size else 0.0
    if ref <= 1e-9:
        return [0.0] * n_blocks
    return [round(float(v), 4) for v in np.clip(amp / ref, 0.0, 1.0)]


def _beat_features(power: np.ndarray, sr: int, beats: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Per-beat (chroma 12 + log loudness + melody-band level) features and
    per-beat low-end level (log energy below LOW_ONSET_HZ at the beat). Row i covers beats[i]..beats[i+1]."""
    frames = librosa.time_to_frames(beats, sr=sr, hop_length=HOP)
    frames = np.clip(frames, 0, power.shape[1] - 1)
    chroma = librosa.feature.chroma_stft(S=power, sr=sr, n_fft=N_FFT, hop_length=HOP)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    band = (freqs >= MELODY_BAND_HZ[0]) & (freqs <= MELODY_BAND_HZ[1])
    loud = np.log1p(power.sum(axis=0))[None, :]
    mel = np.log1p(power[band].sum(axis=0))[None, :]
    low = np.log(power[freqs <= LOW_ONSET_HZ].sum(axis=0) + 1e-6)

    feats = librosa.util.sync(np.vstack([chroma, loud, mel]), frames, aggregate=np.mean)
    # low-end level right at each beat (the kick on beat 1 is usually the hardest)
    onset = np.array([low[max(0, f - 1):f + 4].max() for f in frames])
    # sync() adds a segment before the first beat; drop it so row i = beat i
    return feats[:, 1:len(beats) + 1].T, onset


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(s > 1e-9, s, 1.0)


def downbeat_phase(feats: np.ndarray, onset: np.ndarray) -> int:
    """Which of the first 4 beats is a downbeat: the low end hits hardest and
    chords change on beat 1 (chord change weighted half: it is noisy)."""
    chroma = feats[:, :12]
    change = np.concatenate([[0.0], np.linalg.norm(np.diff(chroma, axis=0), axis=1)])
    score = 0.5 * _z(change[:, None])[:, 0] + _z(onset[:, None])[:, 0]
    return int(np.argmax([score[d::BEATS_PER_BAR].mean() for d in range(BEATS_PER_BAR)]))


def _bar_novelty(bar_feats: np.ndarray) -> np.ndarray:
    """Change at each bar line: distance between the mean of the bars before
    and after it (up to NOVELTY_BARS each side). Entry 0 is 0."""
    z = _z(bar_feats)
    nov = np.zeros(len(z))
    for b in range(1, len(z)):
        before = z[max(0, b - NOVELTY_BARS):b].mean(axis=0)
        after = z[b:b + NOVELTY_BARS].mean(axis=0)
        nov[b] = float(np.linalg.norm(after - before))
    return nov


def phrases_and_sections(
    power: np.ndarray, sr: int, beats: list[float], duration: float,
    energy_rms: list[float], hop_seconds: float,
) -> tuple[list[float], list[Section], list[str]]:
    notes: list[str] = []
    beats = [b for b in beats if 0 <= b < duration]
    if len(beats) < 2 * BEATS_PER_BAR:
        return [], [], ["too few beats for phrase detection; no phrase starts or sections"]

    feats, onset = _beat_features(power, sr, beats)
    d = downbeat_phase(feats, onset)
    bar_idx = list(range(d, len(beats), BEATS_PER_BAR))
    bar_feats = np.array([feats[i:i + BEATS_PER_BAR].mean(axis=0) for i in bar_idx])

    nov = _bar_novelty(bar_feats)
    if len(bar_idx) >= 2 * PHRASE_BARS:
        k = int(np.argmax([nov[o::PHRASE_BARS].mean() for o in range(PHRASE_BARS)]))
    else:
        k = 0
        notes.append(f"fewer than {2 * PHRASE_BARS} bars: phrases counted from the first downbeat")
    phrase_bars = list(range(k, len(bar_idx), PHRASE_BARS))
    phrase_starts = [round(beats[bar_idx[b]], 4) for b in phrase_bars]

    # sections: merge consecutive phrases that sound alike
    z = _z(bar_feats)
    bounds = phrase_bars + [len(bar_idx)]
    pf = np.array([z[a:b].mean(axis=0) for a, b in zip(bounds, bounds[1:])])
    if len(pf) >= 2:
        change = np.linalg.norm(np.diff(pf, axis=0), axis=1)
        keep = change > SECTION_SPLIT * float(np.median(change))
    else:
        keep = np.zeros(0, dtype=bool)
    starts = [0.0] + [phrase_starts[j + 1] for j in range(len(keep)) if keep[j]]
    # a sliver after the last kept boundary (the song's final hit) joins the section before it
    phrase_len = PHRASE_BARS * BEATS_PER_BAR * float(np.median(np.diff(beats)))
    if len(starts) > 1 and duration - starts[-1] < 0.5 * phrase_len:
        starts.pop()
    ends = starts[1:] + [round(duration, 4)]

    rms = np.asarray(energy_rms, dtype=np.float64)

    def mean_energy(a: float, b: float) -> float:
        i0, i1 = int(a / hop_seconds), max(int(a / hop_seconds) + 1, int(np.ceil(b / hop_seconds)))
        seg = rms[i0:i1]
        return float(seg.mean()) if seg.size else 0.0

    raw = [mean_energy(a, b) for a, b in zip(starts, ends)]
    top = max(raw) if raw and max(raw) > 0 else 1.0
    sections: list[Section] = []
    for i, (a, b, e) in enumerate(zip(starts, ends, raw)):
        energy = round(e / top, 3)
        label = None
        if len(starts) >= 3:
            if i == 0 and energy < QUIET_SECTION:
                label = "intro"
            elif i == len(starts) - 1 and energy < QUIET_SECTION:
                label = "outro"
            elif 0 < i < len(starts) - 1 and energy < BREAK_SECTION:
                label = "break"
        sections.append(Section(start_seconds=round(a, 4), end_seconds=round(b, 4),
                                energy=energy, label=label))
    return phrase_starts, sections, notes


def analyze_cues(
    y: np.ndarray, sr: int, beats: list[float], duration: float,
    energy_rms: list[float], hop_seconds: float,
) -> dict:
    """melody_activity, phrase_starts_seconds, sections and notes for one track."""
    power = _spectrum(y, sr)
    melody = melody_activity(power, sr, hop_seconds, len(energy_rms))
    phrases, sections, notes = phrases_and_sections(power, sr, beats, duration, energy_rms, hop_seconds)
    return {
        "melody_activity": melody,
        "phrase_starts_seconds": phrases,
        "sections": sections,
        "notes": notes,
    }
