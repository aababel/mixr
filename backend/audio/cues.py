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
    where the feature change is clearly above this track's typical change or
    the loudness steps up or down, and no section is longer than
    MAX_SECTION_PHRASES phrases (16 bars). Contiguous and covering the whole
    track (first starts at 0, last ends at the duration). Labels only where
    the shape makes them obvious: a quiet first section is "intro", a quiet
    last one "outro", a quiet one in between "break"; a loud section the
    music lifts into is a "drop" (little lead melody) or "chorus" (melodic),
    and a loud melodic section that comes back later is a "chorus" too.
    Everything else stays None.
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
ENERGY_STEP = 0.2  # a phrase this much louder or quieter than the one before starts a section
MAX_SECTION_PHRASES = 2  # longer sections are split at their biggest change (16 bars, ~30 s)
LOUD_SECTION = 0.8  # chorus / drop candidates
LIFT = 0.2  # energy rise from the section before that marks a chorus or drop entrance
MELODIC_SECTION = 0.5  # mean melody_activity: chorus at or above, drop below
REPEAT_SIMILARITY = 0.6  # "comes back": feature distance below this x the median between sections


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


def _span_mean(series: np.ndarray | None, hop: float, a: float, b: float) -> float:
    if series is None or not series.size or hop <= 0:
        return 0.0
    i0 = int(a / hop)
    seg = series[i0:max(i0 + 1, int(np.ceil(b / hop)))]
    return float(seg.mean()) if seg.size else 0.0


def _sections(
    bar_feats: np.ndarray, phrase_bars: list[int], phrase_starts: list[float],
    beats: list[float], duration: float, energy_rms: list[float], hop: float,
    melody: list[float] | None,
) -> list[Section]:
    """Group phrases into sections and label the obvious ones.

    Boundaries: a phrase start is a section start when the music changes
    clearly there (feature change > SECTION_SPLIT x this track's median
    change) or the loudness steps by >= ENERGY_STEP. Any section longer than
    MAX_SECTION_PHRASES phrases is then split at its biggest internal change,
    so no section can swallow half a song.
    """
    n_bars = len(bar_feats)
    bounds = phrase_bars + [n_bars]
    m = len(phrase_bars)
    z = _z(bar_feats)
    pf = np.array([z[a:b].mean(axis=0) for a, b in zip(bounds, bounds[1:])])
    rms = np.asarray(energy_rms, dtype=np.float64)
    mel = np.asarray(melody, dtype=np.float64) if melody else None
    # phrase j covers [p_start[j], p_end[j]); the first one also takes the lead-in
    p_start = [0.0] + phrase_starts[1:]
    p_end = phrase_starts[1:] + [duration]
    p_energy = np.array([_span_mean(rms, hop, a, b) for a, b in zip(p_start, p_end)])
    top_e = p_energy.max() if p_energy.size and p_energy.max() > 0 else 1.0
    p_energy = p_energy / top_e

    # change at the start of phrase j (j >= 1)
    change = np.zeros(m)
    if m >= 2:
        change[1:] = np.linalg.norm(np.diff(pf, axis=0), axis=1)
        med = float(np.median(change[1:])) or 1e-9
        keep = {j for j in range(1, m)
                if change[j] > SECTION_SPLIT * med or abs(p_energy[j] - p_energy[j - 1]) >= ENERGY_STEP}
    else:
        keep = set()

    def split_long(a: int, b: int) -> None:  # phrases [a, b)
        if b - a <= MAX_SECTION_PHRASES:
            return
        j = max(range(a + 1, b), key=lambda j: change[j])
        keep.add(j)
        split_long(a, j)
        split_long(j, b)

    cuts = [0] + sorted(keep) + [m]
    for a, b in zip(cuts, cuts[1:]):
        split_long(a, b)
    cuts = [0] + sorted(keep) + [m]

    # a sliver after the last boundary (the song's final hit) joins the section before it
    phrase_len = PHRASE_BARS * BEATS_PER_BAR * float(np.median(np.diff(beats)))
    if len(cuts) > 2 and duration - p_start[cuts[-2]] < 0.5 * phrase_len:
        cuts.pop(-2)

    groups = list(zip(cuts, cuts[1:]))
    starts = [round(p_start[a], 4) for a, _ in groups]
    ends = starts[1:] + [round(duration, 4)]
    raw = [_span_mean(rms, hop, a, b) for a, b in zip(starts, ends)]
    top = max(raw) if raw and max(raw) > 0 else 1.0
    energy = [e / top for e in raw]
    melo = [_span_mean(mel, hop, a, b) if mel is not None else None for a, b in zip(starts, ends)]
    feats = [pf[a:b].mean(axis=0) for a, b in groups]
    labels = _labels(energy, melo, feats)
    return [Section(start_seconds=a, end_seconds=b, energy=round(e, 3), label=lab)
            for a, b, e, lab in zip(starts, ends, energy, labels)]


def _labels(energy: list[float], melody: list[float | None], feats: list[np.ndarray]) -> list[str | None]:
    """Labels only where the shape makes them obvious; everything else None.

    intro / outro: a quiet first / last section. break: a quiet one between
          (energy alone can't tell a softer verse from a break, so only clearly
          quiet ones).
    drop: a loud section that the music lifts into (>= LIFT above the section
          before it) with little lead melody: the beat is the hook.
    chorus: a loud section with lead melody that either follows such a lift
          or comes back after something quieter (sounds like another loud
          section, with a section at least LIFT quieter between them).
    A loud section right after a chorus / drop that sounds the same keeps its
    label (one long chorus split by the length cap).
    """
    n = len(energy)
    labels: list[str | None] = [None] * n
    if n < 3:
        return labels
    for i, e in enumerate(energy):
        if i == 0 and e < QUIET_SECTION:
            labels[i] = "intro"
        elif i == n - 1 and e < QUIET_SECTION:
            labels[i] = "outro"
        elif 0 < i < n - 1 and e < BREAK_SECTION:
            labels[i] = "break"

    loud = [i for i, e in enumerate(energy) if e >= LOUD_SECTION and labels[i] is None]
    if not loud:
        return labels
    dists = [float(np.linalg.norm(feats[i] - feats[j])) for i in range(n) for j in range(i + 1, n)]
    alike = REPEAT_SIMILARITY * float(np.median(dists)) if dists else 0.0
    def returns(i: int) -> bool:
        """Another loud section sounds like this one, with something at
        least LIFT quieter in between (a chorus comes back after a verse)."""
        for j in loud:
            if abs(i - j) <= 1 or float(np.linalg.norm(feats[i] - feats[j])) > alike:
                continue
            a, b = sorted((i, j))
            if min(energy[a + 1:b]) <= energy[i] - LIFT:
                return True
        return False

    for i in loud:
        lifted = i > 0 and energy[i] - energy[i - 1] >= LIFT
        mel = melody[i]
        if lifted and mel is not None and mel < MELODIC_SECTION:
            labels[i] = "drop"
        elif (lifted or returns(i)) and (mel is None or mel >= MELODIC_SECTION):
            labels[i] = "chorus"
    # a long chorus or drop split by the 16-bar cap: the loud part right after
    # it that sounds the same keeps the label
    for i in range(1, n):
        if (labels[i] is None and labels[i - 1] in ("chorus", "drop") and energy[i] >= LOUD_SECTION
                and float(np.linalg.norm(feats[i] - feats[i - 1])) <= alike):
            labels[i] = labels[i - 1]
    return labels


def phrases_and_sections(
    power: np.ndarray, sr: int, beats: list[float], duration: float,
    energy_rms: list[float], hop_seconds: float, melody: list[float] | None = None,
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

    sections = _sections(bar_feats, phrase_bars, phrase_starts, beats, duration,
                         energy_rms, hop_seconds, melody)
    if sections and max(s.end_seconds - s.start_seconds for s in sections) > 0:
        n_lab = sum(1 for s in sections if s.label)
        notes.append(f"{len(sections)} sections, {n_lab} labeled")
    return phrase_starts, sections, notes


def analyze_cues(
    y: np.ndarray, sr: int, beats: list[float], duration: float,
    energy_rms: list[float], hop_seconds: float,
) -> dict:
    """melody_activity, phrase_starts_seconds, sections and notes for one track."""
    power = _spectrum(y, sr)
    melody = melody_activity(power, sr, hop_seconds, len(energy_rms))
    phrases, sections, notes = phrases_and_sections(power, sr, beats, duration, energy_rms,
                                                    hop_seconds, melody)
    return {
        "melody_activity": melody,
        "phrase_starts_seconds": phrases,
        "sections": sections,
        "notes": notes,
    }
