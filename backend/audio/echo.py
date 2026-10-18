"""Echo-out for seams that can't be blended (owner: Aarav).

Two songs whose tempos are too far apart to beat-match (and that both have a
beat at the seam) would trainwreck in any long blend; the same goes for two
melodies at once when the beats can't be matched for a vocal swap. A DJ
does not blend those: they let A finish its phrase, echo it out and drop B
on a downbeat. That is what this does:

  * A plays its last bars dry (ECHO_BARS, picked from the requested length),
    with its lows (<200 Hz) ramped out across them so B's kick lands on a
    clean low end.
  * A's last beat is sent to a 1-beat delay (ECHO_REPEATS repeats for one
    bar, two more per doubling, each ECHO_FEEDBACK quieter, high-passed at
    ECHO_HIGHPASS_HZ so the echo has no bass). The first repeat lands on B's
    first beat and the tail rings under B's opening: no silent beat.
  * B starts so its first beat lands where A's last bar ends. If B has more
    than a beat of lead-in before that beat, B starts right at the bar line
    instead, so its lead-in never plays over A.

Every sample of both tracks still plays: A up to its last sample, B from
its sample 0. Only the echo is added.

The melody decision (melody_clash) looks at fixed stretches of each side
(CLASH_SECONDS before A's fade tail, after B's fade head), not at the
requested overlap, so both A/B previews of a seam always get the same
treatment; only the echo's length follows the request.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt

from backend.audio.bands import split_bands
from backend.audio.seam_plan import SeamPlan

CLASH_SECONDS = 8.0  # how much of each side's edge is checked for melody
MELODY_CLASH = 0.5  # both edges at or above this mean melody_activity: echo out
ECHO_REPEATS = 4  # for a 1-bar filter-out; 2 more per doubling of the bars
ECHO_FEEDBACK = 0.5  # each repeat 6 dB quieter (1 bar); longer tails decay more slowly
ECHO_BARS = (1, 2, 4)  # outgoing bars filtered out, picked from the requested length
ECHO_WET = 0.7
ECHO_HIGHPASS_HZ = 300.0
BAR_BEATS = 4
END_FADE_SECONDS = 0.02  # anti-click fade on A's very last samples (a window may end mid-note)
START_FADE_SECONDS = 0.005  # and on B's first samples
DEFAULT_PERIOD = 0.5  # seconds per beat when a side has no beat grid


def _edge_mean(melody: Sequence[float], hop: float, t0: float, t1: float) -> float | None:
    if not melody or hop <= 0 or t1 <= t0:
        return None
    i0, i1 = max(0, int(t0 / hop)), int(np.ceil(t1 / hop))
    seg = np.asarray(melody[i0:i1], dtype=np.float64)
    return float(seg.mean()) if seg.size else None


def melody_clash(
    out_melody: Sequence[float] | None, in_melody: Sequence[float] | None, hop: float | None,
    out_dur: float, tail: float = 0.0, head: float = 0.0,
) -> tuple[bool, float | None, float | None]:
    """(clash, outgoing edge melody, incoming edge melody). No data: no clash."""
    if not out_melody or not in_melody or not hop:
        return False, None, None
    end = out_dur - tail
    a = _edge_mean(out_melody, hop, max(0.0, end - CLASH_SECONDS), end)
    b = _edge_mean(in_melody, hop, head, head + CLASH_SECONDS)
    if a is None or b is None:
        return False, a, b
    return a >= MELODY_CLASH and b >= MELODY_CLASH, a, b


@dataclass
class EchoPlan(SeamPlan):
    offset: int = 0  # seam sample where incoming sample 0 plays
    period: float = DEFAULT_PERIOD  # outgoing beat period, seconds
    bars: int = 1  # outgoing bars with their lows ramped out
    repeats: int = ECHO_REPEATS
    feedback: float = ECHO_FEEDBACK


def echo_bars(requested: float | None, period: float) -> int:
    """Outgoing bars to filter out for a requested length: the closest of
    ECHO_BARS (1 without a request)."""
    if not requested or requested <= 0:
        return 1
    return min(ECHO_BARS, key=lambda b: abs(b * BAR_BEATS * period - requested))


@lru_cache(maxsize=4)
def _highpass(sr: int):
    return butter(4, ECHO_HIGHPASS_HZ, btype="high", fs=sr, output="sos")


def last_bar_end(out_dur: float, grid) -> tuple[float, float]:
    """(time of A's final bar line, beat period). With a grid, the beat nearest
    the end (a window cut sits on a phrase start, i.e. a beat); without one,
    the end itself."""
    if grid is None:
        return out_dur, DEFAULT_PERIOD
    p = grid.period
    k = (out_dur - grid.first) / p
    k = round(k) if abs(k - round(k)) < 0.1 else np.floor(k)
    end = grid.first + k * p
    if end > out_dur or end < p:
        end = out_dur
    return float(end), p


def plan_echo_out(out_len: int, in_len: int, sr: int, g_out, in_first_beat: float | None,
                  max_in_fraction: float = 0.5, requested: float | None = None,
                  in_head: float = 0.0) -> EchoPlan:
    """requested: seconds of the outgoing song to filter out (echo_bars); a
    longer filter-out gets a longer echo tail under the incoming song.
    in_head: the incoming song's quiet lead-in (edges.py), played under the
    outgoing bar so its sound starts on the bar line."""
    end_t, p = last_bar_end(out_len / sr, g_out)
    end = int(round(end_t * sr))
    bars = echo_bars(requested, p)
    while bars > 1 and bars * BAR_BEATS * p > max_in_fraction * out_len / sr:
        bars //= 2
    repeats = ECHO_REPEATS + 2 * (bars.bit_length() - 1)
    feedback = ECHO_FEEDBACK ** (ECHO_REPEATS / repeats)  # the tail fades to the same level
    bar = int(round(bars * BAR_BEATS * p * sr))
    o_start = max(0, end - bar)
    out_part = out_len - o_start
    bar_line = end - o_start  # seam sample where A's last bar ends and B's downbeat lands
    first = int(round((in_first_beat or 0.0) * sr))
    if first <= int(round(p * sr)):
        offset = max(0, bar_line - first)
        where = "first beat on A's last bar line"
    else:
        head = min(int(round(in_head * sr)), bar_line)
        offset = bar_line - head
        where = (f"sound starting at A's last bar line ({first / sr:.2f}s lead-in before its first beat"
                 + (f", its {head / sr:.2f}s quiet start under A)" if head else ")"))
    tail = int(round(repeats * p * sr))
    seam_len = max(out_part, bar_line + tail)
    in_cons = min(seam_len - offset, int(max_in_fraction * in_len))
    seam_len = max(out_part, offset + in_cons)
    plan = EchoPlan(kind="echo_out", o_start=o_start, in_consumed=in_cons, seam_len=seam_len,
                    beats=[bar_line], target_seconds=bar / sr, offset=offset, period=p,
                    bars=bars, repeats=repeats, feedback=feedback)
    what = f"last {bars} bars" if bars > 1 else "last bar"
    plan.notes.append(f"echo-out: outgoing plays its {what} ({bar / sr:.2f}s) with the bass "
                      f"filtered out, its last beat echoes {repeats}x on a 1-beat delay; "
                      f"incoming placed with its {where}")
    return plan


def render_echo_out(out: np.ndarray, inc: np.ndarray, sr: int, plan: EchoPlan,
                    g_out: np.float32, g_in: np.float32) -> np.ndarray:
    """Seam audio for an echo-out plan, gains applied. (seam_len, ch) float32."""
    n, ch = plan.seam_len, out.shape[1]
    p = int(round(plan.period * sr))
    bar_line = plan.beats[0]
    seam = np.zeros((n, ch), dtype=np.float64)

    a = out[plan.o_start:].astype(np.float64) * float(g_out)
    low, mid, high = split_bands(a, sr)
    ramp = np.ones(a.shape[0])
    ramp[:bar_line] = np.linspace(1.0, 0.0, bar_line) if bar_line > 0 else 1.0
    ramp[bar_line:] = 0.0
    dry = low * ramp[:, None] + mid + high
    k = min(a.shape[0], int(END_FADE_SECONDS * sr))
    if k > 1:
        dry[-k:] *= np.linspace(1.0, 0.0, k)[:, None]
    seam[:a.shape[0]] += dry

    send = a[max(0, bar_line - p):bar_line]
    if send.shape[0] > 30:
        send = sosfiltfilt(_highpass(sr), send, axis=0) * ECHO_WET
        f = max(2, send.shape[0] // 5)  # each repeat fades out over its last 20%: no click
        send[-f:] *= np.linspace(1.0, 0.0, f)[:, None]
        for r in range(plan.repeats):
            at = bar_line + r * p
            if at >= n:
                break
            m = min(send.shape[0], n - at)
            seam[at:at + m] += send[:m] * plan.feedback ** r

    off = plan.offset
    b = inc[:plan.in_consumed].astype(np.float64) * float(g_in)
    k = min(b.shape[0], int(START_FADE_SECONDS * sr))
    if k > 1:  # a window may start mid-note: no click
        b[:k] *= np.linspace(0.0, 1.0, k)[:, None]
    seam[off:off + plan.in_consumed] += b
    return seam.astype(np.float32)
