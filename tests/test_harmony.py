import numpy as np
import pytest

from backend.audio.harmony import Chroma, compatibility
from backend.logic import windows as W
from test_windows import make_track

C_MAJOR = np.array([1, 0, 0.2, 0, 0.8, 0.3, 0, 0.9, 0, 0.2, 0, 0.3])
F_SHARP = np.roll(C_MAJOR, 6)  # a tritone away: the most distant key


def test_compatibility_is_high_in_key_and_low_a_tritone_away():
    assert compatibility(C_MAJOR, C_MAJOR) == pytest.approx(1.0)
    assert compatibility(C_MAJOR, np.roll(C_MAJOR, 7)) > compatibility(C_MAJOR, F_SHARP)  # fifth vs tritone
    assert compatibility(C_MAJOR, F_SHARP) < 0.4
    assert compatibility(None, C_MAJOR) is None
    assert compatibility(np.ones(12), C_MAJOR) is None  # flat chroma says nothing


def chroma(duration, spans, hop=0.5):
    """Chroma series in F# except the (start, end) spans, which are in C."""
    frames = np.tile(F_SHARP, (int(duration / hop), 1))
    for t0, t1 in spans:
        frames[int(t0 / hop):int(t1 / hop)] = C_MAJOR
    return Chroma(frames.astype(np.float32), hop)


def test_seam_prefers_the_part_of_b_in_key_with_a():
    a, b = make_track("a"), make_track("b")
    ca, cb = W._cues(a), W._cues(b)
    ca.chroma = chroma(192, [(0, 192)])  # A in C throughout
    cb.chroma = chroma(192, [(64, 96)])  # B in C only from 64 to 96 s
    seam = W._score_seam(ca, cb, 8.0)
    end = int(np.flatnonzero(ca.grid == 192.0)[0])
    best = cb.grid[int(np.argmax(seam.harmony[end]))]
    assert 64 <= best < 96
    assert seam.harmony[end].min() < 0.4


def test_choose_windows_enters_where_the_notes_line_up():
    loud = [(96, 192, 1.0)]  # B's best section starts at 96
    tracks = [make_track("a"), make_track("b", energy=loud)]
    chromas = [chroma(192, [(0, 192)]), chroma(192, [(64, 80)])]  # in key only from 64 to 80
    wa, wb = W.choose_windows(tracks, [8.0], 120.0, chromas)
    assert wb.start_seconds == 64.0
    assert "notes line up" in wa.reason
    plain = W.choose_windows(tracks, [8.0], 120.0)
    assert plain[1].start_seconds != 64.0
    assert all("harmonic match" not in w.reason for w in plain)
