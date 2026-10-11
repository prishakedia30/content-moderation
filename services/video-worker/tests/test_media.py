"""FFmpeg-level checks on a generated clip: audio detection and that muting really silences the span."""
import shutil
import subprocess

import pytest

from pipeline import has_audio, mute
from spans import Span

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def make_clip(path, audio=True):
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=3"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)


def mean_volume(path, start, dur):
    out = subprocess.run(["ffmpeg", "-ss", str(start), "-t", str(dur), "-i", str(path),
                          "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True).stderr
    return float(out.split("mean_volume:")[1].split("dB")[0])


def test_has_audio(tmp_path):
    make_clip(tmp_path / "a.mp4", audio=True)
    make_clip(tmp_path / "b.mp4", audio=False)
    assert has_audio(tmp_path / "a.mp4") and not has_audio(tmp_path / "b.mp4")


def test_mute_silences_only_the_span(tmp_path):
    make_clip(tmp_path / "in.mp4")
    mute(tmp_path / "in.mp4", [Span(1.0, 2.0, "x", 1.0)], tmp_path / "out.mp4")
    assert mean_volume(tmp_path / "out.mp4", 1.2, 0.6) < -80     # muted
    assert mean_volume(tmp_path / "out.mp4", 0.2, 0.6) > -30     # untouched


def test_no_spans_copies(tmp_path):
    make_clip(tmp_path / "in.mp4")
    mute(tmp_path / "in.mp4", [], tmp_path / "out.mp4")
    assert (tmp_path / "out.mp4").stat().st_size > 0
