from io import StringIO
from types import SimpleNamespace

from vision_lens.pipeline import progress


class _TerminalStream(StringIO):
    def isatty(self) -> bool:
        return True


def test_image_progress_and_status_are_visible_in_a_terminal(monkeypatch):
    terminal = _TerminalStream()
    monkeypatch.setattr(progress.sys, "stderr", terminal)

    progress.status("Loading model example")
    batches = [SimpleNamespace(count=2), SimpleNamespace(count=1)]
    tracked = list(
        progress.track_image_batches(
            batches,
            total=3,
            description="Analyze images",
        )
    )
    assert tracked == batches

    output = terminal.getvalue()
    assert "vision-lens: Loading model example" in output
    assert "Analyze images" in output
    assert "3/3" in output


def test_video_progress_counts_frames_not_batches(monkeypatch):
    terminal = _TerminalStream()
    monkeypatch.setattr(progress.sys, "stderr", terminal)

    batches = [
        SimpleNamespace(frames=(1, 2)),
        SimpleNamespace(frames=(3, 4, 5)),
    ]
    tracked = list(
        progress.track_video_batches(
            batches,
            total=5,
            description="Analyze video",
        )
    )
    assert tracked == batches

    output = terminal.getvalue()
    assert "Analyze video" in output
    assert "5/5" in output


def test_noninteractive_logs_do_not_include_progress_control_characters(monkeypatch):
    output = StringIO()
    monkeypatch.setattr(progress.sys, "stderr", output)

    progress.status("Loading model example")
    list(
        progress.track_image_batches(
            [SimpleNamespace(count=1)],
            total=1,
            description="Analyze images",
        )
    )

    assert output.getvalue() == "vision-lens: Loading model example\n"


def test_progress_output_can_be_silenced_and_restored(monkeypatch):
    terminal = _TerminalStream()
    monkeypatch.setattr(progress.sys, "stderr", terminal)

    with progress.progress_output(False):
        progress.status("hidden")
        list(
            progress.track_image_batches(
                [SimpleNamespace(count=1)],
                total=1,
                description="Hidden work",
            )
        )

    progress.status("visible")

    assert terminal.getvalue() == "vision-lens: visible\n"
