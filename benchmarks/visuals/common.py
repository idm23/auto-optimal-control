"""Shared pieces of the benchmark animations: palette, recording, writing."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # this package / a reference or second case
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281",
        "#0d366b"]                              # sequential blue, light -> dark


def recording(cls):
    """A subclass of shooting class ``cls`` that keeps every point the root finder evaluates."""

    class Recording(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.evaluations = []

        def residual_jacobian(self, v):
            R, J = super().residual_jacobian(v)
            self.evaluations.append((np.array(v, dtype=float), float(np.linalg.norm(R))))
            return R, J

        __call__ = residual_jacobian

    Recording.__name__ = f"Recording{cls.__name__}"
    return Recording


@dataclass
class Problem:
    """What an animation solves: shown on a title card, then as a footer.

    ``formulation`` is a list of ``(label, mathtext)`` rows, ``solved_by``
    the steps in plain words, and ``summary`` the one-line footer.
    """

    name: str
    story: str
    formulation: list
    solved_by: list
    summary: str
    source: str = ""
    extra: dict = field(default_factory=dict)


def save(data, path):
    np.savez_compressed(path, **data)


def load(path):
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


def style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURF, "axes.facecolor": SURF})
    return plt


def header(fig):
    """Title and caption text artists at the top of a 16:9 figure."""
    title = fig.text(0.05, 0.94, "", fontsize=17, weight="bold")
    caption = fig.text(0.05, 0.85, "", fontsize=10.5, color=INK2, va="bottom")
    return title, caption


def _with_problem(fig, draw, problem):
    """Wrap ``draw`` so a ``("problem", 0)`` frame shows the title card, and
    add the footer. Everything else on the figure is hidden behind the card."""
    footer = fig.text(0.05, 0.008, "Problem: " + problem.summary, fontsize=9, color=INK2)
    card = [fig.text(0.07, 0.86, problem.name, fontsize=24, weight="bold"),
            fig.text(0.07, 0.80, problem.story, fontsize=12.5, color=INK2, va="top")]
    y = 0.62
    card.append(fig.text(0.07, y + 0.05, "THE PROBLEM", fontsize=10, color=MUTED, weight="bold"))
    for label, math in problem.formulation:
        card.append(fig.text(0.07, y, label, fontsize=12, color=INK2, va="center"))
        card.append(fig.text(0.22, y, math, fontsize=14, va="center"))
        y -= 0.068
    y -= 0.02
    card.append(fig.text(0.07, y, "HOW IT IS SOLVED", fontsize=10, color=MUTED, weight="bold"))
    y -= 0.05
    for i, step in enumerate(problem.solved_by, 1):
        card.append(fig.text(0.07, y, f"{i}.  {step}", fontsize=11.5, color=INK2, va="top"))
        y -= 0.032 * (step.count("\n") + 1) + 0.016
    if problem.source:
        card.append(fig.text(0.07, 0.04, problem.source, fontsize=9, color=MUTED))
    for a in card:
        a.set_visible(False)
    hidden = []

    def wrapped(frame):
        if frame[0] == "problem":
            if not hidden:
                for a in fig.axes + [t for t in fig.texts if t not in card]:
                    if a.get_visible():
                        hidden.append(a)
                        a.set_visible(False)
                for a in card:
                    a.set_visible(True)
            return ()
        if hidden or card[0].get_visible():
            for a in hidden:
                a.set_visible(True)
            hidden.clear()
            for a in card:
                a.set_visible(False)
            footer.set_visible(True)
        return draw(frame)

    return wrapped


def write(fig, draw, plan, path, fps, stills_dir=None, prefix="", still_frames=None,
          problem=None, card_seconds=6):
    """Render ``draw(frame)`` over ``plan`` to ``path`` (.mp4 via ffmpeg, .gif via Pillow).

    With ``stills_dir``, first save one PNG per stage (the first element of
    each frame): by default its last frame, or ``still_frames[stage]``.
    ``draw`` must reset its artists when the stage changes. A
    :class:`Problem` opens the animation with a title card for
    ``card_seconds`` (stage ``"problem"``) and stays as a footer.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    if problem is not None:
        draw = _with_problem(fig, draw, problem)
        plan = [("problem", 0)] * int(card_seconds * fps) + list(plan)
        still_frames = {key: j + int(card_seconds * fps)
                        for key, j in (still_frames or {}).items()}
    path = Path(path)
    if stills_dir is not None:
        stills_dir = Path(stills_dir)
        stills_dir.mkdir(parents=True, exist_ok=True)
        ends = {}
        for j, frame in enumerate(plan):
            ends[frame[0]] = j
        ends.update(still_frames or {})
        for stage, j in ends.items():
            draw(plan[j])
            fig.savefig(stills_dir / f"{prefix}{stage}.png")
        draw(("__reset__", 0))
    writer = (animation.PillowWriter(fps=fps) if path.suffix == ".gif"
              else animation.FFMpegWriter(fps=fps, bitrate=4000))
    anim = animation.FuncAnimation(fig, draw, frames=plan, blit=False, cache_frame_data=False)
    anim.save(path, writer=writer)
    plt.close(fig)
    return path
