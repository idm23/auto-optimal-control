"""Shared pieces of the benchmark animations: palette, recording, writing."""

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


def write(fig, draw, plan, path, fps, stills_dir=None, prefix="", still_frames=None):
    """Render ``draw(frame)`` over ``plan`` to ``path`` (.mp4 via ffmpeg, .gif via Pillow).

    With ``stills_dir``, first save one PNG per stage (the first element of
    each frame): by default its last frame, or ``still_frames[stage]``.
    ``draw`` must reset its artists when the stage changes.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

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
