"""Viewer and capture plumbing shared by the RoboChess Genesis scripts.

Genesis' rendering stack is smaller than Newton's and shaped differently, so this module is
a *reduction* of ``robochess_newton.viewer_utils`` rather than a translation of it.  What
Genesis 1.3.3 actually offers:

* an interactive OpenGL **viewer** (``gs.Scene(show_viewer=True)``), which raises without a
  display rather than falling back;
* an offscreen **camera** through the vendored pyrender rasteriser, which on Linux already
  defaults to hardware EGL -- no ``PYOPENGL_PLATFORM`` to set, no pyglet shadow window, and
  no ``--headless`` distinction to make.  ``scene.visualizer.is_software`` reports whether
  it landed on llvmpipe instead, and :func:`describe_renderer` prints it;
* ``camera.start_recording`` / ``stop_recording``, which encode an mp4 with PyAV **from
  inside ``scene.step()``** at a cadence Genesis picks from ``dt``.

The last one is why :class:`FrameCapture` renders and encodes itself instead: this port
drives four 1/240 substeps per rendered frame and two rendered frames per control tick, so
"one video frame every N whole ``scene.step()`` calls" does not line up with anything, and
Genesis would silently re-time the video ("fps=30 is not reachable with dt=0.00417.
Recording at fps=... instead").  Rendering the camera explicitly on the frames we choose
gives the same PNG/MP4 pair the Newton port produces, at the frame rate that was asked for.

**Not available, and not faked.** Genesis has no USD, glTF or JSON scene export of any kind
(grep-verified across the package), so the Newton port's ``--viewer usd|file|rerun|viser``
have no counterpart -- there is no way to write a stage you can open elsewhere.  Its
ray-tracing backend needs LuisaRender built from source, which is not present.  The
``--viewer`` choices below are therefore the three that exist, and asking for one of the
Newton names says so instead of quietly doing something else.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

__all__ = [
    "VIEWER_CHOICES",
    "FrameCapture",
    "add_viewer_args",
    "camera_spec_for",
    "describe_renderer",
    "has_display",
    "resolve_viewer",
    "run_frame_loop",
]

VIEWER_CHOICES = ("auto", "gl", "window", "null")
"""``gl`` renders offscreen, ``window`` opens the interactive viewer, ``null`` renders
nothing, ``auto`` picks between them from ``$DISPLAY`` and the requested output."""

_NEWTON_ONLY_VIEWERS = {
    "usd": "Genesis has no USD export; there is no stage to write.",
    "file": "Genesis has no JSON/binary scene recorder.",
    "rerun": "Genesis has no rerun backend.",
    "viser": "Genesis has no viser backend.",
}


def has_display() -> bool:
    """True when an X/Wayland server is reachable, i.e. an interactive window can open."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def add_viewer_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the viewer/capture CLI flags. Returns *parser* for chaining.

    The names are the Newton port's wherever the flag survives the move, so a command
    written for one port mostly runs on the other.
    """
    parser.add_argument(
        "--viewer",
        default="auto",
        choices=list(VIEWER_CHOICES) + list(_NEWTON_ONLY_VIEWERS),
        metavar="{" + ",".join(VIEWER_CHOICES) + "}",
        help="How to render; 'auto' picks window/gl/null from $DISPLAY and the requested output.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Genesis backend: 'gpu' (default), 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=300,
        help="Frames to simulate and render.",
    )
    parser.add_argument("--save-images", default=None, metavar="DIR", help="Write one PNG per frame.")
    parser.add_argument("--save-video", default=None, metavar="FILE", help="Write an mp4.")
    parser.add_argument("--width", type=int, default=1280, help="Render width.")
    parser.add_argument("--height", type=int, default=720, help="Render height.")
    parser.add_argument("--fps", type=int, default=60, help="Frame rate of the render and the mp4.")
    parser.add_argument("--zoom", type=float, default=1.0, help="Camera distance divisor.")
    # A negative x reads as an option, so these need --camera=-1,0,2 rather than a space.
    parser.add_argument("--camera", type=vec3, default=None, help="Camera eye position '--camera=x,y,z'.")
    parser.add_argument(
        "--camera-target", type=vec3, default=None, help="Camera look-at point '--camera-target=x,y,z'."
    )
    return parser


def vec3(text: str) -> tuple[float, float, float]:
    parts = [float(item) for item in text.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'x,y,z', got {text!r}")
    return parts[0], parts[1], parts[2]


def resolve_viewer(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    """Turn ``--viewer`` into one of ``gl``/``window``/``null``, or refuse with a reason."""
    if args.viewer in _NEWTON_ONLY_VIEWERS:
        parser.error(
            f"--viewer {args.viewer} is a newton-port backend and has no Genesis equivalent: "
            f"{_NEWTON_ONLY_VIEWERS[args.viewer]} Use --viewer gl with --save-images/--save-video, "
            f"--viewer window on a display, or --viewer null."
        )
    if args.viewer != "auto":
        if args.viewer == "window" and not has_display():
            parser.error(
                "--viewer window needs a display and $DISPLAY/$WAYLAND_DISPLAY is unset; "
                "Genesis' viewer has no offscreen mode. Use --viewer gl --save-images/--save-video."
            )
        if args.viewer == "null" and (args.save_images or args.save_video):
            parser.error("--viewer null renders nothing, so --save-images/--save-video cannot be honoured")
        if args.viewer == "gl" and not (args.save_images or args.save_video):
            # gl is the offscreen path: it draws into a camera nobody reads unless one of
            # these is set. Refusing beats spending the render budget on discarded frames.
            parser.error(
                "--viewer gl renders offscreen, so it needs somewhere to put the frames; "
                "add --save-images DIR and/or --save-video FILE.mp4 (or use --viewer null)"
            )
        return args.viewer
    if args.save_images or args.save_video:
        return "gl"
    return "window" if has_display() else "null"


def writable_path(parser: argparse.ArgumentParser, flag: str, path: str) -> str:
    """*path* made absolute, refused now if nothing can be created there.

    Absolute so that every path a run prints is one the reader can open from anywhere;
    checked up front because ``makedirs`` inside the capture raises ``PermissionError``
    halfway through an otherwise good run.
    """
    target = Path(path).expanduser().resolve()
    anchor = next(parent for parent in (target, *target.parents) if parent.is_dir())
    if not os.access(anchor, os.W_OK | os.X_OK):
        parser.error(f"{flag} {target}: {anchor} is not writable")
    return str(target)


def camera_spec_for(args: argparse.Namespace, viewer: str):
    """The :class:`~robochess_genesis.scene.CameraSpec` this run needs, or ``None``.

    Keyed on **whether frames are being saved**, not on which viewer was chosen.  Keying it
    on ``viewer == "gl"`` is the obvious thing and it is wrong: on a machine with a display,
    ``--viewer window --save-video out.mp4`` would then build no offscreen camera, every
    ``grab()`` would short-circuit on a ``None`` frame, and the run would finish and exit 0
    having written nothing and warned about nothing.  Genesis is happy to have a camera and
    a viewer at once (``Visualizer`` builds the rasteriser around both), so the honest
    reading of "show me a window *and* record it" is to do both.
    """
    from .scene import CameraSpec

    if not (args.save_images or args.save_video):
        return None
    return CameraSpec(
        res=(args.width, args.height),
        eye=args.camera,
        target=args.camera_target,
        zoom=args.zoom,
    )


def describe_renderer(scene: Any) -> str:
    """One line about how this run will actually draw, for the scripts to print."""
    visualizer = scene.gs_scene.visualizer
    if scene.camera is None and scene.gs_scene.viewer is None:
        return "  render: none (no camera, no viewer)"
    kind = "window" if scene.gs_scene.viewer is not None else "offscreen"
    software = getattr(visualizer, "is_software", None)
    backend = "llvmpipe/software" if software else "hardware EGL"
    resolution = scene.camera_spec.res if scene.camera_spec else "viewer"
    return f"  render: {kind} rasteriser ({backend}) res={resolution}"


class FrameCapture:
    """Writes the scene camera's frames to PNGs and/or an mp4.

    Rendering is explicit rather than delegated to ``camera.start_recording`` -- see the
    module docstring.  The mp4 needs an ffmpeg backend for imageio; without one the PNG
    directory is still written and a warning names what is missing, which is the behaviour
    the Newton port has for the same situation.
    """

    def __init__(
        self,
        scene: Any,
        image_dir: str | None = None,
        video_path: str | None = None,
        fps: int = 60,
    ) -> None:
        self.scene = scene
        self.image_dir = image_dir
        self.video_path = video_path
        self.fps = fps
        self.frames = 0
        self._writer = None
        self._image_writer: Callable[[str, np.ndarray], None] | None = None
        self.outputs: list[str] = []

        if image_dir:
            Path(image_dir).mkdir(parents=True, exist_ok=True)
        if video_path:
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            self._writer = self._open_video(video_path, fps)
            if self._writer is None:
                # Degrade to PNGs rather than silently producing nothing, and say where.
                self.image_dir = image_dir or str(Path(video_path).with_suffix("")) + "_frames"
                Path(self.image_dir).mkdir(parents=True, exist_ok=True)
                print(
                    f"[WARN] no ffmpeg backend for imageio, so {video_path} cannot be written; "
                    f"writing PNGs to {self.image_dir} instead. Install imageio-ffmpeg to get the mp4.",
                    flush=True,
                )

    @staticmethod
    def _open_video(path: str, fps: int):
        try:
            import imageio.v2 as imageio

            return imageio.get_writer(path, fps=fps, macro_block_size=1)
        except Exception:
            return None

    def _write_image(self, path: str, frame: np.ndarray) -> None:
        if self._image_writer is None:
            import imageio.v2 as imageio

            self._image_writer = imageio.imwrite
        self._image_writer(path, frame)

    def grab(self) -> None:
        """Render one frame and file it. Call once per simulated frame."""
        frame = self.scene.render()
        if frame is None:
            return
        frame = np.asarray(frame)
        if self.image_dir:
            self._write_image(str(Path(self.image_dir) / f"frame_{self.frames:05d}.png"), frame)
        if self._writer is not None:
            self._writer.append_data(frame)
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self.outputs.append(self.video_path)
        if self.image_dir and self.frames:
            self.outputs.append(self.image_dir)

    def report(self) -> None:
        for path in self.outputs:
            if os.path.exists(path):
                print(f"[output] wrote {path}", flush=True)


def run_frame_loop(scene: Any, num_frames: int, capture: FrameCapture | None) -> int:
    """Step *scene* for *num_frames* frames, capturing each one. Returns frames stepped.

    Stops early if an interactive viewer is closed, which is the only stop condition
    Genesis itself supplies.
    """
    viewer = scene.gs_scene.viewer
    for frame in range(num_frames):
        if viewer is not None and not viewer.is_alive():
            return frame
        scene.step()
        if capture is not None:
            capture.grab()
    return num_frames
