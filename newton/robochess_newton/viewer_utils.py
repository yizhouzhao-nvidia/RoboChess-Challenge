"""Viewer plumbing shared by the RoboChess Newton scripts.

Newton ships its own CLI (``newton.examples.create_parser``) and frame loop
(``newton.examples.run``), but neither survives contact with this box:

* ``newton.examples.run`` loops on ``viewer.is_running()``, and only
  ``ViewerNull`` / ``ViewerUSD`` override it -- for ``gl``/``rerun``/``viser``/
  ``file`` it is hardwired ``True``, so ``--num-frames`` is ignored and a
  headless run never terminates.  ``run_viewer_loop`` below counts frames
  itself.
* ``ViewerGL`` cannot start without ``DISPLAY``: newton's own ``--headless``
  only hides the window, pyglet still opens an X shadow window.  Offscreen
  rendering needs ``pyglet.options["headless"] = True`` (EGL), which has to be
  set before pyglet creates a window -- hence the lazy imports here.

The flag names are copied verbatim from ``newton.examples.create_parser`` so the
scripts feel native to anyone who has run a newton example, plus four additions
(``--save-images``, ``--save-video``, ``--width/--height``) that turn a headless
box into a source of PNG/MP4 artifacts.

The module is import-time cheap -- its one package import is
:mod:`~robochess_newton.board_layout`, which pulls in nothing beyond the standard
library and the lab's ``board.py`` -- and it touches neither warp nor newton until
:func:`make_viewer` runs, so a caller can still choose a warp device or set
environment variables after importing it.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

# board_layout first, and unconditionally: importing it installs the package's two
# import-time guards -- PXR_WORK_THREAD_LIMIT and the sys.path shield. make_viewer
# imports newton.viewer, which puts pxr in sys.modules on both newton versions, so
# this module is as much a pxr entry point as the ones that load assets, and it is
# the one a notebook or a test is most likely to import first.
from . import board_layout as bl

if TYPE_CHECKING:
    from newton.viewer import ViewerBase

__all__ = [
    "VIEWER_CHOICES",
    "FrameCapture",
    "SimExample",
    "add_viewer_args",
    "enable_egl_headless",
    "ensure_newton_importable",
    "frame_time",
    "has_display",
    "make_viewer",
    "run_viewer_loop",
]

VIEWER_CHOICES = ("auto", "gl", "usd", "rerun", "viser", "file", "null")


class SimExample(Protocol):
    """The two methods :func:`run_viewer_loop` needs; matches newton's examples."""

    def step(self) -> None: ...

    def render(self) -> None: ...


def ensure_newton_importable() -> None:
    """Drop the repository root from ``sys.path`` so ``import newton`` finds the real one.

    Delegates to :func:`~robochess_newton.board_layout.shield_newton_imports`, which
    already ran when this module was imported.  Kept under this name because it is
    what the scripts and the notes call, and because :func:`make_viewer` runs it
    again once the caller has had every chance to put the root back.  Idempotent.

    Today's layout is not itself broken: ``<repo>/newton/`` has no ``__init__.py``,
    so PEP 420 makes it a namespace *portion*, which does not stop the path scan,
    and the regular package in site-packages still wins (measured on newton 1.2.1
    and 1.6.0.dev0).  The hazard is a future ``newton/__init__.py``: that would be a
    regular package, it would beat site-packages outright, and ``import
    newton.viewer`` would then fail with a bewildering ``ModuleNotFoundError``
    against a ``newton`` that imported fine.  Dropping the entry costs nothing and
    closes that door.
    """
    bl.shield_newton_imports()


def has_display() -> bool:
    """True when an X/Wayland server is reachable, i.e. an interactive window can open."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def enable_egl_headless() -> None:
    """Switch pyglet to its EGL backend.  Must run before pyglet opens any window."""
    import pyglet

    pyglet.options["headless"] = True


def frame_time(frame_index: int, fps: int) -> float:
    """Exact simulation time of frame *frame_index*.

    ``ViewerUSD.begin_frame`` derives its timecode as ``int(time * fps)``.  An
    accumulated ``sim_time += 1/fps`` loses frames to float error (measured: 86
    distinct timecodes out of 90).  Examples should render at ``frame_time(i,
    fps)``; :func:`make_viewer` additionally forces this on the USD viewer.
    """
    return frame_index / fps


def add_viewer_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the viewer/capture CLI flags.  Returns *parser* for chaining."""
    parser.add_argument(
        "--viewer",
        default="auto",
        choices=list(VIEWER_CHOICES),
        help="Backend to render with; 'auto' picks gl/usd from $DISPLAY and the requested output.",
    )
    parser.add_argument("--device", default=None, help="Warp device, e.g. 'cuda:0' or 'cpu'.")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=300,
        help="Frames to simulate and render. 0 runs until an interactive viewer (gl/viser/rerun) is closed, and is capped at 1000 frames for the non-interactive ones.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force (--headless) or forbid (--no-headless) offscreen EGL GL; default: auto from $DISPLAY.",
    )
    parser.add_argument("--output-path", default=None, help="Artifact written by the usd/file/rerun/viser viewers.")
    parser.add_argument("--save-images", default=None, metavar="DIR", help="Write one PNG per frame (gl only).")
    parser.add_argument("--save-video", default=None, metavar="FILE", help="Write an mp4 (gl only).")
    parser.add_argument("--width", type=int, default=1280, help="GL framebuffer width.")
    parser.add_argument("--height", type=int, default=720, help="GL framebuffer height.")
    parser.add_argument("--fps", type=int, default=60, help="Frame rate of the render/USD timeline/mp4.")
    parser.add_argument("--viser-port", type=int, default=8080, help="Port for --viewer viser.")
    parser.add_argument("--rerun-address", default=None, help="gRPC address for --viewer rerun.")
    return parser


def _log(message: str) -> None:
    print(f"[viewer] {message}", file=sys.stderr)


DEFAULT_OUTPUT_DIR = Path(os.environ.get("ROBOCHESS_OUT_DIR") or Path(tempfile.gettempdir()) / "robochess-newton")
"""Where an artifact goes when ``--output-path`` is omitted.

Deliberately not a relative ``out/``: the viewer is built from whatever working
directory the caller happens to be in, and dropping an untracked directory into
someone's repo (or worse, failing on a read-only cwd) is not the library's business.
Overridable with ``$ROBOCHESS_OUT_DIR``; the scripts share this default.
"""


def default_output_path(stem: str, suffix: str) -> str:
    """Absolute path for an artifact named *stem* with *suffix*, under :data:`DEFAULT_OUTPUT_DIR`."""
    return str(DEFAULT_OUTPUT_DIR / f"{stem}{suffix}")


def _prepare_output(path: str) -> str:
    """Make sure *path*'s parent directory exists; return the path unchanged."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    return path


def _resolve_kind(args: argparse.Namespace, wants_capture: bool) -> str:
    """The auto policy: interactive if we can, artifact-producing otherwise."""
    if args.viewer != "auto":
        return args.viewer
    if has_display():
        return "gl"
    # No display: GL is only worth paying for when we read frames back off it.
    return "gl" if wants_capture else "usd"


def _force_exact_timecodes(viewer: Any, fps: int) -> None:
    """Drive ``ViewerUSD``'s timecode off a frame counter instead of the sim clock.

    See :func:`frame_time` -- this makes the USD timeline dense (0..N-1) even for
    an example that accumulates its own drifting ``sim_time``.
    """
    original = viewer.begin_frame
    counter = itertools.count()

    def begin_frame(time: float = 0.0) -> Any:
        return original(next(counter) / fps)

    viewer.begin_frame = begin_frame


def make_viewer(
    args: argparse.Namespace,
    default_stem: str = "scene",
    exact_timecodes: bool = True,
) -> tuple[ViewerBase, FrameCapture | None]:
    """Build the viewer *args* asks for.

    Returns ``(viewer, capture)``; *capture* is a :class:`FrameCapture` when
    ``--save-images``/``--save-video`` were requested and the GL viewer came up,
    else ``None``.  *default_stem* names the artifact when ``--output-path`` is
    omitted; it lands in :data:`DEFAULT_OUTPUT_DIR`, never the cwd.  Pass the result straight to
    :func:`run_viewer_loop`, which owns closing both.
    """
    ensure_newton_importable()

    wants_capture = bool(args.save_images or args.save_video)
    kind = _resolve_kind(args, wants_capture)

    headless = args.headless
    if headless is None:
        headless = not has_display()
    if kind == "gl" and headless:
        # Must precede the first pyglet window, i.e. ViewerGL.__init__.
        enable_egl_headless()

    import warp as wp

    if args.device:
        wp.set_device(args.device)

    import newton.viewer as nv

    if kind == "gl":
        try:
            viewer = nv.ViewerGL(width=args.width, height=args.height, headless=bool(headless))
        except Exception as exc:  # NoSuchDisplayException, EGL init failure, ...
            _log(f"ViewerGL unavailable ({type(exc).__name__}: {exc}); falling back to ViewerUSD")
            kind = "usd"
        else:
            capture = (
                FrameCapture(viewer, image_dir=args.save_images, video_path=args.save_video, fps=args.fps)
                if wants_capture
                else None
            )
            return viewer, capture

    if wants_capture:
        _log(f"--save-images/--save-video need the gl viewer; ignoring them for --viewer {kind}")

    if kind == "usd":
        out = _prepare_output(args.output_path or default_output_path(default_stem, ".usd"))
        # num_frames=None lets the stage grow past the flag; is_running() then never stops us.
        viewer = nv.ViewerUSD(
            output_path=out,
            fps=args.fps,
            up_axis="Z",
            num_frames=args.num_frames if args.num_frames > 0 else None,
        )
        if exact_timecodes:
            _force_exact_timecodes(viewer, args.fps)
        return viewer, None

    if kind == "file":
        out = _prepare_output(args.output_path or default_output_path(default_stem, ".bin"))
        # auto_save would flush every 100 frames; we close() once at the end instead.
        return nv.ViewerFile(output_path=out, auto_save=False), None

    if kind == "rerun":
        if args.output_path and not args.rerun_address:
            # ViewerRerun's __init__ installs an rr.save() sink and then replaces it with
            # serve_grpc(), leaving a blueprint-only rrd.  Faking the notebook check keeps
            # the save sink live, which is the only way to get geometry into the file.
            import newton._src.viewer.viewer_rerun as viewer_rerun

            viewer_rerun.is_jupyter_notebook = lambda: True
            return nv.ViewerRerun(app_id="robochess", record_to_rrd=_prepare_output(args.output_path)), None
        return nv.ViewerRerun(app_id="robochess", address=args.rerun_address, serve_web_viewer=True), None

    if kind == "viser":
        record = _prepare_output(args.output_path) if args.output_path else None
        # .viser recordings drop poses when no browser is attached: preview only.
        return nv.ViewerViser(port=args.viser_port, label="RoboChess", record_to_viser=record), None

    if kind == "null":
        # Nothing can close a null viewer, so "unlimited" falls back to newton's own default.
        return nv.ViewerNull(num_frames=args.num_frames if args.num_frames > 0 else 1000), None

    raise ValueError(f"unknown viewer {kind!r}")


class FrameCapture:
    """Read ``ViewerGL``'s framebuffer back to the host and write PNGs / an mp4.

    ``ViewerGL.get_frame()`` returns an ``(H, W, 3)`` uint8 warp array (origin
    top-left) via a PBO->CUDA readback, and works in EGL headless mode, which is
    what makes visual artifacts possible on a display-less machine.

    The mp4 is streamed frame by frame rather than buffered (300 frames at
    1280x720 would be 830 MB of RAM).  If imageio has no video backend -- the
    newton 1.6 venv has imageio but neither imageio-ffmpeg nor av -- video is
    dropped and the frames are written as PNGs instead of crashing at close().
    """

    def __init__(
        self,
        viewer: Any,
        image_dir: str | None = None,
        video_path: str | None = None,
        fps: int = 60,
    ) -> None:
        if not hasattr(viewer, "get_frame"):
            raise TypeError(f"{type(viewer).__name__} cannot capture frames; only ViewerGL can")

        self.viewer = viewer
        self.image_dir = image_dir
        self.video_path = video_path
        self.fps = fps
        self.frame_count = 0
        self._writer: Any = None

        import imageio.v2 as imageio

        self._imageio = imageio

        if video_path:
            _prepare_output(video_path)
            try:
                # macro_block_size=1 keeps the requested resolution instead of rounding to 16.
                self._writer = imageio.get_writer(video_path, fps=fps, quality=8, macro_block_size=1)
            except Exception as exc:
                _log(f"no video backend for {video_path} ({type(exc).__name__}: {exc}); writing PNGs instead")
                self.video_path = None
                if not self.image_dir:
                    self.image_dir = os.path.splitext(video_path)[0] + "_frames"
                    _log(f"PNG frames go to {self.image_dir}")

        if self.image_dir:
            os.makedirs(self.image_dir, exist_ok=True)

    def grab(self, frame_index: int) -> None:
        """Read back the frame just rendered and write it out."""
        image = self.viewer.get_frame().numpy()
        if self.image_dir:
            self._imageio.imwrite(os.path.join(self.image_dir, f"frame_{frame_index:05d}.png"), image)
        if self._writer is not None:
            self._writer.append_data(image)
        self.frame_count += 1

    def close(self) -> None:
        """Finalise the mp4 and report what was written.  Safe to call twice."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            _log(f"wrote {self.video_path} ({self.frame_count} frames @ {self.fps} fps)")
        if self.image_dir:
            _log(f"wrote {self.frame_count} PNG frames to {self.image_dir}")
            self.image_dir = None


def run_viewer_loop(
    viewer: ViewerBase,
    example: SimExample,
    args: argparse.Namespace,
    capture: FrameCapture | None = None,
) -> int:
    """Step, render and capture until ``--num-frames`` or the viewer quits.

    Replaces ``newton.examples.run``, which never terminates for the gl, rerun,
    viser and file viewers.  ``--num-frames 0`` hands the stop condition back to
    the viewer, which is what an interactive GL session wants.  Closes *viewer*
    and *capture* on the way out -- the USD/file viewers only write their
    artifact in ``close()`` -- including when the example raises, so a crashed
    run still leaves a partial recording.  Returns the number of rendered frames.
    """
    limit = args.num_frames if args.num_frames > 0 else math.inf
    rendered = 0
    try:
        while viewer.is_running() and rendered < limit:
            example.step()
            example.render()
            if capture is not None:
                capture.grab(rendered)
            rendered += 1
    finally:
        if capture is not None:
            capture.close()
        viewer.close()
        _log(f"rendered {rendered} frames with {type(viewer).__name__}")
    return rendered
