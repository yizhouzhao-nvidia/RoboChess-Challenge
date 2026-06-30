import os
import omni.timeline
import carb
from omni.replicator.core import Writer, AnnotatorRegistry, backends
from omni.replicator.core import functional as F
from video_encoding import get_video_encoding_interface


class RGBVideoWriter(Writer):
    """RGB-only .mp4 writer using NVENC, based on CosmosWriter's pattern."""

    def __init__(self, output_dir: str, clip_name: str = "rgb"):
        self._output_dir = output_dir
        self._clip_name = clip_name
        self._frame_id = 0
        self._frame_rate = None
        self._backend = backends.DiskBackend(output_dir=output_dir)
        self.annotators = ["rgb"]   # only request RGB from the render product

    def write(self, data):
        if self._frame_rate is None:
            self._frame_rate = omni.timeline.get_timeline_interface().get_time_codes_per_seconds()

        rgb = data.get("rgb")
        if rgb is None:
            return
        self._backend.schedule(
            F.write_image,
            data=rgb,
            path=f"frames/rgb_{self._frame_id:04}.png",
        )
        self._frame_id += 1

    def on_final_frame(self):
        """Called by replicator when capture ends — encode PNGs → .mp4."""
        if self._frame_id == 0:
            return

        # Wait for all PNG writes to flush before encoding
        from omni.replicator.core.backends import io_queue
        io_queue.wait_until_done()

        video_encoding = get_video_encoding_interface()
        if video_encoding is None:
            carb.log_warn("RGBVideoWriter: NVENC not available, keeping PNG frames.")
            return

        out_path = os.path.join(self._output_dir, f"{self._clip_name}.mp4")
        try:
            video_encoding.start_encoding(out_path, self._frame_rate, self._frame_id, True)
            for i in range(self._frame_id):
                video_encoding.encode_next_frame_from_file(
                    os.path.join(self._output_dir, f"frames/rgb_{i:04}.png")
                )
            video_encoding.finalize_encoding()
            print(f"RGBVideoWriter: wrote {out_path}")
        except Exception as e:
            carb.log_error(f"RGBVideoWriter: encoding failed: {e}")
        finally:
            self._frame_id = 0
