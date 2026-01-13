import cv2
import os

class VideoWriterMP4:
    def __init__(self, output_path, fps, frame_size, codec="mp4v"):
        """
        output_path: path to the .mp4 file
        fps: video frame rate
        frame_size: (width, height)
        codec: mp4v / avc1 / H264 (depend on your system)
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            frame_size
        )

        if not self.writer.isOpened():
            raise RuntimeError("Cannot open VideoWriter")

    def write(self, frame):
        """
        frame: BGR frame (numpy array)
        """
        self.writer.write(frame)

    def release(self):
        self.writer.release()