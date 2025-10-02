from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Generator

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.tools.data_augmentation import AugmentationComposer


class StreamDataLoader:
    def __init__(self, source: str, image_size: tuple[int, int]):
        self.source = source
        assert self.source is not None

        self.running = True

        self.is_stream = False
        # TODO: is_stream case
        # self.is_stream = isinstance(self.source, int) or str(
        #     self.source
        # ).lower().startswith("rtmp://")

        self.transform = AugmentationComposer([], image_size)
        self.stop_event = Event()

        if self.is_stream:
            self.cap = cv2.VideoCapture(self.source)
        else:
            assert not isinstance(self.source, int)
            self.source = Path(self.source)
            self.queue = Queue()
            self.thread = Thread(target=self.load_source)
            self.thread.start()

    def load_source(self):
        if self.source.is_dir():  # image folder
            self.load_image_folder(self.source)
        elif any(
            self.source.suffix.lower().endswith(ext) for ext in [".mp4", ".avi", ".mkv"]
        ):  # Video file
            self.load_video_file(self.source)
        else:  # Single image
            self.process_image(self.source)

    def load_image_folder(self, folder):
        folder_path = Path(folder)
        for file_path in folder_path.rglob("*"):
            if self.stop_event.is_set():
                break
            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                self.process_image(file_path)

    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")
        self.process_frame(image)

    def load_video_file(self, video_path):
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame)
        cap.release()

    def process_frame(self, frame):
        if isinstance(frame, np.ndarray):
            # TODO: we don't need cv2
            import cv2

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
        origin_frame = frame
        frame, _, rev_tensor = self.transform(frame, torch.zeros(0, 5))
        frame = frame[None]
        rev_tensor = rev_tensor[None]
        if not self.is_stream:
            self.queue.put((frame, rev_tensor, origin_frame))
        else:
            self.current_frame = (frame, rev_tensor, origin_frame)

    def __iter__(self) -> Generator[Tensor, None, None]:
        return self

    def __next__(self) -> Tensor:
        if self.is_stream:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                raise StopIteration
            self.process_frame(frame)
            return self.current_frame
        else:
            try:
                frame = self.queue.get(timeout=1)
                return frame
            except Empty:
                raise StopIteration

    def stop(self):
        self.running = False
        if self.is_stream:
            self.cap.release()
        else:
            self.thread.join(timeout=1)

    def __len__(self):
        return self.queue.qsize() if not self.is_stream else 0
