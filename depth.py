from __future__ import annotations
from typing import Sequence

import argparse
import os
import time
import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose

from camera import Camera
from depth_anything import DepthAnything, transform


class DepthEngine:
    def __init__(
        self,
        sensor_id: int | Sequence[int] = 0,
        input_size: int = 308,
        frame_rate: int = 15,
        weights_path: str = 'LiheYoung/depth_anything_vits14',
        save_path: str = None,
        raw: bool = False,
        stream: bool = False,
        record: bool = False,
        save: bool = False,
        grayscale: bool = False,
    ):
        self.camera = Camera(sensor_id=sensor_id, frame_rate=frame_rate)
        self.width = input_size
        self.height = input_size
        self._width = int(self.camera.cap[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.camera.cap[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.save_path = Path(save_path) if isinstance(save_path, str) else Path("results")
        self.raw = raw
        self.stream = stream
        self.record = record
        self.save = save
        self.grayscale = grayscale

        if raw:
            self.raw_depth = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading model from {weights_path!r} on {self.device}")
        self.model = DepthAnything.from_pretrained(weights_path).to(self.device).eval()
        print(f"Model loaded. Camera: {self._width}x{self._height}, input size: {input_size}x{input_size}")

        self.transform = Compose([
            transform.Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=False,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            transform.NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transform.PrepareForNet(),
        ])

        if record:
            self.video = cv2.VideoWriter(
                'results.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),
                frame_rate,
                (2 * self._width, self._height),
            )

        if save:
            os.makedirs(self.save_path, exist_ok=True)
            self.save_path = self.save_path / f'{len(os.listdir(self.save_path)) + 1:06d}'
            os.makedirs(self.save_path, exist_ok=True)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) / 255.0
        image = self.transform({'image': image})['image']
        return torch.from_numpy(image).unsqueeze(0).to(self.device)

    def postprocess(self, depth: np.ndarray) -> np.ndarray:
        print(f"  Output: shape={depth.shape}, min={depth.min():.4f}, max={depth.max():.4f}, mean={depth.mean():.4f}")
        depth = cv2.resize(depth, (self._width, self._height))

        if self.raw:
            return depth

        dmin, dmax = depth.min(), depth.max()
        if dmax - dmin < 1e-6:
            print("  WARNING: depth map is nearly constant")
        depth = (dmax - depth) / (dmax - dmin + 1e-6) * 255.0
        depth = depth.astype(np.uint8)

        if self.grayscale:
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        else:
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        return depth

    def infer(self, image: np.ndarray) -> np.ndarray:
        tensor = self.preprocess(image)
        print(f"  Input: shape={tuple(tensor.shape)}, min={tensor.min():.4f}, max={tensor.max():.4f}")

        t0 = time.time()
        with torch.no_grad():
            depth = self.model(tensor)
        print(f"Inference time: {time.time() - t0:.4f}s")

        depth = depth.squeeze().cpu().numpy()
        return self.postprocess(depth)

    def run(self):
        try:
            while True:
                _, frame = self.camera.cap[0].read()
                depth = self.infer(frame)

                if self.raw:
                    self.raw_depth = depth
                else:
                    results = np.concatenate((frame, depth), axis=1)

                    if self.record:
                        self.video.write(results)

                    if self.save:
                        cv2.imwrite(str(self.save_path / f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}.png'), results)

                    if self.stream:
                        cv2.imshow('Depth', results)
                        if cv2.waitKey(1) == ord('q'):
                            break
        except Exception as e:
            print(e)
        finally:
            if self.record:
                self.video.release()
            if self.stream:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--frame_rate', type=int, default=15)
    args.add_argument('--input_size', type=int, default=308)
    args.add_argument('--weights', type=str, default='LiheYoung/depth_anything_vits14')
    args.add_argument('--raw', action='store_true')
    args.add_argument('--stream', action='store_true')
    args.add_argument('--record', action='store_true')
    args.add_argument('--save', action='store_true')
    args.add_argument('--grayscale', action='store_true')
    args = args.parse_args()

    depth = DepthEngine(
        frame_rate=args.frame_rate,
        input_size=args.input_size,
        weights_path=args.weights,
        raw=args.raw,
        stream=args.stream,
        record=args.record,
        save=args.save,
        grayscale=args.grayscale,
    )
    depth.run()
