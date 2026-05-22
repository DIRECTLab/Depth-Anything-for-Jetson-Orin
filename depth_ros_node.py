#!/usr/bin/env python3
"""
ROS 2 node: subscribes to a camera topic, runs Depth Anything TensorRT
inference, and publishes raw float32 depth + colorized visualization.
"""
from __future__ import annotations

import sys
import os

# Ensure depth_anything package is importable from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa: F401 — required side-effect import
import pycuda.driver as cuda
from torchvision.transforms import Compose

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from depth_anything import transform


class DepthAnythingNode(Node):
    def __init__(self):
        super().__init__('depth_anything_node')

        # --- Parameters ---
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('depth_topic', '/depth/image_raw')
        self.declare_parameter('depth_viz_topic', '/depth/image_viz')
        self.declare_parameter('trt_engine_path', 'weights/depth_anything_vits14_308.trt')
        self.declare_parameter('input_size', 308)  # must be divisible by 14
        self.declare_parameter('publish_viz', True)
        self.declare_parameter('best_effort', False)

        camera_topic    = self.get_parameter('camera_topic').value
        depth_topic     = self.get_parameter('depth_topic').value
        depth_viz_topic = self.get_parameter('depth_viz_topic').value
        trt_engine_path = self.get_parameter('trt_engine_path').value
        input_size      = self.get_parameter('input_size').value
        self.publish_viz = self.get_parameter('publish_viz').value
        best_effort      = self.get_parameter('best_effort').value

        reliability = QoSReliabilityPolicy.BEST_EFFORT if best_effort else QoSReliabilityPolicy.RELIABLE
        self.input_size = input_size  # square model input (width == height)

        # --- TensorRT engine ---
        self.get_logger().info(f'Loading TRT engine from {trt_engine_path}')
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(trt_engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Pinned host memory and device memory for async transfer
        vol = trt.volume((1, 3, input_size, input_size))
        self.h_input  = cuda.pagelocked_empty(vol, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(
            trt.volume((1, 1, input_size, input_size)), dtype=np.float32
        )
        self.d_input  = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.cuda_stream = cuda.Stream()

        # --- Preprocessing transform (matches depth.py exactly) ---
        self.preprocess_fn = Compose([
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

        # --- ROS I/O ---
        qos = QoSProfile(
            reliability=reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.bridge = CvBridge()

        self.sub = self.create_subscription(Image, camera_topic, self._image_cb, qos)

        self.pub_depth = self.create_publisher(Image, depth_topic, qos)
        if self.publish_viz:
            self.pub_viz = self.create_publisher(Image, depth_viz_topic, qos)

        self.get_logger().info(
            f'Depth Anything node ready\n'
            f'  subscribing : {camera_topic}\n'
            f'  depth topic : {depth_topic}\n'
            f'  viz topic   : {depth_viz_topic if self.publish_viz else "(disabled)"}\n'
            f'  input size  : {input_size}x{input_size}'
        )

    # ------------------------------------------------------------------
    # Inference helpers (mirrors DepthEngine but without Camera)
    # ------------------------------------------------------------------

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        img = bgr.astype(np.float32) / 255.0
        img = self.preprocess_fn({'image': img})['image']
        return img[np.newaxis]  # (1, 3, H, W)

    def _infer_raw(self, bgr: np.ndarray) -> np.ndarray:
        """Returns float32 depth map at input resolution (input_size x input_size)."""
        img = self._preprocess(bgr)
        np.copyto(self.h_input, img.ravel())

        cuda.memcpy_htod_async(self.d_input, self.h_input, self.cuda_stream)
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.cuda_stream.handle,
        )
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.cuda_stream)
        self.cuda_stream.synchronize()

        depth = self.h_output.reshape((self.input_size, self.input_size))
        return depth

    def _postprocess(self, depth: np.ndarray, out_h: int, out_w: int):
        """
        Returns (depth_f32, depth_viz):
          depth_f32  — float32 (out_h, out_w), relative depth (higher = farther)
          depth_viz  — uint8 BGR (out_h, out_w, 3) with INFERNO colormap, or None
        """
        depth_f32 = cv2.resize(depth, (out_w, out_h))

        depth_viz = None
        if self.publish_viz:
            norm = (depth_f32 - depth_f32.min()) / (depth_f32.max() - depth_f32.min() + 1e-8)
            depth_u8 = (norm * 255.0).astype(np.uint8)
            depth_viz = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

        return depth_f32, depth_viz

    # ------------------------------------------------------------------
    # ROS callback
    # ------------------------------------------------------------------

    def _image_cb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        out_h, out_w = bgr.shape[:2]

        depth_raw = self._infer_raw(bgr)
        depth_f32, depth_viz = self._postprocess(depth_raw, out_h, out_w)

        stamp = msg.header.stamp
        frame_id = msg.header.frame_id

        # Publish raw float32 depth (32FC1)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_f32, encoding='32FC1')
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = frame_id
        self.pub_depth.publish(depth_msg)

        # Publish colorized visualization (bgr8)
        if self.publish_viz and depth_viz is not None:
            viz_msg = self.bridge.cv2_to_imgmsg(depth_viz, encoding='bgr8')
            viz_msg.header.stamp = stamp
            viz_msg.header.frame_id = frame_id
            self.pub_viz.publish(viz_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthAnythingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
