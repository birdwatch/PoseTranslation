import argparse
import glob
import os
import sys

import cv2
import numpy as np

prev_img = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    return parser.parse_args()


def make_flow_video(video_path: str, output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if ret:
            flow = calc_flow(frame)
            bgr = calc_bgr(flow)
            writer.write(bgr)
        else:
            break
    cap.release()
    writer.release()


def calc_flow(img: np.ndarray) -> np.ndarray:
    global prev_img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if prev_img is None:
        prev_img = img
    flow = cv2.calcOpticalFlowFarneback(prev_img, img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prev_img = img
    return flow


def calc_bgr(flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[:, :, 0] = ang * (180 / np.pi / 2)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = v * 30
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


if __name__ == "__main__":
    args = parse_args()
    input_path = args.input
    output_path = args.output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f"input_path: {input_path}")

    for video_path in glob.glob(os.path.join(input_path, "*.mp4")):
        video_name = os.path.basename(video_path)
        output_video_path = os.path.join(output_path, video_name)
        make_flow_video(video_path, output_video_path)
        print(f"saved {output_video_path}")
