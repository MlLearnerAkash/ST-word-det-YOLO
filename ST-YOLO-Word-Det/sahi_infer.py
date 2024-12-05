#@Author: Akash Manna
#@Date: 2024-11-25
#@Last Modified by: Akash Manna
#@Last Modified time: 2024-11-25

import os
import sys
import time
import random

# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse
from pathlib import Path

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors


class SAHIInference:
    """Runs YOLOv8 and SAHI for object detection on images with options to view and save results."""

    def __init__(self):
        """Initializes the SAHIInference class for performing sliced inference using SAHI with YOLOv8 models."""
        self.detection_model = None

    def load_model(self, weights):
        """Loads a YOLOv8 model with specified weights for object detection using SAHI."""
        yolov8_model_path = f"{weights}"
        download_yolov8s_model(yolov8_model_path)
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8", model_path=yolov8_model_path, confidence_threshold=0.3, device="cpu"
        )

    def inference(self, weights="yolov8n.pt", source="image.jpg", view_img=False, save_img=False, exist_ok=False):
        """
        Run object detection on an image using YOLOv8 and SAHI.

        Args:
            weights (str): Model weights path.
            source (str): Image file path.
            view_img (bool): Show results.
            save_img (bool): Save results.
            exist_ok (bool): Overwrite existing files.
        """
        # Load the image
        image = cv2.imread(source)
        assert image is not None, f"Error reading image file: {source}"

        # Output setup
        save_dir = increment_path(Path("ultralytics_results_with_sahi") / "exp", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"{Path(source).stem}_output.jpg"

        # Load model
        self.load_model(weights)

        # Perform sliced prediction
        annotator = Annotator(image, line_width=1)  # Initialize annotator for plotting detection results
        results = get_sliced_prediction(
            image,
            self.detection_model,
            slice_height=1024,
            slice_width=1024,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1,
        )
        detection_data = [
            (det.category.name, det.category.id, (det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy))
            for det in results.object_prediction_list
        ]

        for det in detection_data:
            annotator.box_label(det[2], label=str(det[0]), color=colors(int(det[1]), True))

        # Save or display the result
        if save_img:
            cv2.imwrite(str(output_path), image)
            print(f"Image saved with detections at {output_path}")
        if view_img:
            cv2.imshow("Detection Results", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def parse_opt(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")
        parser.add_argument("--source", type=str, required=True, help="image file path")
        parser.add_argument("--view-img", action="store_true", help="show results")
        parser.add_argument("--save-img", action="store_true", help="save results")
        parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
        return parser.parse_args()


if __name__ == "__main__":
    inference = SAHIInference()
    inference.inference(**vars(inference.parse_opt()))
