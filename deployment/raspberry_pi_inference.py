#!/usr/bin/env python3
"""
Unified Raspberry Pi Inference Script for EPIC2.0
Supports all model architectures with embedded configurations
"""

import argparse
import io
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

MODEL_CONFIGS = {
    "DeepLabV3Plus_resnet": {
        "architecture": "DeepLabV3Plus",
        "encoder_name": "resnet50",
    },
    "DeepLabV3Plus_efficientnet": {
        "architecture": "DeepLabV3Plus",
        "encoder_name": "timm-efficientnet-b3",
    },
    "UNET_resnet": {"architecture": "Unet", "encoder_name": "resnet50"},
    "UNET_efficientnet": {
        "architecture": "Unet",
        "encoder_name": "timm-efficientnet-b3",
    },
    "FPN_resnet": {"architecture": "FPN", "encoder_name": "resnet50"},
    "FPN_efficientnet": {"architecture": "FPN", "encoder_name": "timm-efficientnet-b3"},
}

# Device setup
device = torch.device("cpu")

# Constants
CLASS_MAPPING = {
    "adult": 1,
    "egg masses": 2,
    "instar nymph (1-3)": 3,
    "instar nymph (4)": 4,
}

COLOR_MAP = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255],
    4: [255, 255, 0],
}


class RaspberryPiInference:
    """Unified inference engine for Raspberry Pi deployment"""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        pruning_ratio: Optional[float] = None,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        confidence_threshold: float = 0.05,
        capture_interval: float = 0.2,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.capture_interval = capture_interval
        self.pruning_ratio = pruning_ratio

        self.script_dir = Path(__file__).parent.resolve()

        if data_dir is None:
            self.data_dir = self.script_dir / "data"
        else:
            self.data_dir = Path(data_dir)

        if output_dir is None:
            if pruning_ratio is not None:
                output_dir = (
                    self.script_dir / f"results/{model_name}_pruned_{pruning_ratio:.1f}"
                )
            else:
                output_dir = self.script_dir / f"results/{model_name}"
        self.output_dir = Path(output_dir)

        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Model {model_name} not found. Available models: {list(MODEL_CONFIGS.keys())}"
            )
        self.model_config = MODEL_CONFIGS[model_name]

        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
        elif pruning_ratio is not None:
            model_file = (
                f"{model_name}_magnitude_taylor_{pruning_ratio:.2f}_kd_refine.pth"
            )
            self.checkpoint_path = self.script_dir / "models" / model_file
            print(f"Looking for pruned model: {self.checkpoint_path}")
        else:
            default_name = f"{model_name}.pth"
            self.checkpoint_path = self.script_dir / "models" / default_name
            print(f"Looking for default model: {self.checkpoint_path}")

        self.setup_directories()

        self.class_labels = self.load_class_labels()

        self.model = self.load_model()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((576, 576)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup_directories(self):
        """Create necessary directories"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "detected_images").mkdir(exist_ok=True)

    def load_class_labels(self) -> list:
        """Load class labels - embedded in script, no external file needed"""
        class_labels_dict = {
            "0": "Others",  # Background
            "1": "adult",
            "2": "instar nymph (1-3)",
            "3": "instar nymph (4)",
            "4": "egg masses",
        }

        class_labels = [
            class_labels_dict[key]
            for key in sorted(class_labels_dict, key=lambda x: int(x))
        ]
        print("Using embedded class labels")
        return class_labels

    def create_model(self) -> nn.Module:
        """Create model based on configuration"""
        architecture = self.model_config["architecture"]
        encoder_name = self.model_config["encoder_name"]

        common_params = {
            "encoder_name": encoder_name,
            "encoder_weights": None,
            "in_channels": 3,
            "classes": 5,
            "activation": None,
        }

        if architecture == "DeepLabV3Plus":
            model = smp.DeepLabV3Plus(**common_params)
        elif architecture == "Unet":
            model = smp.Unet(**common_params)
        elif architecture == "FPN":
            model = smp.FPN(**common_params)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        return model

    def load_model(self) -> nn.Module:
        """Load model with weights"""
        if self.checkpoint_path.exists():
            try:
                model = torch.load(
                    self.checkpoint_path, map_location=device, weights_only=False
                )
                print(f"Loaded model from {self.checkpoint_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating new model with random initialization")
                model = self.create_model()
        else:
            print(f"Checkpoint not found at {self.checkpoint_path}")
            print("Creating new model with random initialization")
            model = self.create_model()

        model = model.to(device)
        model.eval()
        return model

    def capture_image(self, image_path: Optional[Path] = None) -> Path:
        """
        Capture image using camera or use existing image
        Uncomment libcamera-still line for actual Raspberry Pi deployment
        """
        if image_path is None:
            image_path = self.data_dir / "captured.jpg"

        # os.system(f'libcamera-still -o {image_path}')  # Uncomment for Raspberry Pi

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        return image_path

    def mask_to_color(self, mask: np.ndarray) -> np.ndarray:
        """Convert class mask to color image"""
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for class_idx, color in COLOR_MAP.items():
            color_mask[mask == class_idx] = color
        return color_mask

    def get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature (Raspberry Pi specific)"""
        try:
            temp_str = os.popen("vcgencmd measure_temp").readline()
            temp = float(temp_str.replace("temp=", "").replace("'C\n", ""))
            return temp
        except Exception:
            try:
                thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
                if thermal_path.exists():
                    with open(thermal_path) as f:
                        temp = float(f.read()) / 1000.0
                    return temp
            except Exception:
                pass
        return None

    def measure_inference_time(func):
        """Decorator to measure inference time"""

        def wrapper(self, *args, **kwargs):
            start_cpu = time.process_time()
            start_wall = time.perf_counter()

            result = func(self, *args, **kwargs)

            cpu_time = time.process_time() - start_cpu
            wall_time = time.perf_counter() - start_wall

            temp = self.get_cpu_temperature()
            temp_str = f"{temp:.1f}°C" if temp else "N/A"

            log_entry = (
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Model: {self.model_name} | "
                f"CPU Time: {cpu_time:.4f}s | "
                f"Wall Time: {wall_time:.4f}s | "
                f"Temperature: {temp_str}"
            )

            # Log to file
            log_path = self.output_dir / "inference_log.txt"
            with open(log_path, "a") as f:
                f.write(log_entry + "\n")

            print(log_entry)
            return result

        return wrapper

    @measure_inference_time
    def process_image(self, image_path: Path) -> Dict:
        """Process single image and return results"""
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        original_image = image.copy()

        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(input_batch)
            if isinstance(outputs, dict) and "out" in outputs:
                output_tensor = outputs["out"]
            elif isinstance(outputs, torch.Tensor):
                output_tensor = outputs
            else:
                output_tensor = outputs

            # Get predictions
            probabilities = torch.nn.functional.softmax(output_tensor, dim=1)
            class_probs = probabilities.sum(dim=[2, 3])[0]

            # Focus on foreground classes
            foreground_probs = class_probs[1:]

            if foreground_probs.sum().item() > 0:
                foreground_probs = foreground_probs / foreground_probs.sum()
                fg_class_idx = foreground_probs.argmax().item()
                main_class_idx = fg_class_idx + 1
                confidence = foreground_probs[fg_class_idx].item()

                if confidence < self.confidence_threshold:
                    predicted_label = "unknown"
                    print(f"Low confidence ({confidence:.4f}), marking as unknown")
                else:
                    predicted_label = self.class_labels[main_class_idx]
                    print(
                        f"Predicted: {predicted_label} (confidence: {confidence:.4f})"
                    )
            else:
                predicted_label = "unknown"
                confidence = 0.0
                print("No foreground class detected")

        # Generate mask
        predicted_classes = torch.argmax(probabilities, dim=1)
        predicted_mask = predicted_classes[0].cpu().numpy()
        color_mask = self.mask_to_color(predicted_mask)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.output_dir / "detected_images"

        # Save original
        original_path = (
            save_dir / f"{predicted_label}_original_{confidence:.4f}_{timestamp}.png"
        )
        original_image.save(original_path)

        # Save mask
        mask_image = Image.fromarray(color_mask)
        mask_path = (
            save_dir / f"{predicted_label}_mask_{confidence:.4f}_{timestamp}.png"
        )
        mask_image.save(mask_path)

        # Create and save comparison
        comparison = self.create_visualization(
            input_tensor.cpu(), mask_image, predicted_label, confidence
        )
        comparison_path = (
            save_dir / f"{predicted_label}_comparison_{confidence:.4f}_{timestamp}.png"
        )
        comparison.save(comparison_path)

        # Log prediction
        if predicted_label not in ["Others", "unknown"]:
            pred_log_path = self.output_dir / "predictions.txt"
            with open(pred_log_path, "a") as f:
                f.write(
                    f"{timestamp} | {self.model_name} | {predicted_label} | {confidence:.4f}\n"
                )

        return {
            "label": predicted_label,
            "confidence": confidence,
            "mask": predicted_mask,
            "paths": {
                "original": original_path,
                "mask": mask_path,
                "comparison": comparison_path,
            },
        }

    def create_visualization(
        self,
        preprocessed_tensor: torch.Tensor,
        mask_image: Image.Image,
        label: str,
        confidence: float,
    ) -> Image.Image:
        """Create side-by-side visualization"""
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        display_tensor = preprocessed_tensor.clone()
        for t, m, s in zip(display_tensor, mean, std):
            t.mul_(s).add_(m)

        display_image = display_tensor.permute(1, 2, 0).numpy()
        display_image = np.clip(display_image, 0, 1)

        # Create figure
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(display_image)
        plt.title("Processed Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(mask_image))
        plt.title(f"Prediction: {label}\nConfidence: {confidence:.4f}")
        plt.axis("off")

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close()
        buf.seek(0)

        return Image.open(buf).convert("RGB")

    def run_continuous(self):
        """Run continuous inference loop"""
        print(f"Starting continuous inference for model: {self.model_name}")
        print(f"Monitoring: {self.data_dir / 'captured.jpg'}")
        print(f"Output directory: {self.output_dir}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Check interval: {self.capture_interval}s")
        print("Press Ctrl+C to stop\n")

        try:
            last_modified = 0

            while True:
                # Capture or check for new image
                captured_path = self.capture_image()

                if captured_path.exists():
                    current_modified = captured_path.stat().st_mtime

                    if current_modified > last_modified:
                        print(
                            f"\n[{datetime.now().strftime('%H:%M:%S')}] New image detected"
                        )
                        self.process_image(captured_path)
                        last_modified = current_modified

                time.sleep(self.capture_interval)

        except KeyboardInterrupt:
            print("\n\nStopped by user")
        except Exception as e:
            print(f"\nError: {e}")

    def run_single(self, image_path: Path):
        """Run inference on single image"""
        print(f"Running single inference for: {image_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        result = self.process_image(image_path)
        print(f"\nResults saved to: {self.output_dir}")
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Unified Raspberry Pi inference script for EPIC2.0"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model name from configuration",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (overrides auto-detection)",
    )
    parser.add_argument(
        "--pruning-ratio",
        type=float,
        default=None,
        choices=[0.5, 0.6, 0.7, 0.8, 0.9],
        help="Pruning ratio (0.5-0.9) for auto-loading pruned models",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing input data (default: script_dir/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output results (auto-generated if not specified)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.05,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=0.2,
        help="Image capture check interval in seconds",
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Run single inference on specified image path",
    )

    args = parser.parse_args()

    # Initialize inference engine
    engine = RaspberryPiInference(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        pruning_ratio=args.pruning_ratio,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        capture_interval=args.capture_interval,
    )

    # Run inference
    if args.single:
        engine.run_single(Path(args.single))
    else:
        engine.run_continuous()


if __name__ == "__main__":
    main()
