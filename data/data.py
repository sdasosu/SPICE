import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASS_MAPPING = {
    "adult": 1,
    "egg masses": 2,
    "instar nymph (1-3)": 3,
    "instar nymph (4)": 4,
}
NUM_CLASSES = 5

RESIZE_HW = (576, 576)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in _IMG_EXTS


def _paired_xml_for(img_path: Path) -> Path:
    return img_path.with_suffix(".xml")


def _parse_xml_boxes_and_classes(
    xml_path: Path,
) -> Tuple[Tuple[int, int], List[Tuple[int, int, int, int, int]]]:
    """
    Returns:
      (width, height),
      list of (xmin, ymin, xmax, ymax, class_idx) for ALL valid objects
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    size_el = root.find("size")
    if size_el is None:
        raise ValueError("Missing <size> in XML")

    width = int(size_el.find("width").text)
    height = int(size_el.find("height").text)
    if width <= 0 or height <= 0:
        raise ValueError("Non-positive image size in XML")

    boxes = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        cname = name_el.text.strip() if name_el is not None else None
        if not cname or cname not in CLASS_MAPPING:
            continue
        cls_idx = CLASS_MAPPING[cname]

        bb = obj.find("bndbox")
        if bb is None:
            continue

        try:
            xmin = int(bb.find("xmin").text)
            ymin = int(bb.find("ymin").text)
            xmax = int(bb.find("xmax").text)
            ymax = int(bb.find("ymax").text)
        except Exception:
            continue

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width, xmax)
        ymax = min(height, ymax)
        if xmin >= xmax or ymin >= ymax:
            continue

        boxes.append((xmin, ymin, xmax, ymax, cls_idx))

    return (width, height), boxes


def _mask_from_boxes(
    width: int, height: int, boxes: List[Tuple[int, int, int, int, int]]
) -> Image.Image:
    """
    Rasterize axis-aligned boxes into a single-channel mask (uint8), background=0.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for xmin, ymin, xmax, ymax, cls_idx in boxes:
        mask[ymin:ymax, xmin:xmax] = cls_idx
    return Image.fromarray(mask, mode="L")


class SegmentationDataset(Dataset):
    def __init__(self, root_dir: str, split: str):
        self.split = split
        self.split_dir = Path(root_dir) / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.img_tf = transforms.Compose(
            [
                transforms.Resize(
                    RESIZE_HW, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.mask_resize = transforms.Resize(
            RESIZE_HW, interpolation=transforms.InterpolationMode.NEAREST
        )

        self.samples: List[Tuple[Path, Path]] = []
        self.per_class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

        all_imgs = [
            Path(p) for p in glob.glob(str(self.split_dir / "*")) if _is_image(Path(p))
        ]
        all_imgs.sort()

        kept = 0
        for img_path in all_imgs:
            xml_path = _paired_xml_for(img_path)
            if not xml_path.exists():
                continue

            try:
                with Image.open(img_path) as im:
                    im.verify()
            except Exception:
                continue

            try:
                (w, h), boxes = _parse_xml_boxes_and_classes(xml_path)
            except Exception:
                continue

            self.samples.append((img_path, xml_path))
            kept += 1

            for _, _, _, _, cls_idx in boxes:
                if 0 <= cls_idx < NUM_CLASSES:
                    self.per_class_counts[cls_idx] += 1

        self.dataset_info = {
            "split": self.split,
            "samples": kept,
            "path": str(self.split_dir),
            "per_class_counts": self.per_class_counts.tolist(),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, xml_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        (w, h), boxes = _parse_xml_boxes_and_classes(xml_path)
        mask = _mask_from_boxes(w, h, boxes)

        image = self.img_tf(image)
        mask = self.mask_resize(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()

        return image, mask


def worker_init_fn(worker_id):
    """Initialize worker with unique seed for reproducibility"""
    import random

    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_loaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 2,
    img_size: int = 576,
    seed: int = 42,
):
    """
    Returns (train_loader, val_loader, test_loader)
    and prints per-class counts for each split.
    """
    train_set = SegmentationDataset(data_root, split="train")
    val_set = SegmentationDataset(data_root, split="valid")
    test_set = SegmentationDataset(data_root, split="test")

    pin = torch.cuda.is_available()
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    val_generator = torch.Generator()
    val_generator.manual_seed(seed + 1)

    test_generator = torch.Generator()
    test_generator.manual_seed(seed + 2)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=val_generator,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=test_generator,
    )

    return train_loader, val_loader, test_loader
