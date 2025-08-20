# data/data.py
import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------- Classes ----------
CLASS_MAPPING = {
    "adult": 1,
    "egg masses": 2,
    "instar nymph (1-3)": 3,
    "instar nymph (4)": 4,
}
NUM_CLASSES = 5  # background(0) + 4 foreground

# ---------- Fixed resize (per requirement) ----------
RESIZE_HW = (576, 576)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in _IMG_EXTS


def _paired_xml_for(img_path: Path) -> Path:
    return img_path.with_suffix(".xml")


def _parse_xml_boxes_and_classes(xml_path: Path) -> Tuple[Tuple[int, int], List[Tuple[int, int, int, int, int]]]:
    """
    Returns:
      (width, height),
      list of (xmin, ymin, xmax, ymax, class_idx) for ALL valid objects
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # size
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
            # Skip unknown class
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

        # clip + sanity
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width, xmax)
        ymax = min(height, ymax)
        if xmin >= xmax or ymin >= ymax:
            continue

        boxes.append((xmin, ymin, xmax, ymax, cls_idx))

    return (width, height), boxes


def _mask_from_boxes(width: int, height: int, boxes: List[Tuple[int, int, int, int, int]]) -> Image.Image:
    """
    Rasterize axis-aligned boxes into a single-channel mask (uint8), background=0.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for (xmin, ymin, xmax, ymax, cls_idx) in boxes:
        mask[ymin:ymax, xmin:xmax] = cls_idx
    return Image.fromarray(mask, mode="L")


class SegmentationDataset(Dataset):
    def __init__(self, root_dir: str, split: str):
        self.split = split
        self.split_dir = Path(root_dir) / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # transforms
        self.img_tf = transforms.Compose([
            transforms.Resize(RESIZE_HW, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.mask_resize = transforms.Resize(RESIZE_HW, interpolation=transforms.InterpolationMode.NEAREST)

        # discover pairs + sanity check
        self.samples: List[Tuple[Path, Path]] = []  # (img_path, xml_path)
        self.per_class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)  # simple object-count summary

        all_imgs = [Path(p) for p in glob.glob(str(self.split_dir / "*")) if _is_image(Path(p))]
        all_imgs.sort()

        kept = 0
        for img_path in all_imgs:
            xml_path = _paired_xml_for(img_path)
            if not xml_path.exists():
                continue

            # basic image sanity
            try:
                with Image.open(img_path) as im:
                    im.verify()  # quick check
            except Exception:
                continue

            # xml sanity + get objects
            try:
                (w, h), boxes = _parse_xml_boxes_and_classes(xml_path)
            except Exception:
                continue

            # keep even if zero boxes (will be background-only)
            self.samples.append((img_path, xml_path))
            kept += 1

            # update per-class counts by counting objects (not pixels)
            for _, _, _, _, cls_idx in boxes:
                if 0 <= cls_idx < NUM_CLASSES:
                    self.per_class_counts[cls_idx] += 1

        print(f"[{self.split}] usable samples: {kept} in {self.split_dir}")

        # print per-class object counts (class 0 is background—omit from summary)
        names = ["background(0)"] + list(CLASS_MAPPING.keys())
        print(f"[{self.split}] per-class object counts:")
        for c in range(NUM_CLASSES):
            print(f"  {names[c]}: {int(self.per_class_counts[c])}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, xml_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        (w, h), boxes = _parse_xml_boxes_and_classes(xml_path)
        mask = _mask_from_boxes(w, h, boxes)  # PIL Image (L)

        image = self.img_tf(image)
        mask = self.mask_resize(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()  # (H,W) long indices

        return image, mask


def get_loaders( data_root: str, batch_size: int = 16, num_workers: int = 2, img_size: int = 576,  # ignored intentionally (fixed resize per requirement)
):
    """
    Returns (train_loader, val_loader, test_loader)
    and prints per-class counts for each split.
    """
    train_set = SegmentationDataset(data_root, split="train")
    val_set   = SegmentationDataset(data_root, split="valid")
    test_set  = SegmentationDataset(data_root, split="test")

    # NOTE: counts printed inside dataset constructors already

    pin = torch.cuda.is_available()
    train_loader = DataLoader( train_set, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=pin, drop_last=False)
    

    val_loader = DataLoader( val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin, drop_last=False)
    

    test_loader = DataLoader( test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin, drop_last=False)
    

    return train_loader, val_loader, test_loader
