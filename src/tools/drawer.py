import logging
import random

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

logger = logging.getLogger("yolo")


def draw_bboxes(
    img: Image.Image | torch.Tensor,
    bboxes: list[list[int | float]] | list[torch.Tensor],
    *,
    idx2label: list | None = None,
) -> Image.Image:
    """
    Draw bounding boxes on an image.

    Args:
    - img (PIL Image or torch.Tensor): Image on which to draw the bounding boxes.
    - bboxes (List of Lists/Tensors): Bounding boxes with [class_id, x_min, y_min, x_max, y_max],
      where coordinates are normalized [0, 1].
    """
    if isinstance(img, Image.Image):
        pil_img = img.copy()
    else:
        if img.dim() > 3:
            logger.warning("🔍 >3 dimension tensor detected, using the 0-idx image.")
            img = img[0]
        pil_img = to_pil_image(img).copy()

    if isinstance(bboxes, list) or bboxes.ndim == 3:
        bboxes = bboxes[0]  # pyright: ignore[reportAssignmentType]

    label_size = pil_img.size[1] / 30
    draw = ImageDraw.Draw(pil_img, "RGBA")

    try:
        font = ImageFont.truetype("arial.ttf", int(label_size))
    except IOError:
        font = ImageFont.load_default(int(label_size))

    for bbox in bboxes:
        class_id, x_min, y_min, x_max, y_max, *conf = [float(val) for val in bbox]
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)
        bbox = [(x_min, y_min), (x_max, y_max)]

        random.seed(int(class_id))
        color_map = (
            random.randint(0, 200),
            random.randint(0, 200),
            random.randint(0, 200),
        )

        draw.rounded_rectangle(bbox, outline=(*color_map, 200), radius=5, width=2)
        draw.rounded_rectangle(bbox, fill=(*color_map, 100), radius=5)

        class_text = str(idx2label[int(class_id)] if idx2label else int(class_id))
        label_text = f"{class_text}" + (f" {conf[0]: .0%}" if conf else "")

        text_bbox = font.getbbox(label_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = (text_bbox[3] - text_bbox[1]) * 1.5

        text_background = [(x_min, y_min), (x_min + text_width, y_min + text_height)]
        draw.rounded_rectangle(text_background, fill=(*color_map, 175), radius=2)
        draw.text((x_min, y_min), label_text, fill="white", font=font)

    return pil_img
