from dataclasses import dataclass

from pydantic import BaseModel
from torch import nn


@dataclass
class YOLOLayer(nn.Module):
    source: int | str | list[int]
    output: bool
    tags: str
    layer_type: str
    usable: bool
    external: dict | None


@dataclass
class AnchorConfig:
    strides: list[int]
    reg_max: int | None
    anchor_num: int | None
    anchor: list[list[int]]


@dataclass
class LayerConfig:
    args: dict
    source: int | str | list[str | int]
    tags: str


BlockConfig = list[dict[str, LayerConfig]]


@dataclass
class ModelConfig:
    name: str | None
    anchor: AnchorConfig
    model: dict[str, BlockConfig]


class NMSConfig(BaseModel):
    min_confidence: float = 0.5
    min_iou: float = 0.5
    max_bbox: int = 300


class_list = [
    "Person",
    "Bicycle",
    "Car",
    "Motorcycle",
    "Airplane",
    "Bus",
    "Train",
    "Truck",
    "Boat",
    "Traffic light",
    "Fire hydrant",
    "Stop sign",
    "Parking meter",
    "Bench",
    "Bird",
    "Cat",
    "Dog",
    "Horse",
    "Sheep",
    "Cow",
    "Elephant",
    "Bear",
    "Zebra",
    "Giraffe",
    "Backpack",
    "Umbrella",
    "Handbag",
    "Tie",
    "Suitcase",
    "Frisbee",
    "Skis",
    "Snowboard",
    "Sports ball",
    "Kite",
    "Baseball bat",
    "Baseball glove",
    "Skateboard",
    "Surfboard",
    "Tennis racket",
    "Bottle",
    "Wine glass",
    "Cup",
    "Fork",
    "Knife",
    "Spoon",
    "Bowl",
    "Banana",
    "Apple",
    "Sandwich",
    "Orange",
    "Broccoli",
    "Carrot",
    "Hot dog",
    "Pizza",
    "Donut",
    "Cake",
    "Chair",
    "Couch",
    "Potted plant",
    "Bed",
    "Dining table",
    "Toilet",
    "Tv",
    "Laptop",
    "Mouse",
    "Remote",
    "Keyboard",
    "Cell phone",
    "Microwave",
    "Oven",
    "Toaster",
    "Sink",
    "Refrigerator",
    "Book",
    "Clock",
    "Vase",
    "Scissors",
    "Teddy bear",
    "Hair drier",
    "Toothbrush",
]
