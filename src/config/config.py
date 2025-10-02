from typing import TypedDict

from pydantic import BaseModel


class AnchorConfig(TypedDict):
    strides: list[int]
    reg_max: int | None


class LayerConfig(TypedDict, total=False):
    args: dict
    source: int | str | list[str | int]
    tags: str
    output: bool
    external: list


BlockConfig = list[dict[str, LayerConfig]]


class ModelConfig(TypedDict):
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
