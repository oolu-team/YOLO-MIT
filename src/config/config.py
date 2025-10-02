from dataclasses import dataclass, field

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
    source: int | str | list[int]
    tags: str


BlockConfig = list[dict[str, LayerConfig]]


@dataclass
class ModelConfig:
    name: str | None
    anchor: AnchorConfig
    model: dict[str, BlockConfig] = field(default_factory=dict)


class NMSConfig(BaseModel):
    min_confidence: float = 0.5
    min_iou: float = 0.5
    max_bbox: int = 300
