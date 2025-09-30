from dataclasses import dataclass

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


@dataclass
class BlockConfig:
    block: list[dict[str, LayerConfig]]


@dataclass
class ModelConfig:
    name: str | None
    anchor: AnchorConfig
    model: dict[str, BlockConfig]


@dataclass
class DownloadDetail:
    url: str
    file_size: int


@dataclass
class DownloadOptions:
    details: dict[str, DownloadDetail]


@dataclass
class DatasetConfig:
    path: str
    class_num: int
    class_list: list[str]
    auto_download: DownloadOptions | None


@dataclass
class DataConfig:
    shuffle: bool
    batch_size: int
    pin_memory: bool
    cpu_num: int
    image_size: list[int]
    data_augment: dict[str, int]
    source: str | int | None
    dynamic_shape: bool | None


@dataclass
class NMSConfig:
    min_confidence: float
    min_iou: float
    max_bbox: int


@dataclass
class InferenceConfig:
    task: str
    nms: NMSConfig
    data: DataConfig
    fast_inference: None
    save_predict: bool


@dataclass
class Config:
    task: InferenceConfig
    dataset: DatasetConfig
    model: ModelConfig
    name: str

    device: str | int | list[int]
    cpu_num: int

    image_size: list[int]

    out_path: str
    exist_ok: bool

    lucky_number: int
    use_wandb: bool
    use_tensorboard: bool

    weight: str | None


IDX_TO_ID = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]
