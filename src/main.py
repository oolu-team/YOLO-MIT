import logging
import typing
from pathlib import Path

from lightning import LightningModule, Trainer
from omegaconf import OmegaConf
from PIL import Image
from rich.logging import RichHandler

from src.config.config import ModelConfig, NMSConfig
from src.model.yolo import YOLO
from src.tools.data_loader import FileDataLoader
from src.tools.drawer import draw_bboxes
from src.utils.bounding_box_utils import Vec2Box
from src.utils.model_utils import PostProcess

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


class BaseModel(LightningModule):
    def __init__(self, model: ModelConfig, class_num: int):
        super().__init__()
        self.model = YOLO(model, class_num=class_num)

    def forward(self, x):
        return self.model(x)


class InferenceModel(BaseModel):
    def __init__(
        self,
        model_config: ModelConfig,
        class_list: list[str],
        source: str,
        image_size: tuple[int, int],
        nms_config: NMSConfig,
    ):
        super().__init__(model_config, class_num=len(class_list))
        self.model_config = model_config
        self.class_list = class_list
        self.nms_config = nms_config
        self.image_size = image_size

        self._save_predict = True
        # TODO: Add FastModel

        self.predict_loader = FileDataLoader(source=source, image_size=image_size)

    def setup(self, stage):
        self.vec2box = Vec2Box(
            self.model,
            self.model_config.anchor,
            self.image_size,
            self.device,
        )
        self.post_process = PostProcess(self.vec2box, self.nms_config)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx: int):
        images, rev_tensor, origin_frame = batch
        predicts = self.post_process(self(images), rev_tensor=rev_tensor)
        img = draw_bboxes(origin_frame, predicts, idx2label=self.class_list)
        if getattr(self.predict_loader, "is_stream", None):
            # fps = self._display_stream(img)
            raise NotImplementedError("stream case")
        else:
            fps = None
        if self._save_predict:
            self._save_image(img, batch_idx)
        return img, fps

    def _save_image(self, img: Image.Image, batch_idx: int):
        save_image_path = (
            Path(self.trainer.default_root_dir) / f"frame{batch_idx:03d}.png"
        )
        img.save(save_image_path)
        print(f"💾 Saved visualize image at {save_image_path}")


def _setup_logger(logger_name: str, quite=False):
    rich_handler = RichHandler(markup=True)
    rich_handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    rich_logger = logging.getLogger(logger_name)
    if rich_logger.handlers:
        rich_logger.handlers.clear()
    rich_logger.addHandler(rich_handler)
    if quite:
        rich_logger.setLevel(logging.ERROR)

    return rich_logger


def init_logger(quite: bool = False):
    _setup_logger("lightning.pytorch")
    _setup_logger("lightning.fabric")

    coco_logger = logging.getLogger("faster_coco_eval.core.cocoeval")
    coco_logger.setLevel(logging.ERROR)

    logger = _setup_logger("yolo")
    logger.propagate = False
    if not quite:
        logger.setLevel(logging.DEBUG)
    return logger


def main():
    model_name = "v9-c"
    quite = False

    logger = init_logger(quite=quite)

    save_path = Path("runs", "inference", model_name)
    save_path.mkdir(parents=True, exist_ok=True)
    if not quite:
        logger.info(f"📄 Created log folder: [blue b u]{save_path}[/]")

    trainer = Trainer(
        accelerator="auto",
        precision="16-mixed",
        logger=[],
        log_every_n_steps=1,
        gradient_clip_val=10,
        gradient_clip_algorithm="norm",
        deterministic=True,
        enable_progress_bar=not quite,
        default_root_dir=save_path,
    )
    logger.info("Trainer initialized")

    model_schema = OmegaConf.structured(ModelConfig)
    model_conf = OmegaConf.load(f"src/config/model/{model_name}.yaml")
    model_conf = typing.cast(ModelConfig, OmegaConf.merge(model_schema, model_conf))

    model = InferenceModel(
        model_conf,
        class_list=class_list,
        source="./5564_first.jpg",
        image_size=(640, 640),
        nms_config=NMSConfig(),
    )

    result = trainer.predict(model)


if __name__ == "__main__":
    main()
