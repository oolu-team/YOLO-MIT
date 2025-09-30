import logging
from pathlib import Path

import hydra
from lightning import LightningModule, Trainer
from PIL import Image
from rich.logging import RichHandler

from src.config.config import Config, ModelConfig
from src.model.yolo import create_model
from src.tools.data_loader import StreamDataLoader
from src.tools.drawer import draw_bboxes
from src.utils.bounding_box_utils import Vec2Box
from src.utils.model_utils import PostProcess


class BaseModel(LightningModule):
    def __init__(self, model: ModelConfig, class_num: int):
        super().__init__()
        self.model = create_model(model, class_num=class_num)

    def forward(self, x):
        return self.model(x)


class InferenceModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg.model, class_num=cfg.dataset.class_num)
        self.cfg = cfg
        self._save_predict = True
        # TODO: Add FastModel

        self.predict_loader = StreamDataLoader(cfg.task.data)

    def setup(self, stage):
        # def create_converter(model_version: str = "v9-c", *args, **kwargs) -> Anc2Box | Vec2Box:
        #     if "v7" in model_version:  # check model if v7
        #         converter = Anc2Box(*args, **kwargs)
        #     else:
        #         converter = Vec2Box(*args, **kwargs)
        #     return converter

        self.vec2box = Vec2Box(
            # self.cfg.model.name,
            self.model,
            self.cfg.model.anchor,
            self.cfg.image_size,
            self.device,
        )
        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx: int):
        images, rev_tensor, origin_frame = batch
        predicts = self.post_process(self(images), rev_tensor=rev_tensor)
        img = draw_bboxes(origin_frame, predicts, idx2label=self.cfg.dataset.class_list)
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
    # else:
    #     rich_logger.setLevel(logging.DEBUG)

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


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    name = cfg.name
    quite = False

    logger = init_logger(quite=quite)

    save_path = Path("runs", "inference", name)
    save_path.mkdir(parents=True, exist_ok=True)
    if not quite:
        logger.info(f"📄 Created log folder: [blue b u]{save_path}[/]")

    # logger.addHandler(logging.FileHandler(save_path / "output.log"))

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

    model = InferenceModel(cfg)

    result = trainer.predict(model)
    # print(result[0]) => tuple(img, fps)


if __name__ == "__main__":
    main()
