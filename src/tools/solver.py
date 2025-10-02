import logging
from pathlib import Path

from lightning import LightningModule
from PIL import Image

from src.config.config import ModelConfig, NMSConfig
from src.model.yolo import YOLO
from src.tools.data_loader import FileDataLoader
from src.tools.drawer import draw_bboxes
from src.utils.bounding_box_utils import Vec2Box
from src.utils.model_utils import PostProcess


class BaseModule(LightningModule):
    WEIGHT_PATH = Path("weights")

    def __init__(self, model: ModelConfig, class_num: int):
        super().__init__()
        self.model = YOLO(model, class_num=class_num)
        self.model.save_load_weight(self.WEIGHT_PATH / f"{model['name']}.pt")

    def forward(self, x):
        return self.model(x)


class InferenceModel(BaseModule):
    def __init__(
        self,
        model_config: ModelConfig,
        class_list: list[str],
        source: str,
        image_size: tuple[int, int],
        nms_config: NMSConfig,
    ):
        super().__init__(model_config, class_num=len(class_list))

        self._logger = logging.getLogger("yolo")
        anchor_config = model_config["anchor"]
        if "strides" in anchor_config:
            self._logger.info(
                f":japanese_not_free_of_charge_button: Found stride of model {anchor_config['strides']}"
            )
            self.strides = anchor_config["strides"]
        else:
            self._logger.info(
                ":teddy_bear: Found no stride of model, performed a dummy test for auto-anchor size"
            )
            self.strides = Vec2Box.create_auto_anchor(self.model, image_size)

        self.class_list = class_list
        self.nms_config = nms_config
        self.image_size = image_size

        self._save_predict = True
        # TODO: Add FastModel

        self.predict_loader = FileDataLoader(source=source, image_size=image_size)

    def setup(self, stage):
        self.vec2box = Vec2Box(
            self.strides,
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
        self._logger.info(f"💾 Saved visualize image at {save_image_path}")
