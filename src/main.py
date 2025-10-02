import logging
from pathlib import Path

from lightning import Trainer
from pydantic import TypeAdapter
from rich.logging import RichHandler

from src.config.config import ModelConfig, NMSConfig, class_list
from src.tools.solver import InferenceModel


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

    with open(f"src/config/model/{model_name}.json", "r") as f:
        model_conf = TypeAdapter(ModelConfig).validate_json(
            f.read(),
            strict=True,
        )

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
