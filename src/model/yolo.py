import logging
from pathlib import Path
from typing import MutableSequence, cast

import torch
from torch import nn

from src.config.config import BlockConfig, LayerConfig, ModelConfig
from src.utils.module_utils import get_layer_map

logger = logging.getLogger("yolo")


class YOLOLayer(nn.Module):
    source: int | str | list[int]
    output: bool
    tags: str
    layer_type: str
    usable: bool
    external: dict | None


class YOLO(nn.Module):
    """
    A preliminary YOLO (You Only Look Once) model class still under development.

    Parameters:
        model_cfg: Configuration for the YOLO model. Expected to define the layers,
                   parameters, and any other relevant configuration details.
    """

    def __init__(self, model_cfg: ModelConfig, class_num: int = 80):
        super(YOLO, self).__init__()

        self.num_classes = class_num
        self.layer_map = get_layer_map()  # Get the map Dict[str: Module]
        self.model = nn.ModuleList()

        self.reg_max = model_cfg["anchor"].get("reg_max", 16)
        self.build_model(model_cfg["model"])

    def build_model(self, model_arch: dict[str, BlockConfig]):
        self.layer_index = {}
        output_dim, layer_idx = [3], 1
        logger.info(":tractor: Building YOLO")
        for arch_name in model_arch:
            if not model_arch[arch_name]:
                continue
            logger.info(f"  :building_construction:  Building {arch_name}")
            for layer_idx, layer_spec in enumerate(
                model_arch[arch_name], start=layer_idx
            ):
                layer_type, layer_info = next(iter(layer_spec.items()))
                layer_args = layer_info.get("args", {})

                # Get input source
                source = self.get_source_idx(layer_info.get("source", -1), layer_idx)

                # Find in channels
                if any(
                    module in layer_type
                    for module in ["Conv", "ELAN", "ADown", "AConv", "CBLinear"]
                ):
                    assert not isinstance(source, list)
                    layer_args["in_channels"] = output_dim[source]

                if any(
                    module in layer_type
                    for module in ["Detection", "Segmentation", "Classification"]
                ):
                    if isinstance(source, list):
                        layer_args["in_channels"] = [output_dim[idx] for idx in source]
                    else:
                        layer_args["in_channel"] = output_dim[source]
                    layer_args["num_classes"] = self.num_classes
                    layer_args["reg_max"] = self.reg_max

                # create layers
                layer = self.create_layer(layer_type, source, layer_info, **layer_args)
                if layer.tags:
                    if layer.tags in self.layer_index:
                        raise ValueError(f"Duplicate tag '{layer.tags}' found.")
                    self.layer_index[layer.tags] = layer_idx

                out_channels = self.get_out_channels(
                    layer_type, layer_args, output_dim, source
                )
                output_dim.append(out_channels)
                setattr(layer, "out_c", out_channels)
                self.model.append(layer)

            layer_idx += 1

    def forward(self, x, external: dict | None = None, shortcut: str | None = None):
        y = {0: x, **(external or {})}
        output = dict()
        for index, layer in enumerate(self.model, start=1):
            layer = cast(YOLOLayer, layer)
            if isinstance(layer.source, list):
                model_input = [y[idx] for idx in layer.source]
            else:
                model_input = y[layer.source]

            external_input = (
                {source_name: y[source_name] for source_name in layer.external}
                if layer.external
                else {}
            )

            x = layer(model_input, **external_input)
            y[-1] = x
            if layer.usable:
                y[index] = x

            if layer.output:
                output[layer.tags] = x
                if layer.tags == shortcut:
                    return output

        return output

    def get_out_channels(
        self,
        layer_type: str,
        layer_args: dict,
        output_dim: list,
        source: int | list,
    ):
        if "out_channels" in layer_args:
            return layer_args["out_channels"]

        if layer_type == "CBFuse":
            assert isinstance(source, list)
            return output_dim[source[-1]]
        if isinstance(source, int):
            return output_dim[source]
        if isinstance(source, list):
            return sum(output_dim[idx] for idx in source)

    def _get_source_idx(self, source: int | str, layer_idx: int) -> int:
        if isinstance(source, str):
            source = self.layer_index[source]
        assert not isinstance(source, str)

        if source < -1:
            source += layer_idx
        if source > 0:  # Using Previous Layer's Output
            layer = cast(YOLOLayer, self.model[source - 1])
            layer.usable = True
        return source

    def get_source_idx(
        self, source: int | str | list[int | str], layer_idx: int
    ) -> int | list[int]:
        # TODO: change to list (MutableSequence for ListConfig)
        if isinstance(source, MutableSequence):
            return [self._get_source_idx(index, layer_idx) for index in source]
        return self._get_source_idx(source, layer_idx)

    def create_layer(
        self, layer_type: str, source: int | list, layer_info: LayerConfig, **kwargs
    ) -> YOLOLayer:
        if layer_type in self.layer_map:
            layer = self.layer_map[layer_type](**kwargs)
            setattr(layer, "layer_type", layer_type)
            setattr(layer, "source", source)
            setattr(layer, "in_c", kwargs.get("in_channels", None))
            setattr(layer, "tags", layer_info.get("tags", None))
            # setattr(layer, "output", False)
            setattr(layer, "output", layer_info.get("output", False))
            # setattr(layer, "external", [])
            setattr(layer, "external", layer_info.get("external", []))
            setattr(layer, "usable", 0)
            return layer
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def save_load_weight(self, weights: Path):
        """
        Update the model's weights with the provided weights.

        args:
            weights: A OrderedDict containing the new weights.
        """
        if isinstance(weights, Path):
            weights = torch.load(
                weights, map_location=torch.device("cpu"), weights_only=False
            )
        if "state_dict" in weights:
            weights = {
                name.removeprefix("model.model."): key
                for name, key in weights["state_dict"].items()
            }
        model_state_dict = self.model.state_dict()

        # TODO1: autoload old version weight
        # TODO2: weight transform if num_class difference

        error_dict = {"Mismatch": set(), "Not Found": set()}
        for model_key, model_weight in model_state_dict.items():
            if model_key not in weights:
                error_dict["Not Found"].add(tuple(model_key.split(".")[:-2]))
                continue
            if model_weight.shape != weights[model_key].shape:
                error_dict["Mismatch"].add(tuple(model_key.split(".")[:-2]))
                continue
            model_state_dict[model_key] = weights[model_key]

        for error_name, error_set in error_dict.items():
            error_dict = dict()
            for layer_idx, *layer_name in error_set:
                if layer_idx not in error_dict:
                    error_dict[layer_idx] = [".".join(layer_name)]
                else:
                    error_dict[layer_idx].append(".".join(layer_name))
            for layer_idx, layer_name in error_dict.items():
                layer_name.sort()
                logger.warning(
                    f":warning: Weight {error_name} for Layer {layer_idx}: {', '.join(layer_name)}"
                )

        self.model.load_state_dict(model_state_dict)
