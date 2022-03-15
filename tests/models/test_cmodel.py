import yaml
import torch
import pytest
import torch.nn as nn

from pytorch_tools.models import CModel as OriginalCModel
from pytorch_tools.models.cmodel import _update_dict

from modules import Concat

INP = torch.randn(1, 3, 16, 16)


class CModel(OriginalCModel):
    @staticmethod
    def module_from_name(name):
        return eval(name)


def test_update_dict():
    to_dict = dict(foo=1, bar=dict(arg1=10, arg2=20, arg3=dict(deep_arg1=100, deep_arg2=200)))
    from_dict = dict(bar=dict(arg2=25, arg3=dict(deep_arg2=242)))
    expected_dict = dict(foo=1, bar=dict(arg1=10, arg2=25, arg3=dict(deep_arg1=100, deep_arg2=242)))

    res_dict = _update_dict(to_dict, from_dict)
    assert res_dict == expected_dict


def test_linear_config():
    # parsing from dict
    dict_config = """
    layer_config:
        - module: nn.Conv2d
          args: [3, 32, 7, 2, 3]
          kwargs:
            bias: False
        - module: nn.Conv2d
          args: [32, 32, 3]
          kwargs:
            padding: 1
            # example of passing string
            padding_mode: "'circular'"
    """
    dict_layer_config = yaml.safe_load(dict_config)["layer_config"]
    dict_model = CModel(dict_layer_config)
    assert dict_model(INP).shape == (1, 32, 8, 8)


def test_external_module():
    config = """
    layer_config:
        - module: nn.Conv2d
          args: [3, 8, 1]
          tag: os2
        - module: nn.Conv2d
          args: [8, 16, 1]
        - module: Concat
          inputs: [_prev_, os2]
    """
    layer_config = yaml.safe_load(config)["layer_config"]
    model = CModel(layer_config)
    assert model(INP).shape == (1, 8 + 16, 16, 16)


def test_fpn_config():
    fpn_config = """
    layer_config:
        - module: nn.Conv2d
          args: [3, 8, 7, 2, 3]
          tag: os2
        - module: nn.Conv2d
          args: [8, 16, 7, 2, 3]
          tag: os4
        - module: nn.Conv2d
          args: [16, 32, 7, 2, 3]
        # this is just an example of working `inputs` in real code some logic would be wrapped in separate class
        # to avoid such monstrous and hard-to-understand configs
        - module: torch.nn.Upsample
          kwargs:
            scale_factor: 4
          tag: os8_up4
        - module: torch.nn.Upsample
          kwargs:
            scale_factor: 2
          inputs: [os4]
        - module: Concat
          inputs: [_prev_, os8_up4, os2]
    """
    fpn_layer_config = yaml.safe_load(fpn_config)["layer_config"]
    fpn_model = CModel(fpn_layer_config)
    assert fpn_model(INP).shape == (1, 32 + 16 + 8, 8, 8)


def test_unet_config():
    unet_config = """
    layer_config:
        - module: nn.Conv2d
          args: [3, 8, 7, 2, 3]
          tag: os2
        - module: nn.Conv2d
          args: [8, 16, 7, 2, 3]
          tag: os4
        - module: nn.Conv2d
          args: [16, 32, 7, 2, 3]
        # Example of working `inputs`. in real code logic will be in separate class
        # to avoid such monstrous and hard-to-understand configs
        - module: torch.nn.Upsample
          kwargs:
            scale_factor: 2
        - module: Concat
          inputs: [_prev_, os4]
        - module: torch.nn.Upsample
          kwargs:
            scale_factor: 2
        - module: Concat
          inputs: [_prev_, os2]
    """
    unet_layer_config = yaml.safe_load(unet_config)["layer_config"]
    unet_model = CModel(unet_layer_config)
    assert unet_model(INP).shape == (1, 32 + 16 + 8, 8, 8)


def test_unused_tag_config():
    config = """
    layer_config:
        - module: nn.Conv2d
          args: [3, 8, 1]
          tag: os2
        - module: nn.Conv2d
          args: [8, 16, 1]
          tag: os4
    """
    layer_config = yaml.safe_load(config)["layer_config"]
    model = CModel(layer_config)
    assert model(INP).shape == (1, 16, 16, 16)


def test_extra_kwargs():
    extra_config = """
    layer_config:
        - module: nn.Conv2d
          kwargs:
            in_channels: 3
            out_channels: 32
        - module: nn.Conv2d
          kwargs:
            in_channels: 32
            out_channels: 48
    extra_kwargs:
        nn.Conv2d:
            kernel_size: 3
            padding: 1
    """
    extra_config = yaml.safe_load(extra_config)  # ["layer_config"]
    extra_model = CModel(extra_config["layer_config"], extra_config["extra_kwargs"])
    assert extra_model(INP).shape == (1, 48, 16, 16)


def test_eval():
    extra_config = """
    layer_config:
        - module: nn.Conv2d
        - module: nn.Conv2d
    extra_kwargs:
        nn.Conv2d:
            kernel_size: 3
            padding: 1
            in_channels: 3
            out_channels: 3
    """
    extra_config = yaml.safe_load(extra_config)  # ["layer_config"]
    extra_model = CModel(extra_config["layer_config"], extra_config["extra_kwargs"])
    # check that default Module.eval() works
    extra_model = extra_model.eval().requires_grad_(False)
    assert extra_model(INP).shape == (1, 3, 16, 16)


def test_jit_script_sequential():
    config_str = """
    layer_config:
        - module: nn.Conv2d
        - module: nn.Conv2d
    extra_kwargs:
        nn.Conv2d:
            kernel_size: 3
            padding: 1
            in_channels: 3
            out_channels: 3
    """
    config = yaml.safe_load(config_str)
    model = CModel(config["layer_config"], config["extra_kwargs"])
    jit_model = torch.jit.script(model)
    assert jit_model(INP).shape == (1, 3, 16, 16)


def test_jit_trace_custom():
    config_str = """
    layer_config:
        - { module: nn.Conv2d, tag: os2 }
        - module: nn.Conv2d
        - module: Concat
          inputs: [_prev_, os2]
    extra_kwargs:
        nn.Conv2d:
            in_channels: 3
            out_channels: 3
            kernel_size: 3
            padding: 1
    """
    config = yaml.safe_load(config_str)
    model = CModel(config["layer_config"], config["extra_kwargs"])
    trace_model = torch.jit.trace(model, INP)
    assert trace_model(INP).shape == (1, 3 + 3, 16, 16)


def test_jit_script_custom():
    config_str = """
    layer_config:
        - { module: nn.Conv2d, tag: os2 }
        - module: nn.Conv2d
        - module: Concat
          inputs: [_prev_, os2]
    extra_kwargs:
        nn.Conv2d:
            in_channels: 3
            out_channels: 3
            kernel_size: 3
            padding: 1
    """
    config = yaml.safe_load(config_str)
    model = CModel(config["layer_config"], config["extra_kwargs"])
    script_model = torch.jit.script(model)
    assert script_model(INP).shape == (1, 3 + 3, 16, 16)
