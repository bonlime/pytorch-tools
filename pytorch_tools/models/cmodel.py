"""C(onfig)Model constructor"""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable

import torch
import torch.nn as nn


def _update_dict(to_dict: Dict, from_dict: Dict) -> Dict:
    """close to `to_dict.update(from_dict)` but correctly updates internal dicts"""
    for k, v in from_dict.items():
        if hasattr(v, "keys") and k in to_dict.keys():
            # recursively update internal dicts
            _update_dict(to_dict[k], v)
        else:
            to_dict[k] = v
    return to_dict


def listify(p: Any) -> Iterable:
    if p is None:
        p = []
    elif not isinstance(p, Iterable):
        p = [p]
    return p


@dataclass
class ModuleStructure:
    """Dataclass that defines single model layer"""

    module: Union[str, nn.Module]
    args: List = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    repeat: int = 1
    # List of tags to use as input.
    inputs: List[str] = field(default_factory=lambda: ["_prev_"])
    tag: Optional[str] = None


class InputWrapper(nn.Module):
    """This wrapper is needed to make the CModel scriptable"""

    def __init__(self, block, n_inputs=1):
        super().__init__()
        self.block = block
        self.n_inputs = n_inputs
        if n_inputs == 1:
            self.forward = self.forward_1
        else:
            self.forward = self.forward_many

    def forward_1(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.block(x[0])

    def forward_many(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.block(x)


class CModel(nn.Sequential):
    """
    Abstract builder that can be used to create models from config.

    Args:
        layer_config: List of all model layers. Layer kwargs overwrite `extra_kwargs`!
        extra_kwargs: If given, would pass extra parameters to all modules with corresponding name.
    """

    def __init__(
        self,
        layer_config: List[Union[List, Dict]],
        extra_kwargs: Dict[str, Dict] = None,
    ):
        layer_config = [ModuleStructure(**layer) for layer in layer_config]

        if extra_kwargs is not None:
            self._update_config_with_extra_params(layer_config, extra_kwargs)
        layers, self.saved_layers_idx = self._parse_config(layer_config)
        super().__init__(*layers)
        #  Implement custom forward only if model is non-linear.
        if len(self.saved_layers_idx) > 0:
            self.forward = self.custom_forward

    @staticmethod
    def _update_config_with_extra_params(layer_config: List[ModuleStructure], extra_kwargs: Dict[str, Dict]):
        for extra_layer_name, extra_layer_kwargs in extra_kwargs.items():
            for layer in layer_config:
                if layer.module == extra_layer_name:
                    # kwargs from layer should overwrite global extra_kwargs
                    layer.kwargs = _update_dict(deepcopy(extra_layer_kwargs), layer.kwargs)

    def _parse_config(self, layer_config: List[ModuleStructure]) -> Tuple[nn.ModuleList, List[int]]:
        saved_layers_idx = []
        layers = []
        # skip unused tags
        used_tags = set([inp for layer in layer_config for inp in layer.inputs])
        tag_to_idx = {l.tag: idx for idx, l in enumerate(layer_config) if l.tag is not None and l.tag in used_tags}
        tag_to_idx["_prev_"] = -1

        for layer_idx, l in enumerate(layer_config):
            l.module = self._maybe_eval(l.module)
            # eval all strings by default. if you need to pass a string write "'my string'" in your config
            l.args = [self._maybe_eval(i) for i in listify(l.args)]
            l.kwargs = {k: self._maybe_eval(v) for k, v in l.kwargs.items()}

            m = l.module(*l.args, **l.kwargs)
            if l.repeat > 1:
                m = nn.Sequential(*[l.module(*l.args, **l.kwargs) for _ in range(l.repeat)])

            if len(tag_to_idx) > 1:  # only wrap if not sequential
                m = InputWrapper(m, len(l.inputs))

            # add some information about from/idx to module
            m.input_indexes = [tag_to_idx[inp] for inp in l.inputs]
            m.idx = layer_idx

            layers.append(m)
            # output of which layers do we need. skip -1 because its' output we would have anyway
            saved_layers_idx.extend(idx for idx in m.input_indexes if idx != -1)
        return nn.ModuleList(layers), saved_layers_idx

    def custom_forward(self, x: torch.Tensor):
        saved_outputs: List[torch.Tensor] = []
        for layer in self.children():
            inp: List[torch.Tensor] = [x if j == -1 else saved_outputs[j] for j in layer.input_indexes]
            x = layer(inp)
            # append None even if don't need this output in order to preserve ordering
            saved_outputs.append(x if layer.idx in self.saved_layers_idx else torch.empty(0))
        return x

    def _maybe_eval(self, name: str):
        return self.module_from_name(name) if isinstance(name, str) else name

    @staticmethod
    def module_from_name(name):
        return eval(name)
