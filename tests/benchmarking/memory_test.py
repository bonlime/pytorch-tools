import sys
import time
import torch
import pytest
import argparse
import numpy as np

# from apex import amp
import torchvision as tv
import torch.backends.cudnn as cudnn

import pytorch_tools as pt
from pytorch_tools import models
import pytorch_tools.segmentation_models as pt_sm

# import effdet
# import timm

# sys.path.append("/home/zakirov/repoz/GPU-Efficient-Networks/")
# import GENet


class AverageMeter:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


@pytest.mark.skip("Not meant for pytest")
def test_model(model, forward_only=False):
    optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)
    f_times = []
    fb_times = []
    with cudnn.flags(enabled=True, benchmark=True), torch.set_grad_enabled(not forward_only):
        start = torch.cuda.Event(enable_timing=True)
        f_end = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        def run_once():
            start.record()
            if TARGET is not None and hasattr(model, "is_detection_model"):
                output = model(INP, TARGET)
            else:
                output = model(INP)
            if isinstance(output, (list, tuple)):
                output = output[0]
            f_end.record()
            if forward_only:
                torch.cuda.synchronize()
                return start.elapsed_time(f_end), start.elapsed_time(f_end)
            loss = output.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(f_end), start.elapsed_time(end)

        # benchmark runs. usually much slower than following ones
        for _ in range(2):
            run_once()
        # during cudnn benchmarking a lot of memory is used. we need to reset
        # in order to get max mem alloc by the fastest algorithm
        torch.cuda.reset_max_memory_allocated(0)
        for _ in range(N_RUNS):
            f_meter = AverageMeter()
            fb_meter = AverageMeter()
            for _ in range(RUN_ITERS):
                f_t, fb_t = run_once()
                f_meter.update(f_t)
                fb_meter.update(fb_t)
            f_times.append(f_meter.avg)
            fb_times.append(fb_meter.avg)
        f_times = np.array(f_times)
        fb_times = np.array(fb_times)
    print(
        "Mean of {} runs {} iters each BS={}, SZ={}:\n\t {:.2f}+-{:.2f} msecs Forward. {:.2f}+-{:.2f} msecs Backward. Max memory: {:.2f}Mb. {:.2f} imgs/sec".format(
            N_RUNS,
            RUN_ITERS,
            BS,
            SZ,
            f_times.mean(),
            f_times.std(),
            (fb_times - f_times).mean(),
            (fb_times - f_times).std(),
            torch.cuda.max_memory_allocated(0) / 2**20,
            BS * 1000 / fb_times.mean(),
        )
    )
    del optimizer
    del model


import torch.nn as nn


class DetectionTrainWrapper(nn.Module):
    def __init__(self, modell, size):
        super().__init__()
        self.loss = None
        self.model = modell
        anchors = pt.utils.box.generate_anchors_boxes(size)[0]
        # self.loss = pt.losses.DetectionLoss(anchors)
        self.loss = torch.jit.script(pt.losses.DetectionLoss(anchors))
        self.is_detection_model = True

    def forward(self, inp, target):
        # cls_out, box_out = self.model(inp)
        # return box_out
        loss = self.loss(self.model(inp), target)
        return loss
        # return box_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model Benchmarking")
    parser.add_argument("--forward", "-f", action="store_true", help="Flag to only run forward. Disables grads")
    parser.add_argument("--amp", action="store_true", help="Measure speed using apex mixed precision")
    parser.add_argument("--print", action="store_true", help="Print model")
    parser.add_argument("--torch_amp", action="store_true", help="Measure speed using torch native mixed precision")
    parser.add_argument("--channels_last", action="store_true", help="Use channels last memory format")
    parser.add_argument(
        "--bs",
        default=64,
        type=int,
        help="BS for benchmarking",
    )
    parser.add_argument(
        "--sz",
        default=224,
        type=int,
        help="Size of images for benchmarking",
    )
    args = parser.parse_args()
    # all models are first init to cpu memory to find errors earlier
    # fmt: off
    models_dict = {
        "EffDet0 My": pt.detection_models.efficientdet_d0(match_tf_same_padding=False, pretrained=None),
        "EffDet0 My Wrapped": DetectionTrainWrapper(
            modell=pt.detection_models.efficientdet_d0(match_tf_same_padding=False, pretrained=None),
            size=args.sz,
        ),
        "R50": pt.models.resnet50(),
        # "Simp_R50": pt.models.simpl_resnet50(),
        "R34": pt.models.resnet34(),
        # "R34 est": pt.models.resnet34(norm_layer="estimated_abn"),
        # "R34 abcn": pt.models.resnet34(norm_layer="abcn"),
        # "R34 est abcn": pt.models.resnet34(norm_layer="abcn_micro"),
        # "R34 agn": pt.models.resnet34(norm_layer="agn"),
        # "R34 SE 0.5": pt.models.resnet34(attn_type="se"),
        # "R34 ECA": pt.models.resnet34(attn_type="eca"),
        # "Simp_R34": pt.models.simpl_resnet34(),
        # "Simp_R34 s2d": pt.models.simpl_resnet34(stem_type="s2d"),
        # "Simp_R34 s2d mblnv3 head": pt.models.simpl_resnet34(stem_type="s2d", mobilenetv3_head=True),

        # "Simp_R34 s2d gr=16": pt.models.simpl_resnet34(stem_type="s2d", groups_width=16),
        # "CSP simpl R34": pt.models.csp_simpl_resnet34(stem_type="s2d"),
        # "CSP simpl R34 no x2 tr": pt.models.csp_simpl_resnet34(
        #     stem_type="s2d", x2_transition=False, csp_block_ratio=0.75, groups_width=16
        # ),
        # "CSP simpl R34 0.75 no 1st csp": pt.models.csp_simpl_resnet34(
        #     stem_type="s2d", x2_transition=False, no_first_csp=True, csp_block_ratio=0.75
        # ),
        # "CSP simpl R34 0.75 no x2 tr": pt.models.csp_simpl_resnet34(
        #     stem_type="s2d", csp_block_ratio=0.75, x2_transition=False
        # ),
        # "CSP simpl R34 0.75 no x2 tr gr_w16": pt.models.csp_simpl_resnet34(
        #     stem_type="s2d", csp_block_ratio=0.75, x2_transition=False, groups_width=16,
        # ),
        # "Simp_preR34": pt.models.simpl_preactresnet34(),
        # "Simp_preR34 s2d": pt.models.simpl_preactresnet34(stem_type="s2d"),
        # "Simp_preR34 s2d gr=16": pt.models.simpl_preactresnet34(stem_type="s2d", groups=16),
        # "Simp_preR34 s2d gr_w=16": pt.models.simpl_preactresnet34(stem_type="s2d", groups_width=16),
        # "Darknet like": pt.models.simpl_dark(stem_type="s2d", dim_reduction="stride -> expand"),
        # "Darknet like": pt.models.simpl_dark(stem_type="s2d", dim_reduction="expand -> stride"),

        # "Darknet like nb": pt.models.simpl_dark(stem_type="s2d", dim_reduction="stride -> expand new block", groups_width=1),
        # "Simp ddd": pt.models.b_model.DarkNet(
        #     stem_type="s2d",
        #     # x2_transition=False,
        #     stage_fn=pt.modules.residual.SimpleStage,
        #     block_fn=pt.modules.residual.SimpleBasicBlock,
        #     layers=[3, 4, 6, 3],
        #     channels=[64, 128, 256, 512],
        #     bottle_ratio=1,
            # x2_transition=False,
            # no_first_csp=True,
            # csp_block_ratio=0.75
            # csp_block_ratio=0.75,
            # groups_width=16
            # groups=16,
            # channels=[64, 160, 400, 1024],
            # bottle_ratio=1,
        # ),
        # "Simp csp": pt.models.b_model.DarkNet(
        #     stem_type="s2d",
        #     stage_fn=pt.modules.residual.CrossStage,
        #     block_fn=pt.modules.residual.SimpleBasicBlock,
        #     layers=[3, 4, 6, 3],
        #     channels=[64, 128, 256, 512],
        #     bottle_ratio=1,
        #     no_first_csp=True,
        #     x2_transition=False,
        #     csp_block_ratio=0.75
        # ),
        # "GENet": GENet.genet_normal(pretrained=False),
        # My BNet is almost 10% faster than GENet
        # "Bnet (almost GENet)": pt.models.BNet(
        #     **{
        #         "stage_fns": ["simpl", "simpl", "simpl", "simpl"],
        #         "block_fns": ["Pre_XX", "Pre_XX", "Pre_Custom_2", "Pre_Custom_2"],
        #         "stage_args": [
        #             {"dim_reduction": "stride & expand", "bottle_ratio": 1, "force_residual": True},
        #             {"dim_reduction": "stride & expand", "bottle_ratio": 1, "force_residual": True},
        #             {"bottle_ratio": 1, "dw_str2_kernel_size": 9, "filter_steps": 32},
        #             {"bottle_ratio": 1, "dw_str2_kernel_size": 9, "filter_steps": 128},
        #         ],
        #         # "layers": [1, 2, 6, 5],
        #         # "channels": [128, 192, 640, 1024],
        #         "layers": [2, 4, 8, 2],
        #         "channels": [64, 128, 256, 512],
        #         "stem_width": 32,
        #         "stem_type": "s2d",
        #         "norm_act": "leaky_relu",

        #         # "mobilenetv3_head": False,
        #         "head_width": 2560,
        #         "head_type": "default",
        #         # "head_width": [1536, 2560],
        #         # "head_type": "mlp_2",
        #         # "head_width": [1024, 1536, 2560],
        #         # "head_type": "mlp_3",
        #         "head_norm_act": "swish_hard",
        #     }
        # )
        # "Eff B0": pt.models.efficientnet_b0(),
        # "TR50": pt.models.tresnetm(norm_layer="abn"),
        # "D53 timm": timm.models.darknet53(),
        # "CSPD53 timm": timm.models.cspdarknet53(),
        # "CSPR50 timm": timm.models.cspresnet50(),
        # "CSPR50d timm": timm.models.cspresnet50d(),
        # "CSPX50 timm": timm.models.cspresnext50(),
        # "TRes ": pt.models.tresnetm(),
        # "R50": pt.models.resnet50(),
        # "ResNet50": pt.models.resnet50(),
    }
    # fmt: on
    segm_models_dict = {
        # "Unet Resnet34": pt_sm.Unet("resnet34"),
    }

    print("Initialized models")
    BS = args.bs
    SZ = args.sz
    N_RUNS = 5
    RUN_ITERS = 10
    INP = torch.ones((BS, 3, SZ, SZ), requires_grad=not args.forward).cuda(0)
    # TARGET = None
    TARGET = torch.randint(0, 50, (BS, 20, 5)).cuda().float()

    for name, model in models_dict.items():
        print(f"{name} {count_parameters(model) / 1e6:.2f}M params")
        model = model.cuda(0)
        if args.amp:
            model = amp.initialize(model, verbosity=0, opt_level="O1")
            INP = INP.half()
        if args.torch_amp:
            INP = INP.half()
            model = model.half()
            model.forward = torch.cuda.amp.autocast()(model.forward)
        if args.forward:
            model.eval()
        if args.print:
            print(model)
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)
            INP = INP.to(memory_format=torch.channels_last)
        # with torch.cuda.amp.autocast(enabled=args.torch_amp):
        test_model(model, forward_only=args.forward)

    # now test segmentation models
    INP = torch.ones((BS, 3, SZ, SZ), requires_grad=True).cuda(0)

    for name, model in segm_models_dict.items():
        enc_params = count_parameters(model.encoder) / 1e6
        total_params = count_parameters(model) / 1e6
        print(
            f"{name}. Encoder {enc_params:.2f}M. Decoder {total_params - enc_params:.2f}M. Total {total_params:.2f}M params"
        )
        model = model.cuda(0)
        if args.amp:
            model = amp.initialize(model, verbosity=0, opt_level="O1")
            INP = INP.half()
        if args.forward:
            model.eval()
        test_model(model, forward_only=args.forward)
