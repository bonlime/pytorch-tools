from torch.cuda.amp import GradScaler
from collections import defaultdict
from typing import Dict, Optional

from .utils import env_rank, env_world_size, AverageMeter, reduce_meter


class RunnerState:
    """
    An object that is used to pass internal state during train/valid/infer.
    This class prohibits creating new attributes after init
    """

    __isfrozen = False

    def __init__(
        self,
        *,
        model=None,
        optimizer=None,
        criterion=None,
        use_fp16=False,
        accumulate_steps=1,
    ):
        # base
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.tb_logger = None  # place for TensorBoard logger

        # make state aware of fp16 and scale. if use_fp16 is False, grad_scaler is NoOp
        self.use_fp16 = use_fp16
        self.grad_scaler = GradScaler(enabled=use_fp16)

        # data pipeline
        self.input = None
        self.output = None

        # counters
        self.num_epochs = 1
        self.epoch = 0
        self.train_loss = AverageMeter("loss")
        self.loss_meter = AverageMeter("loss")
        self.train_metrics: Dict[str, AverageMeter] = defaultdict(AverageMeter())
        self.metric_meters: Dict[str, AverageMeter] = defaultdict(AverageMeter())
        self.val_loss: Optional[AverageMeter] = None
        self.val_metrics: Optional[Dict[str, AverageMeter]] = None
        self.is_train = True
        self.epoch_size = None
        self.batch_size = 0
        # number of steps performed. resets each epoch!
        self.step = None
        # total number of samples seen. usefull to log independentely of batch_size or world_size
        self.global_sample_step = 0
        # number of steps to accumulate
        self.accumulate_steps = accumulate_steps
        # dict for communication between callbacks
        self.communication_dict = dict()

        # for DDP
        self.rank = env_rank()
        self.world_size = env_world_size()

        self.__isfrozen = True

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError(f"{self} is a frozen class")
        object.__setattr__(self, key, value)

    @property
    def epoch_log(self):
        return self.epoch + 1

    def reduce_meters(self):
        """aggregate loss and metrics from all processes"""
        meters = list(self.train_metrics.values()) + [self.train_loss]
        meters = meters + list(self.metric_meters.values()) + [self.loss_meter]
        if self.val_loss is not None:
            meters = meters + list(self.val_metrics.values()) + [self.val_loss]
        for meter in meters:
            reduce_meter(meter)  # NoOp if world_size == 1
