from loguru import logger
from torch.cuda.amp import GradScaler
from pytorch_tools.utils import misc as utils


class RunnerState:
    """
    An object that is used to pass internal state during train/valid/infer.
    This class prohibits creating new attributes after init
    """

    __isfrozen = False

    def __init__(
        self, *, model=None, optimizer=None, criterion=None, use_fp16=False,
    ):
        # base
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger = logger

        # make state aware of fp16 and scale. if use_fp16 is False, grad_scaler is NoOp
        self.use_fp16 = use_fp16
        self.grad_scaler = GradScaler(enabled=use_fp16)

        # data pipeline
        self.input = None
        self.output = None

        # counters
        self.num_epochs = 1
        self.epoch = 0
        self.train_loss = None
        self.train_metrics = None
        self.val_loss = None
        self.val_metrics = None
        self.is_train = True
        self.epoch_size = None
        self.step = None
        self.batch_size = 0
        self.metric_meters = {}
        self.loss_meter = utils.AverageMeter("loss")

        # for DDP
        self.rank = utils.env_rank()
        self.world_size = utils.env_world_size()

        self.__is_frozen = True

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
            utils.reduce_meter(meter)  # NoOp if world_size == 1
