# from typing import Dict, Optional  # isort:skip
# from collections import defaultdict, OrderedDict
# from pathlib import Path

from ..utils.misc import listify
from ..utils.misc import AverageMeter
from ..utils.misc import TimeMeter


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
        metrics=None,
        # scheduler=None,
        # logdir: str = None,
        # stage: str = "infer",
        # num_epochs: int = 1,
        # main_metric: str = "loss",
        # minimize_metric: bool = True,
        # valid_loader: str = "valid",
        verbose=True,
        # checkpoint_data: Dict = None,
        # batch_consistant_metrics: bool = True,
        # **kwargs
    ):
        # self.logdir = Path(logdir) if logdir is not None else None
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        # self.scheduler = scheduler

        # special info
        # self.stage = stage
        # self.device = device
        # self.loader_name = None
        # self.phase = None

        # data pipeline
        self.input = None
        self.output = None

        # counters
        self.num_epochs = 1
        self.epoch = 0
        self.verbose = verbose
        self.train_loss = None
        self.train_metrics = None
        self.val_loss = None
        self.val_metrics = None
        self.is_train = True
        self.ep_size = None
        self.step = None

        self.batch_size = 0
        # self.loader_len = 0
        # self.batch_size = 0
        # self.step = 0
        # self.epoch = 0
        # self.stage_epoch = 0
        # self.num_epochs = num_epochs
        self.metrics = listify(metrics)
        self.metric_meters = [AverageMeter(name=m.name) for m in self.metrics]
        self.loss_meter = AverageMeter("loss")

        # for timer callback
        self.timer = TimeMeter()
        # metrics & logging
        # self.main_metric = main_metric
        # self.minimize_metric = minimize_metric
        # self.valid_loader = valid_loader
        # self.metrics = MetricManager(
        #     valid_loader=valid_loader,
        #     main_metric=main_metric,
        #     minimize=minimize_metric,
        #     batch_consistant_metrics=batch_consistant_metrics
        # )
        self.verbose = verbose
        # self.loggers = OrderedDict()
        # self.timer = TimerManager()

        # base metrics
        # single_optimizer = isinstance(optimizer, Optimizer)
        # self.lr = None if single_optimizer else defaultdict(lambda: None)
        # self.momentum = None if single_optimizer else defaultdict(lambda: None)
        # self.loss = None

        # extra checkpoint data for saving in checkpoint files
        # self.checkpoint_data = checkpoint_data or {}

        # other
        # self.need_backward = False
        # self.early_stop = False
        # for k, v in kwargs.items():
            # setattr(self, k, v)

        # self.exception: Optional[Exception] = None
        # self.need_reraise_exception: bool = True

        self.__is_frozen = True

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError(f"{self} is a frozen class")
        object.__setattr__(self, key, value)

    # @property
    # def epoch_log(self):
    #     return self.epoch + 1
