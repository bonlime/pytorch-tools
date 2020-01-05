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
        verbose=True,
    ):
        # base 
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.verbose = verbose
        self.metrics = listify(metrics)

        # data pipeline
        self.input = None
        self.output = None

        # counters
        self.num_epochs = 1
        self.epoch = 0
        self.verbose = verbose
        self.train_loss = AverageMeter()
        self.train_metrics = [AverageMeter(),]
        self.val_loss = AverageMeter()
        self.val_metrics = [AverageMeter(),]
        self.is_train = True
        self.epoch_size = None
        self.step = None
        self.batch_size = 0
        self.metric_meters = [AverageMeter(name=m.name) for m in self.metrics]
        self.loss_meter = AverageMeter("loss")

        # for timer callback
        self.timer = TimeMeter()
        self.__is_frozen = True

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError(f"{self} is a frozen class")
        object.__setattr__(self, key, value)

    @property
    def epoch_log(self):
        return self.epoch + 1