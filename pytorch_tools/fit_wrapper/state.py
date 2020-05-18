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
        self, *, model=None, optimizer=None, criterion=None, metrics=None,
    ):
        # base
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = listify(metrics)

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


class RunnerStateGAN(RunnerState):
    """
    An object that is used to pass internal state during train/valid/infer.
    This class prohibits creating new attributes after init.
    
    Args:
        model: Generator model
        model_disc: Discriminator model
        optimizer: Generator optimizers
        optimizer_disc: Discriminator optimizers
        criterion: Generator loss
        criterion_disc: Discrimitor loss
        metrics (List): Optional metrics to measure during training. All metrics
            must have `name` attribute. Defaults to None.
    """

    __isfrozen = False

    def __init__(
        self, 
        *, 
        model=None,
        model_disc=None,
        optimizer=None,
        optimizer_disc=None,
        criterion=None,
        criterion_disc=None,
        metrics=None
    ):
        # Base
        self.metrics = listify(metrics)

        # Generator
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        # Disciminator
        self.model_disc = model_disc
        self.optimizer_disc = optimizer_disc
        self.criterion_disc = criterion_disc

        # Data pipeline
        self.input = None
        self.output = None

        # Counters
        self.num_epochs = 1
        self.epoch = 0
        self.train_loss = None
        self.train_loss_disc = None
        self.train_metrics = None

        self.val_loss = None
        self.val_loss_disc = None
        self.val_metrics = None
        self.is_train = True
        self.epoch_size = None
        self.step = None
        self.batch_size = 0

        # By default metrics apply only to Generator Model
        self.metric_meters = [AverageMeter(name=m.name) for m in self.metrics]
        self.loss_meter = AverageMeter("loss")
        self.loss_meter_disc = AverageMeter("loss_disc")

        # for timer callback
        self.timer = TimeMeter()
        self.__is_frozen = True

