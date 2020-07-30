import copy
import torch
from torch.cuda import amp
from pytorch_tools.fit_wrapper.state import RunnerState
from pytorch_tools.fit_wrapper.callbacks import Callbacks
from pytorch_tools.fit_wrapper.callbacks import ConsoleLogger
from pytorch_tools.utils.misc import to_numpy


class Runner:
    """

    Args:
        model: model
        optimizer: optimizer
        criterion: Loss used for training
        callbacks (List): List of Callbacks to use. Defaults to ConsoleLogger().
        gradient_clip_val (float): Gradient clipping value. 0 means no clip. Causes ~5% training slowdown
        accumulate_steps (int): if > 1 uses gradient accumulation across iterations to simulate larger batch size
        use_fp16 (bool): Flag which enables mixed precision
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        callbacks=ConsoleLogger(),
        gradient_clip_val=0,
        accumulate_steps=1,
        use_fp16=False,
    ):
        super().__init__()
        self.state = RunnerState(
            model=model, optimizer=optimizer, criterion=criterion, use_fp16=use_fp16
        )
        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_state(self.state)
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_steps = accumulate_steps

    def fit(
        self, train_loader, steps_per_epoch=None, val_loader=None, val_steps=None, epochs=1, start_epoch=0,
    ):
        """
        Args:
            train_loader: DataLoader with defined `len` and `batch_size`
            steps_per_epoch (int): How many steps to count as an epochs. Useful
                when epoch is very long or it's not clearly defined. Defaults to None.
            val_loader: Validation DataLoader with defined `len` and `batch_size` Defaults to None.
            val_steps (int): same as `steps_per_epoch` but for val data. Defaults to None.
            epochs (int): Number of epochs to train for. Defaults to 1.
            start_epoch (int): From which epoch to start. Useful on restarts. Defaults to 0.
        """
        self.state.num_epochs = epochs
        self.state.batch_size = train_loader.batch_size if hasattr(train_loader, "batch_size") else 1
        self.callbacks.on_begin()
        for epoch in range(start_epoch, epochs):
            self.state.is_train = True
            self.state.epoch = epoch
            self.callbacks.on_epoch_begin()
            self.state.model.train()
            self._run_loader(train_loader, steps=steps_per_epoch)
            self.state.train_loss = copy.copy(self.state.loss_meter)
            self.state.train_metrics = copy.deepcopy(self.state.metric_meters)

            if val_loader is not None:
                self.evaluate(val_loader, steps=val_steps)
                self.state.val_loss = copy.copy(self.state.loss_meter)
                self.state.val_metrics = copy.deepcopy(self.state.metric_meters)
            self.state.reduce_meters()
            self.callbacks.on_epoch_end()
        self.callbacks.on_end()

    def evaluate(self, loader, steps=None):
        self.state.is_train = False
        self.state.model.eval()
        self._run_loader(loader, steps=steps)
        return self.state.loss_meter.avg, [m.avg for m in self.state.metric_meters.values()]

    def _make_step(self):
        data, target = self.state.input

        with amp.autocast(self.state.use_fp16):
            output = self.state.model(data)
            loss = self.state.criterion(output, target)
        self.state.output = output

        if self.state.is_train:
            # backward for every batch
            self.state.grad_scaler.scale(loss / self.accumulate_steps).backward()
            # everything else only before making step
            if self.state.step % self.accumulate_steps == 0:

                if self.gradient_clip_val > 0:
                    self.state.grad_scaler.unscale_(self.state.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.state.model.parameters(), self.gradient_clip_val)

                self.state.grad_scaler.step(self.state.optimizer)
                self.state.grad_scaler.update()
                self.state.optimizer.zero_grad()
            torch.cuda.synchronize()

        # Update loss
        self.state.loss_meter.update(to_numpy(loss))
        # Metrics are now updated inside callbacks

    def _run_loader(self, loader, steps=None):
        self.state.loss_meter.reset()
        for metric in self.state.metric_meters.values():
            metric.reset()
        self.state.epoch_size = steps or len(loader)  # steps overwrites len
        self.callbacks.on_loader_begin()
        with torch.set_grad_enabled(self.state.is_train):
            for i, batch in enumerate(loader):
                if i == self.state.epoch_size:
                    break
                self.state.step = i
                self.state.input = batch
                self.callbacks.on_batch_begin()
                self._make_step()
                self.callbacks.on_batch_end()
        self.callbacks.on_loader_end()
        return
