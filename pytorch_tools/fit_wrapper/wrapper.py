import torch
from copy import copy
from apex import amp
from .state import RunnerState
from .callbacks import Callbacks
from .callbacks import ConsoleLogger
from ..utils.misc import to_numpy


class Runner:
    def __init__(
        self, model, optimizer, criterion, metrics=None, callbacks=ConsoleLogger(), verbose=True
    ):
        super().__init__()
        self.state = RunnerState(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            metrics=metrics,
            verbose=verbose,
        )
        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_state(self.state)

    def fit(
        self,
        train_loader,
        steps_per_epoch=None,
        val_loader=None,
        val_steps=None,
        epochs=1,
        start_epoch=0,
    ):
        self.state.num_epochs = epochs
        self.state.batch_size = (
            train_loader.batch_size if hasattr(train_loader, "batch_size") else 1
        )
        self.callbacks.on_train_begin()
        for epoch in range(start_epoch, epochs):
            self.state.is_train = True
            self.state.epoch = epoch
            self.callbacks.on_epoch_begin()
            self.state.model.train()
            self._run_loader(train_loader, steps=steps_per_epoch)
            self.state.train_loss = copy(self.state.loss_meter)
            self.state.train_metrics = [copy(m) for m in self.state.metric_meters]

            if val_loader is not None:
                self.evaluate(val_loader, steps=val_steps)
                self.state.val_loss = copy(self.state.loss_meter)
                self.state.val_metrics = [copy(m) for m in self.state.metric_meters]
            self.callbacks.on_epoch_end()
        self.callbacks.on_train_end()

    def evaluate(self, loader, steps=None):
        self.state.is_train = False
        self.state.model.eval()
        self._run_loader(loader, steps=steps)
        return self.state.loss_meter.avg, [m.avg for m in self.state.metric_meters]

    def _make_step(self):
        data, target = self.state.input
        output = self.state.model(data)
        self.state.output = output
        loss = self.state.criterion(output, target)
        if self.state.is_train:
            self.state.optimizer.zero_grad()
            with amp.scale_loss(loss, self.state.optimizer) as scaled_loss:
                scaled_loss.backward()
            # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.state.optimizer.step()
            torch.cuda.synchronize()

        # update metrics
        self.state.loss_meter.update(to_numpy(loss))
        for metric, meter in zip(self.state.metrics, self.state.metric_meters):
            meter.update(to_numpy(metric(output, target).squeeze()))

    def _run_loader(self, loader, steps=None):
        self.state.loss_meter.reset()
        self.state.timer.reset()
        for metric in self.state.metric_meters:
            metric.reset()
        self.state.epoch_size = steps or len(loader)  # steps overwrites len
        with torch.set_grad_enabled(self.state.is_train):
            for i, batch in enumerate(loader):
                if i == self.state.epoch_size:
                    break
                self.state.step = i
                self.state.input = batch
                self.callbacks.on_batch_begin()
                self._make_step()
                self.callbacks.on_batch_end()
        return
