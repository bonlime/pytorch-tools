import torch
from copy import copy
from apex import amp
from .state import RunnerStateGAN
from .gan_callbacks import CallbacksGAN
from .callbacks import ConsoleLogger
from ..utils.misc import to_numpy
from .wrapper import Runner


class GANRunner(Runner):
    """
    Args:
        model_gen: Generator model
        model_disc: Discriminator model
        optimizer_gen: Generator optimizers
        optimizer_disc: Discriminator optimizers
        criterion_gen: Generator loss
        criterion_disc: Discrimitor loss
        metrics (List): Optional metrics to measure during training. All metrics
            must have `name` attribute. Defaults to None.
        callbacks (List): List of Callbacks to use. Defaults to ConsoleLogger().
        gradient_clip_val (float): Gradient clipping value. 0 means no clip. Causes ~5% training slowdown

    """
    def __init__(
        self,
        model_gen, 
        model_disc,
        optimizer_gen,
        optimizer_disc,
        criterion_gen,
        criterion_disc,
        metrics=None,
        callbacks=ConsoleLogger(),
        gradient_clip_val=0
    ):

        if not hasattr(amp._amp_state, "opt_properties"):
            model_optimizer_gen = amp.initialize(model_gen, optimizer_gen, enabled=False)
            model_gen, optimizer_gen = (model_optimizer_gen, None) if optimizer_gen is None else model_optimizer_gen
            model_optimizer_disc = amp.initialize(model_disc, optimizer_disc, enabled=False)
            model_disc, optimizer_disc = (model_optimizer_disc, None) if optimizer_disc is None else model_optimizer_disc
    
        self.state = RunnerStateGAN(
            model=model_gen,
            model_disc=model_disc,
            optimizer=optimizer_gen,
            optimizer_disc=optimizer_disc,
            criterion=criterion_gen,
            criterion_disc=criterion_disc,
            metrics=metrics
        )
        self.callbacks = CallbacksGAN(callbacks)
        self.callbacks.set_state(self.state)
        self.gradient_clip_val = gradient_clip_val

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
            self.state.model_disc.train()
            self._run_loader(train_loader, steps=steps_per_epoch)
            self.state.train_loss = copy(self.state.loss_meter)
            self.state.train_loss_disc = copy(self.state.loss_meter_disc)
            self.state.train_metrics = [copy(m) for m in self.state.metric_meters]

            if val_loader is not None:
                self.evaluate(val_loader, steps=val_steps)
                self.state.val_loss = copy(self.state.loss_meter)
                self.state.val_loss_disc = copy(self.state.loss_meter_disc)
                self.state.val_metrics = [copy(m) for m in self.state.metric_meters]
            self.callbacks.on_epoch_end()
        self.callbacks.on_end()

    def evaluate(self, loader, steps=None):
        self.state.is_train = False
        self.state.model.eval()
        self.state.model_disc.eval()
        self._run_loader(loader, steps=steps)
        return self.state.loss_meter.avg, [m.avg for m in self.state.metric_meters]

    def _make_step_generator(self):
        self.state.optimizer.zero_grad()
        data, target = self.state.input
        output = self.state.model(data)
        self.state.output = output
        loss = self.state.criterion(self.state.model, self.state.model_disc, output, target)
        if self.state.is_train:
            self.state.optimizer.zero_grad()
            with amp.scale_loss(loss, self.state.optimizer) as scaled_loss:
                scaled_loss.backward()
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.state.model.parameters(), self.gradient_clip_val)
                torch.nn.utils.clip_grad_norm_(self.state.model_disc.parameters(), self.gradient_clip_val)
            self.state.optimizer.step()
            torch.cuda.synchronize()

        # update metrics
        self.state.loss_meter.update(to_numpy(loss))
        for metric, meter in zip(self.state.metrics, self.state.metric_meters):
            meter.update(to_numpy(metric(output, target).squeeze()))

    def _make_step_discriminator(self):
        self.state.optimizer_disc.zero_grad()
        data, target = self.state.input
        output = self.state.model(data)
        self.state.output = output
        loss = self.state.criterion_disc(self.state.model, self.state.model_disc, output, target)
        if self.state.is_train:
            self.state.optimizer_disc.zero_grad()
            with amp.scale_loss(loss, self.state.optimizer_disc) as scaled_loss:
                scaled_loss.backward()
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.state.model_disc.parameters(), self.gradient_clip_val)
            self.state.optimizer_disc.step()
            torch.cuda.synchronize()

        # Update loss
        self.state.loss_meter_disc.update(to_numpy(loss))

    def _run_loader(self, loader, steps=None):
        self.state.loss_meter.reset()
        self.state.timer.reset()
        for metric in self.state.metric_meters:
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
                self._make_step_discriminator()
                self._make_step_generator()
                self.callbacks.on_batch_end()
        self.callbacks.on_loader_end()
        return
