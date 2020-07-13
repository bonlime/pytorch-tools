# Overview
This module contains model runner (very close to `model.fit` in Keras) for **supervised** tasks.  
`Runner` is used to actually run the train loop calling `Callbacks` at appropriate times.  
Mixed precision is also supported. Pass `use_fp16` to Runner to enable it. No more changes are required.

Main idea of this runner is to be as simple as possible. All core functionality is ~100 lines of code.  

## Minimal example
This code will run training for 5 epochs.
```python
from pytorch_tools.fit_wrapper import Runner
model = ... (any model)
optimizer = ... (any optimizer)
criterions = ... (any loss function)
metrics = ... (any metrics or None)
train_loader = ... (dataloader with defined __len__)
runner = Runner(model, optimizer, criterion, metrics)
runner.fit(train_loader, epochs=5)
```

## Real example 
This code will run training for 5 epochs in mixed precision with validation every epoch. 
* `Timer` will measure performance of your dataloader in comparison to model time. 
* `ConsoleLogger` will print training logs to console. 
* `TensorBoard` will log training loss and metrics to TensorBoard. 
* `CheckpointSaver` will save best model according to validation loss.
```python
import pytorch_tools as pt
val_loader = ... (validation dataloader with defined __len__)
runner = pt.fit_wrapper.Runner(
    model,
    optimizer,
    criterion,
    metrics=[pt.metrics.Accuracy(5)],
    callbacks=[
        Timer(),
        ConsoleLogger(),
        TensorBoard(log_dir="/tmp/"),
        CheckpointSaver(save_dir="/tmp/"),
    ]
    use_fp16=True,
)
runner.fit(train_loader, epochs=5, val_loader=val_loader)
```
## Distributed Training
This runner is very simple and requires writing thin wrapper for DDP every time. The preferred (and the only tested way) of running distributed training is using
`python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS <your training code .py>`
`torch.distributed.launch` implicitly sets environmental variables which code for callbacks relies on. Make sure to properly set `RANK` and `WORLD_SIZE`  

## How to
### Add custom step logic  
Monkey patch `Runner._make_step` function with yours  

### Process multiple inputs/outputs
Instead of modifying Runner move this logic inside Loss function.