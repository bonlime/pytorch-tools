# Overview
This module contains model runner (very close to `model.fit` in Keras) for **supervised** tasks.  
`Runner` is used to actually run the train loop calling `Callbacks` at appropriate times.  
Mixed precision (powered by apex) is supported implicitly. Users are expected to initialize their models before creating runner using `apex.amp.initialize`.

## Minimal example
This code will run training for 5 epochs.
```python
from pytorch_tools.fit_wrapper import Runner
import apex
model = ... (any model)
optimizer = ... (any optimizer)
criterions = ... (any loss function)
metrics = ... (any metrics or None)
train_loader = ... (dataloader with defined __len__)
model, optimizer = apex.amp.initialize(model, optimizer)
runner = Runner(model, optimizer, criterion, metrics)
runner.fit(train_loader, epochs=5)
```

## Real example 
This code will run training for 5 epochs with validation every epoch. 
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
)
runner.fit(train_loader, epochs=5, val_loader=val_loader)
```