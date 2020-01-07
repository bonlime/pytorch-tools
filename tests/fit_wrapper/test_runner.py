import os
import pytest
import torch
import torch.nn as nn
import apex
from pytorch_tools.metrics import Accuracy
from pytorch_tools.fit_wrapper import Runner
from pytorch_tools.losses import CrossEntropyLoss
import pytorch_tools.fit_wrapper.callbacks as pt_clb

HIDDEN_DIM = 16
NUM_CLASSES = 10
IMG_SHAPE = 16
LOADER_LEN = 20
BS = 2


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, HIDDEN_DIM, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(HIDDEN_DIM)
        self.conv2 = nn.Conv2d(HIDDEN_DIM, HIDDEN_DIM, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Loader:
    def __init__(self):
        self.i = 1
        pass

    def __iter__(self):
        return self

    def __len__(self):
        return LOADER_LEN

    def __next__(self):
        img = torch.randn(BS, 3, IMG_SHAPE, IMG_SHAPE)
        target = torch.randint(NUM_CLASSES, (BS,))
        return img.cuda(), target.cuda()


TestLoader = Loader()
TestModel = Model().cuda()
TestOptimizer = torch.optim.SGD(TestModel.parameters(), lr=1e-3)
TestCriterion = CrossEntropyLoss().cuda()
TestMetric = Accuracy()

TestModel, TestOptimizer = apex.amp.initialize(TestModel, TestOptimizer, verbosity=0)


def test_default():
    runner = Runner(
        model=TestModel, optimizer=TestOptimizer, criterion=TestCriterion, metrics=TestMetric,
    )
    runner.fit(TestLoader, epochs=2)


def test_val_loader():
    runner = Runner(
        model=TestModel,
        optimizer=TestOptimizer,
        criterion=TestCriterion,
        metrics=TestMetric,
        callbacks=None,
    )
    runner.fit(TestLoader, epochs=2, steps_per_epoch=10, val_loader=TestLoader, val_steps=20)


# We only test that callbacks don't crash NOT that they do what they should do
TMP_PATH = "/tmp/pt_tools2/"
os.makedirs(TMP_PATH, exist_ok=True)

def test_Timer_callback():
    runner = Runner(
        model=TestModel,
        optimizer=TestOptimizer,
        criterion=TestCriterion,
        metrics=TestMetric,
        callbacks=pt_clb.Timer(),
    )
    runner.fit(TestLoader, epochs=2)


def test_ReduceLROnPlateau_callback():
    runner = Runner(
        model=TestModel,
        optimizer=TestOptimizer,
        criterion=TestCriterion,
        metrics=TestMetric,
        callbacks=pt_clb.ReduceLROnPlateau(),
    )
    runner.fit(TestLoader, epochs=2)


def test_CheckpointSaver_callback():
    runner = Runner(
        model=TestModel,
        optimizer=TestOptimizer,
        criterion=TestCriterion,
        metrics=TestMetric,
        callbacks=pt_clb.CheckpointSaver(TMP_PATH, save_name="model.chpn"),
    )
    runner.fit(TestLoader, epochs=2)


def test_FileLogger_callback():
    runner = Runner(
        model=TestModel,
        optimizer=TestOptimizer,
        criterion=TestCriterion,
        metrics=TestMetric,
        callbacks=pt_clb.FileLogger(TMP_PATH),
    )
    runner.fit(TestLoader, epochs=2)


def test_TensorBoard():
    runner = Runner(
        model=TestModel,
        optimizer=TestOptimizer,
        criterion=TestCriterion,
        metrics=TestMetric,
        callbacks=pt_clb.TensorBoard(log_dir=TMP_PATH),
    )
    runner.fit(TestLoader, epochs=2)

def test_Cutmix():
    runner = Runner(
        model=TestModel,
        optimizer=TestOptimizer,
        criterion=TestCriterion,
        metrics=TestMetric,
        callbacks=pt_clb.Cutmix(1.0, NUM_CLASSES)
    )
    runner.fit(TestLoader, epochs=2)

def test_Mixup():
    runner = Runner(
        model=TestModel,
        optimizer=TestOptimizer,
        criterion=TestCriterion,
        metrics=TestMetric,
        callbacks=pt_clb.Mixup(0.2, NUM_CLASSES)
    )
    runner.fit(TestLoader, epochs=2)