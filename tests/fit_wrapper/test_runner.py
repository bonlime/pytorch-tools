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
        self.conv1 = nn.Conv2d(3, HIDDEN_DIM, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(HIDDEN_DIM)
        self.conv2 = nn.Conv2d(HIDDEN_DIM, HIDDEN_DIM, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout2d(0.1)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        self.last_conv = nn.Conv2d(HIDDEN_DIM, NUM_CLASSES, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SegmModel(Model):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.last_conv(x)
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


class SegmLoader(Loader):
    def __next__(self):
        img = torch.randn(BS, 3, IMG_SHAPE, IMG_SHAPE)
        target = torch.randint(2, (BS, NUM_CLASSES, IMG_SHAPE, IMG_SHAPE))
        return img.cuda(), target.cuda()


TEST_LOADER = Loader()
TEST_SEGM_LOADER = SegmLoader()
TEST_MODEL = Model().cuda()
TEST_SEGM_MODEL = SegmModel().cuda()
TEST_OPTIMIZER = torch.optim.SGD(TEST_MODEL.parameters(), lr=1e-3)
TEST_SEGM_OPTIMZER = torch.optim.SGD(TEST_SEGM_MODEL.parameters(), lr=1e-3)
TEST_CRITERION = CrossEntropyLoss().cuda()
TEST_METRIC = Accuracy()

TEST_MODEL, TEST_OPTIMIZER = apex.amp.initialize(TEST_MODEL, TEST_OPTIMIZER, verbosity=0)
TEST_SEGM_MODEL, TEST_SEGM_OPTIMZER = apex.amp.initialize(TEST_SEGM_MODEL, TEST_SEGM_OPTIMZER)


def test_default():
    runner = Runner(
        model=TEST_MODEL,
        optimizer=TEST_OPTIMIZER,
        criterion=TEST_CRITERION,
        metrics=TEST_METRIC,
        callbacks=None,
    )
    runner.fit(TEST_LOADER, epochs=2)


def test_val_loader():
    runner = Runner(model=TEST_MODEL, optimizer=TEST_OPTIMIZER, criterion=TEST_CRITERION, metrics=TEST_METRIC)
    runner.fit(TEST_LOADER, epochs=2, steps_per_epoch=100, val_loader=TEST_LOADER, val_steps=200)



def test_grad_clip_loader():
    runner = Runner(
        model=TEST_MODEL,
        optimizer=TEST_OPTIMIZER,
        criterion=TEST_CRITERION,
        metrics=TEST_METRIC,
        gradient_clip_val=1.0
    )
    runner.fit(TEST_LOADER, epochs=2)


# We only test that callbacks don't crash NOT that they do what they should do
TMP_PATH = "/tmp/pt_tools2/"
os.makedirs(TMP_PATH, exist_ok=True)


@pytest.mark.parametrize(
    "callback",
    [
        pt_clb.Timer(),
        pt_clb.ReduceLROnPlateau(),
        pt_clb.CheckpointSaver(TMP_PATH, save_name="model.chpn"),
        pt_clb.CheckpointSaver(
            TMP_PATH, save_name="model.chpn", monitor=TEST_METRIC.name, mode="max"
        ),
        pt_clb.TensorBoard(log_dir=TMP_PATH),
        pt_clb.TensorBoardWithCM(log_dir=TMP_PATH),
        pt_clb.ConsoleLogger(),
        pt_clb.FileLogger(TMP_PATH),
        pt_clb.Mixup(0.2, NUM_CLASSES),
        pt_clb.Cutmix(1.0, NUM_CLASSES),
        pt_clb.ScheduledDropout(),
    ],
)
def test_callback(callback):
    runner = Runner(
        model=TEST_MODEL,
        optimizer=TEST_OPTIMIZER,
        criterion=TEST_CRITERION,
        metrics=TEST_METRIC,
        callbacks=callback,
    )
    runner.fit(TEST_LOADER, epochs=2)


@pytest.mark.parametrize(
    "callback", [pt_clb.SegmCutmix(1.0),],
)
def test_segm_callback(callback):
    runner = Runner(
        model=TEST_SEGM_MODEL,
        optimizer=TEST_SEGM_OPTIMZER,
        criterion=TEST_CRITERION,
        callbacks=callback,
    )
    runner.fit(TEST_SEGM_LOADER, epochs=2)