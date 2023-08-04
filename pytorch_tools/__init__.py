"""pweb package."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pytorch_tools")
except PackageNotFoundError:
    __version__ = "unknown version"

from . import fit_wrapper
from . import losses
from . import metrics
from . import models
from . import modules
from . import optim
from . import segmentation_models
from . import tta_wrapper
from . import utils
from . import detection_models
