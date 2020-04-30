import gc
import logging
from enum import Enum

import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.core.debugger import set_trace
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.python.ops import math_ops
from tqdm import tqdm

import utils

# Plot inline
# %matplotlib inline


class KernelRunningState(Enum):
    INIT_DONE = 1
    PREPARE_DATA_DONE = 2
    TRAINING_DONE = 3
    EVL_DEV_DONE = 4
    SAVE_SUBMISSION_DONE = 5


class KernelGroup:
    "Kernel Group to try different combination of kernels hyperparameter"

    def __init__(self, *kernels):
        self.kernels = kernels


class KaggleKernel:
    def __init__(self, logger=None):
        self.model = None
        self.model_metrics = []
        self.model_loss = None

        self.train_X = None
        self.train_Y = None

        self.dev_X = None
        self.dev_Y = None

        self.test_X = None

        self.result_analyzer = None  # for analyze the result

        self._stage = KernelRunningState.INIT_DONE
        self.logger = logger

    def _add_logger_handler(self, handler):
        self.logger.addHandler(handler)

    def set_logger(self, name, level=logging.DEBUG, handler=None):
        FORMAT = "[%(levelname)s]%(asctime)s:%(name)s:%(message)s"
        logging.basicConfig(format=FORMAT)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if handler is not None:
            logger.addHandler(rabbit)
        self.logger = logger

    def set_random_seed(self):
        pass

    def set_data_size(self):
        "might be useful when test different input datasize"

    def save_model(self):
        pass

    def load_model_weight(self):
        pass

    def build_and_set_model(self):
        pass

    def train_model(self):
        pass

    def set_model(self):
        pass

    def set_loss(self):
        pass

    def set_metrics(self):
        """
        set_metrics for model training

        :return: None
        """

    def set_result_analyzer(self):
        pass

    def pre_prepare_data_hook(self):
        pass

    def after_prepare_data_hook(self):
        pass

    def prepare_train_dev_data(self):
        pass

    def prepare_test_data(self):
        pass

    def predict_on_test(self):
        pass

    def dump_state(self, exec_flag=False):
        logger.debug(f"state {self._stage}")
        if exec_flag:
            logger.debug(f"dumping state {self._stage}")
            # dump_obj(self, 'run_state.pkl', force=True)  # too large
            dump_obj(self, f"run_state_{self._stage}.pkl", force=True)

    def run(
        self,
        start_stage=None,
        end_stage=KernelRunningState.SAVE_SUBMISSION_DONE,
        dump_flag=False,
    ):
        """

        :param start_stage: if set, will overwrite the stage
        :param end_stage:
        :param dump_flag:
        :return:
        """
        self.continue_run(
            start_stage=start_stage, end_stage=end_stage, dump_flag=dump_flag
        )

    def continue_run(
        self,
        start_stage=None,
        end_stage=KernelRunningState.SAVE_SUBMISSION_DONE,
        dump_flag=False,
    ):
        if start_stage is not None:
            assert start_stage.value < end_stage.value
            self._stage = start_stage

        if self._stage.value < KernelRunningState.PREPARE_DATA_DONE.value:
            self.pre_prepare_data_hook()
            self.prepare_train_dev_data()
            self.after_prepare_data_hook()

            self._stage = KernelRunningState.PREPARE_DATA_DONE
            self.dump_state(exec_flag=dump_flag)
            if self._stage.value >= end_stage.value:
                return

        if self._stage.value < KernelRunningState.TRAINING_DONE.value:
            self.pre_train()
            self.build_and_set_model()
            self.train_model()
            self.after_train()

            self.save_model()

            self._stage = KernelRunningState.TRAINING_DONE
            self.dump_state(exec_flag=dump_flag)
            if self._stage.value >= end_stage.value:
                return

        if self._stage.value < KernelRunningState.EVL_DEV_DONE.value:
            self.set_result_analyzer()

            self._stage = KernelRunningState.EVL_DEV_DONE
            self.dump_state(exec_flag=dump_flag)
            if self._stage.value >= end_stage.value:
                return

        if self._stage.value < KernelRunningState.SAVE_SUBMISSION_DONE.value:
            self.pre_test()
            self.prepare_test_data()
            self.predict_on_test()
            self.after_test()

            self._stage = KernelRunningState.SAVE_SUBMISSION_DONE
            self.dump_state(exec_flag=dump_flag)
            if self._stage.value >= end_stage.value:
                return

    @classmethod
    def _load_state(cls, stage=None, file_name="run_state.pkl"):
        """

        :param file_name:
        :return: the kernel object, need to continue
        """
        if stage is not None:
            file_name = f"run_state_{stage}.pkl"
        logger.debug(f"restore from {file_name}")
        return get_obj_or_dump(filename=file_name)

    def load_state_data_only(self, file_name="run_state.pkl"):
        pass

    @classmethod
    def load_state_continue_run(cls, file_name="run_state.pkl"):
        """

        :param file_name:
        :return: the kernel object, need to continue
        """
        self = cls._load_state(file_name=file_name)
        self.continue_run()

    def pre_train(self):
        pass

    def after_train(self):
        pass

    def pre_test(self):
        pass

    def after_test(self):
        pass


def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return " ".join(rle)


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position: current_position + lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)
