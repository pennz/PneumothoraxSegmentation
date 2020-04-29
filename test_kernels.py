import importlib
import os
import time
import shutil
import sys
import unittest

import numpy as np
import pytest
import torch
from IPython.core.debugger import set_trace
from torch.autograd import Variable

import kernel

# import modelTester
import pytorchKernel
import runner

# from kernel import Kernel
import utils
from kernel import KernelRunningState
import coordinator
import pytest

@pytest.mark.skip()
class TestPSKernel():
    # this function will run before every test. We re-initialize group in this
    # function. So for every test, new group is used.
    @classmethod
    def setup_class(cls):
        utils.logger.debug("Have a good day")
        importlib.reload(utils)
        importlib.reload(kernel)
        importlib.reload(pytorchKernel)
        # importlib.reload(modelTester)

    def test_class(self):
        ps_kernel = kernel.PS()
        assert len(ps_kernel.model_metrics) == 0

    def test_dump_load_continue(self):
        ps_kernel = kernel.PS()
        ps_kernel.run(end_stage=KernelRunningState.TRAINING_DONE)
        assert ps_kernel._stage == KernelRunningState.TRAINING_DONE

        kernel_load_back = utils.KaggleKernel._load_state()
        assert kernel_load_back._stage == KernelRunningState.TRAINING_DONE
        kernel_load_back.run()
        assert kernel_load_back._stage == KernelRunningState.SAVE_SUBMISSION_DONE

    def test_prepare_data(self):
        ps_kernel = kernel.PS()
        ps_kernel.run(
            end_stage=KernelRunningState.PREPARE_DATA_DONE, dump_flag=True
        )  # will also analyze data
        assert ps_kernel.train_X is not None
        assert len(ps_kernel.train_X) == len(ps_kernel.train_Y)
        # self.assertIsNotNone(ps_kernel.test_X)  // don't care this now
        assert ps_kernel.dev_X is not None
        assert len(ps_kernel.dev_X) == len(ps_kernel.dev_Y)

    def test_train(self):
        kernel_load_back = utils.KaggleKernel._load_state(
            KernelRunningState.PREPARE_DATA_DONE
        )
        kernel_load_back.run(end_stage=KernelRunningState.TRAINING_DONE)
        assert kernel_load_back.model is not None

    def test_read_tf(self):
        k = kernel.PS()
        k._recover_from_tf()
        # k.run(start_stage=KernelRunningState.PREPARE_DATA_DONE,
        # end_stage=KernelRunningState.TRAINING_DONE)
        assert k.ds is not None

    # def test_convert_tf(self):
    #    kernel_withdata
    #    = utils.KaggleKernel._load_state(KernelRunningState.PREPARE_DATA_DONE)
    #    k = kernel.PS()
    #    k._clone_data(kernel_withdata)
    #    k.after_prepare_data_hook()

    #    self.assertTrue(os.path.isfile('train.tfrec'))
    def test_data_aug(self):
        self._prepare_data()

        k = pytorchKernel.PS_torch()
        k.load_state_data_only(KernelRunningState.PREPARE_DATA_DONE)
        # k.run(end_stage=KernelRunningState.PREPARE_DATA_DONE,
        # dump_flag=True)  # will also analyze data
        k.data_loader.dataset._test_(1019)
        k.data_loader.dataset._test_(2019)
        # k.run()  # dump not working for torch

        assert k is not None

    def test_pytorch_model(self):
        k = pytorchKernel.PS_torch()
        # k.run(end_stage=KernelRunningState.PREPARE_DATA_DONE,
        # dump_flag=True)  # will also analyze data
        k.submit_run = True
        k.run(dump_flag=True)  # dump not working for torch
        assert k is not None

    def test_pytorch_model_dev(self):
        k = pytorchKernel.PS_torch()
        # k.run(end_stage=KernelRunningState.PREPARE_DATA_DONE,
        # dump_flag=True)  # will also analyze data
        # k._debug_less_data = True
        k.run(
            end_stage=KernelRunningState.TRAINING_DONE, dump_flag=True
        )  # dump not working for torch
        assert k is not None

    def test_torch_show_model_detail(self):
        k = pytorchKernel.PS_torch()
        k._build_show_model_detail()  # not work

    def _prepare_data(self):
        _stage = KernelRunningState.PREPARE_DATA_DONE
        data_stage_file_name = f"run_state_{_stage}.pkl"
        if not os.path.isfile(data_stage_file_name):
            self.test_pytorch_starter_dump()

    def test_FL_in_model_early_stop(self):
        self._prepare_data()
        kernel_load_back = pytorchKernel.PS_torch()

        kernel_load_back.load_state_data_only(
            KernelRunningState.PREPARE_DATA_DONE)
        kernel_load_back.build_and_set_model()
        kernel_load_back.train_model()

    def test_cv_train_dev(self):
        k = pytorchKernel.PS_torch()

        k._debug_less_data = True
        k.run(end_stage=KernelRunningState.PREPARE_DATA_DONE)

        k.build_and_set_model()
        k.num_epochs = 10
        k.train_model()

    def test_cv_data_prepare(self):
        k = pytorchKernel.PS_torch()
        k.run(end_stage=KernelRunningState.PREPARE_DATA_DONE, dump_flag=True)

        k2 = pytorchKernel.PS_torch()
        k2.run(end_stage=KernelRunningState.PREPARE_DATA_DONE)

        l = len(k.data_loader)
        l_dev = len(k.data_loader_dev)
        ratio = l / l_dev

        assert ratio > 3.5  # around 0.8/0.2
        assert ratio < 4.5

    def test_focal_loss_func(self):
        inputs = torch.tensor(
            [
                [0.0, 0.2, 0.4, 0.0, 0.2, 0.4, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                [0.0, 0.2, 0.4, 0.0, 0.2, 0.4, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ]
        )
        targets = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        )

        FL = pytorchKernel.FocalLoss(gamma=2)
        FL_normal_CE = pytorchKernel.FocalLoss(gamma=0)
        alpha_for_pos = 0.75
        FL_alpha_balance = pytorchKernel.FocalLoss(
            gamma=0, alpha=alpha_for_pos
        )  # alpha for positive class weights

        print("----inputs----")
        print(inputs)
        print("---target-----")
        print(targets)
        losses = []
        grads = []

        for loss_func in [FL, FL_normal_CE, FL_alpha_balance, FL_normal_CE]:
            inputs_fl = Variable(inputs.clone(), requires_grad=True)
            targets_fl = Variable(targets.clone())
            fl_loss = loss_func(inputs_fl, targets_fl)
            print(fl_loss.data)
            fl_loss.backward()
            print(inputs_fl.grad.data)
            losses.append(fl_loss.data.numpy())
            grads.append(inputs_fl.grad.data.numpy())
        assert losses[-1] == losses[1]
        if alpha_for_pos >= 0.5:
            assert losses[2] < losses[1]
        else:
            assert losses[2] > losses[1]

        assert (grads[2][0, 1] / grads[1][0, 1]) == (1 - alpha_for_pos) / 0.5

    def test_pytorch_starter_dump(self):
        k = pytorchKernel.PS_torch()
        k.run(
            end_stage=KernelRunningState.PREPARE_DATA_DONE, dump_flag=True
        )  # will also analyze data
        # kernel_load_back = pytorchKernel.PS_torch()

        # kernel_load_back.load_state_data_only(KernelRunningState.PREPARE_DATA_DONE)
        # kernel_load_back.run(end_stage=KernelRunningState.TRAINING_DONE)

    def test_pytorch_starter_load(self):
        kernel_load_back = pytorchKernel.PS_torch()

        kernel_load_back.load_state_data_only(KernelRunningState.TRAINING_DONE)
        kernel_load_back.load_model_weight()

    def test_pytorch_starter_load_continue_train(self):
        kernel_load_back = pytorchKernel.PS_torch()

        kernel_load_back.load_state_data_only(KernelRunningState.TRAINING_DONE)
        kernel_load_back._debug_continue_training = True
        kernel_load_back.load_model_weight_continue_train()
        # kernel_load_back.run(end_stage=KernelRunningState.TRAINING_DONE)

    def test_pytorch_starter_load_then_submit(self):
        kernel_load_back = pytorchKernel.PS_torch()

        kernel_load_back.load_state_data_only(KernelRunningState.TRAINING_DONE)
        kernel_load_back.load_model_weight()
        kernel_load_back.run()
        # kernel_load_back.run(end_stage=KernelRunningState.TRAINING_DONE)

    def test_convert_tf_from_start(self):  # won't work
        ps_kernel = kernel.PS()
        ps_kernel.run(end_stage=KernelRunningState.PREPARE_DATA_DONE)
        assert os.path.isfile("train_dev.10.tfrec")

    def test_tf_model_zoo(self):
        t = modelTester.TF_model_zoo_tester()
        t.run_logic()

    def test_tf_model_zoo_model(self):
        t = modelTester.TF_model_zoo_tester()
        t.load_model()
        assert t.model is not None

    def test_tf_model_zoo_model(self):
        t = modelTester.TF_model_zoo_tester()
        t.set_model("mask_rcnn_resnet101_atrous_coco_2018_01_28")
        t.run_prepare()
        # t.check_graph()
        t.run_test()  # result is great!!!
        assert t.detection_graph is not None

    def test_analyze_RPN(self):
        assert False

    def test_analyze_predict_error(self):
        assert False

    def test_analyze_predict_score_threshold(self):
        ts = np.exp([0.5, 0.6, 0.7])
        for t in ts:
            check_predict_statistics()
        assert False

    def test_TTA(self):
        assert False

    def test_L_loss(self):
        assert False

    def test_dataset_mean_std(self):
        k = pytorchKernel.PS_torch()
        # k.run(end_stage=KernelRunningState.PREPARE_DATA_DONE,
        # dump_flag=True)  # will also analyze data
        k._debug_less_data = True
        k.run(
            end_stage=KernelRunningState.PREPARE_DATA_DONE, dump_flag=False
        )  # dump not working for torch
        k.pre_train()
        assert k.img_mean is not None


if "__main__" == __name__:
    unittest.main()
