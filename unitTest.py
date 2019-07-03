import unittest
import kernel
import numpy as np
import utils
import os
import importlib
#import modelTester
import pytorchKernel
import torch
from torch.autograd import Variable

from IPython.core.debugger import set_trace

class PSKenelTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(PSKenelTest, self).__init__(*args, **kwargs)

    def setUp(self):  # this function will run before every test. We re-initialize group in this function. So for every test, new group is used.
        utils.logger.debug('Have a good day')
        importlib.reload(utils)
        importlib.reload(kernel)
        importlib.reload(pytorchKernel)
        #importlib.reload(modelTester)


    def test_class(self):
        ps_kernel = kernel.PS()
        self.assertEqual(len(ps_kernel.model_metrics), 0)

    def test_dump_load_continue(self):
        ps_kernel = kernel.PS()
        ps_kernel.run(end_stage=utils.KernelRunningState.TRAINING_DONE)
        self.assertEqual(ps_kernel._stage, utils.KernelRunningState.TRAINING_DONE)

        kernel_load_back = utils.KaggleKernel._load_state()
        self.assertEqual(kernel_load_back._stage, utils.KernelRunningState.TRAINING_DONE)
        kernel_load_back.run()
        self.assertEqual(kernel_load_back._stage, utils.KernelRunningState.SAVE_SUBMISSION_DONE)

    def test_prepare_data(self):
        ps_kernel = kernel.PS()
        ps_kernel.run(end_stage=utils.KernelRunningState.PREPARE_DATA_DONE, dump_flag=True)  # will also analyze data
        self.assertIsNotNone(ps_kernel.train_X)
        self.assertEqual(len(ps_kernel.train_X), len(ps_kernel.train_Y))
        #self.assertIsNotNone(ps_kernel.test_X)  // don't care this now
        self.assertIsNotNone(ps_kernel.dev_X)
        self.assertEqual(len(ps_kernel.dev_X), len(ps_kernel.dev_Y))

    def test_train(self):
        kernel_load_back = utils.KaggleKernel._load_state(utils.KernelRunningState.PREPARE_DATA_DONE)
        kernel_load_back.run(end_stage=utils.KernelRunningState.TRAINING_DONE)
        self.assertIsNotNone(kernel_load_back.model)

    def test_read_tf(self):
        k = kernel.PS()
        k._recover_from_tf()
        #k.run(start_stage=utils.KernelRunningState.PREPARE_DATA_DONE, end_stage=utils.KernelRunningState.TRAINING_DONE)
        self.assertIsNotNone(k.ds)

    #def test_convert_tf(self):
    #    kernel_withdata = utils.KaggleKernel._load_state(utils.KernelRunningState.PREPARE_DATA_DONE)
    #    k = kernel.PS()
    #    k._clone_data(kernel_withdata)
    #    k.after_prepare_data_hook()

    #    self.assertTrue(os.path.isfile('train.tfrec'))
    def test_data_aug(self):
        self._prepare_data()

        k = pytorchKernel.PS_torch()
        k.load_state_data_only(utils.KernelRunningState.PREPARE_DATA_DONE)
        #k.run(end_stage=utils.KernelRunningState.PREPARE_DATA_DONE, dump_flag=True)  # will also analyze data
        k.data_loader.dataset._test_(1019)
        k.data_loader.dataset._test_(2019)
        #k.run()  # dump not working for torch

        self.assertIsNotNone(k)

    def test_pytorch_model(self):
        k = pytorchKernel.PS_torch()
        #k.run(end_stage=utils.KernelRunningState.PREPARE_DATA_DONE, dump_flag=True)  # will also analyze data
        k.submit_run = True
        k.run(dump_flag=True)  # dump not working for torch
        self.assertIsNotNone(k)

    def test_pytorch_model_dev(self):
        k = pytorchKernel.PS_torch()
        #k.run(end_stage=utils.KernelRunningState.PREPARE_DATA_DONE, dump_flag=True)  # will also analyze data
        k.run(end_stage=utils.KernelRunningState.TRAINING_DONE, dump_flag=True)  # dump not working for torch
        self.assertIsNotNone(k)

    def test_torch_show_model_detail(self):
        k = pytorchKernel.PS_torch()
        k._build_show_model_detail()  # not work

    def _prepare_data(self):
        _stage = utils.KernelRunningState.PREPARE_DATA_DONE
        data_stage_file_name = f'run_state_{_stage}.pkl'
        if not os.path.isfile(data_stage_file_name):
            self.test_pytorch_starter_dump()

    def test_FL_in_model_early_stop(self):
        self._prepare_data()
        kernel_load_back = pytorchKernel.PS_torch()

        kernel_load_back.load_state_data_only(utils.KernelRunningState.PREPARE_DATA_DONE)
        kernel_load_back.build_and_set_model()
        kernel_load_back.train_model()

    def test_cv_train_dev(self):
        k = pytorchKernel.PS_torch()

        k._debug_less_data = True
        k.run(end_stage=utils.KernelRunningState.PREPARE_DATA_DONE)

        k.build_and_set_model()
        k.num_epochs = 10
        k.train_model()

    def test_cv_data_prepare(self):
        k = pytorchKernel.PS_torch()
        k.run(end_stage=utils.KernelRunningState.PREPARE_DATA_DONE, dump_flag=True)

        k2 = pytorchKernel.PS_torch()
        k2.run(end_stage=utils.KernelRunningState.PREPARE_DATA_DONE)

        l = len(k.data_loader)
        l_dev = len(k.data_loader_dev)
        ratio = l / l_dev

        self.assertGreater(ratio, 3.5)  # around 0.8/0.2
        self.assertLess(ratio, 4.5)

    def test_focal_loss_func(self):
        inputs = torch.tensor([[0., 0.2, 0.4,0., 0.2, 0.4,0., 0.2, 0.4, 0.6, 0.8, 1.],[0., 0.2, 0.4,0., 0.2, 0.4,0., 0.2, 0.4, 0.6, 0.8, 1.]])
        targets = torch.tensor([[0., 0., 0.,0., 0., 0.,0., 0., 0., 1., 1., 1.],[0., 0., 0.,0., 0., 0.,0., 0., 0., 1., 1., 1.]])

        FL = pytorchKernel.FocalLoss(gamma=2)
        FL_normal_CE = pytorchKernel.FocalLoss(gamma=0)
        alpha_for_pos = 0.75
        FL_alpha_balance = pytorchKernel.FocalLoss(gamma=0, alpha=alpha_for_pos)  # alpha for positive class weights


        print('----inputs----')
        print(inputs)
        print('---target-----')
        print(targets)
        losses = []
        grads = []

        for loss_func in [FL, FL_normal_CE, FL_alpha_balance,FL_normal_CE]:
            inputs_fl = Variable(inputs.clone(), requires_grad=True)
            targets_fl = Variable(targets.clone())
            fl_loss = loss_func(inputs_fl, targets_fl)
            print(fl_loss.data)
            fl_loss.backward()
            print(inputs_fl.grad.data)
            losses.append(fl_loss.data.numpy())
            grads.append(inputs_fl.grad.data.numpy())
        self.assertEqual(losses[-1], losses[1])
        if alpha_for_pos >= 0.5:
            self.assertLess(losses[2], losses[1])
        else:
            self.assertGreater(losses[2], losses[1])

        self.assertEqual((grads[2][0,1]/grads[1][0,1]), (1-alpha_for_pos)/0.5)

    def test_pytorch_starter_dump(self):
        k = pytorchKernel.PS_torch()
        k.run(end_stage=utils.KernelRunningState.PREPARE_DATA_DONE, dump_flag=True)  # will also analyze data
        #kernel_load_back = pytorchKernel.PS_torch()

        #kernel_load_back.load_state_data_only(utils.KernelRunningState.PREPARE_DATA_DONE)
        #kernel_load_back.run(end_stage=utils.KernelRunningState.TRAINING_DONE)

    def test_pytorch_starter_load(self):
        kernel_load_back = pytorchKernel.PS_torch()

        kernel_load_back.load_state_data_only(utils.KernelRunningState.TRAINING_DONE)
        kernel_load_back.load_model_weight()

    def test_pytorch_starter_load_continue_train(self):
        kernel_load_back = pytorchKernel.PS_torch()

        kernel_load_back.load_state_data_only(utils.KernelRunningState.TRAINING_DONE)
        kernel_load_back._debug_continue_training = True
        kernel_load_back.load_model_weight_continue_train()
        #kernel_load_back.run(end_stage=utils.KernelRunningState.TRAINING_DONE)

    def test_pytorch_starter_load_then_submit(self):
        kernel_load_back = pytorchKernel.PS_torch()

        kernel_load_back.load_state_data_only(utils.KernelRunningState.TRAINING_DONE)
        kernel_load_back.load_model_weight()
        kernel_load_back.run()
        #kernel_load_back.run(end_stage=utils.KernelRunningState.TRAINING_DONE)

    def test_convert_tf_from_start(self):  # won't work
        ps_kernel = kernel.PS()
        ps_kernel.run(end_stage=utils.KernelRunningState.PREPARE_DATA_DONE)
        self.assertTrue(os.path.isfile('train_dev.10.tfrec'))

    def test_tf_model_zoo(self):
        t = modelTester.TF_model_zoo_tester()
        t.run_logic()

    def test_tf_model_zoo_model(self):
        t = modelTester.TF_model_zoo_tester()
        t.load_model()
        self.assertIsNotNone(t.model)

    def test_tf_model_zoo_model(self):
        t = modelTester.TF_model_zoo_tester()
        t.set_model('mask_rcnn_resnet101_atrous_coco_2018_01_28')
        t.run_prepare()
        #t.check_graph()
        t.run_test()  # result is great!!!
        self.assertIsNotNone(t.detection_graph)


if '__main__' == __name__:
    unittest.main()