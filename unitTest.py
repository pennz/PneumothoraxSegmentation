import unittest
import kernel
import numpy as np
import utils
import os
import importlib
import modelTester

from IPython.core.debugger import set_trace

class PSKenelTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(PSKenelTest, self).__init__(*args, **kwargs)

    def setUp(self):  # this function will run before every test. We re-initialize group in this function. So for every test, new group is used.
        utils.logger.debug('Have a good day')
        importlib.reload(utils)
        importlib.reload(kernel)
        importlib.reload(modelTester)


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
        set_trace()


if '__main__' == __name__:
    unittest.main()