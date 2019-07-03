import utils
from utils import KaggleKernel
from glob import glob
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from tqdm import tqdm
import pydicom
import math

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops

import gc

from IPython.core.debugger import set_trace
from matplotlib import pyplot as plt

# Plot inline
#%matplotlib inline


class PS(KaggleKernel):
    def __init__(self):
        super(PS, self).__init__()
        self._PS_init_()

    def _PS_init_(self):
        self.developing = True
        # self.train_dev_mask = None  # for train dev split
        self.DATA_PATH_BASE = '../input/siim-acr-pneumothorax-segmentation-data/pneumothorax'
        self._im_chan = 1

        self.BS = 16
        try:
            self.ds  # if has this attribute, no need to overwrite
        except AttributeError:
            self.ds = None  # for train
            self.ds_len = 0
            self.dev_ds = None
            self.dev_ds_len = 0
            self.tf_data_handler = None

    def analyze_data(self):
        pass

    @staticmethod
    def _PS_data_preprocess_np(fns, df, TARGET_COLUMN, im_height, im_width, im_chan):
        X_train = np.zeros((len(fns), im_height, im_width, im_chan), dtype=np.uint8)
        Y_train = np.zeros((len(fns), im_height, im_width, 1), dtype=np.uint8)
        print('Getting train images and masks ... ')
        #sys.stdout.flush()
        for n, _id in tqdm(enumerate(fns), total=len(fns)):
            dataset = pydicom.read_file(_id)
            _id_keystr = _id.split('/')[-1][:-4]
            X_train[n] = np.expand_dims(dataset.pixel_array, axis=2)
            try:
                mask_data = df.loc[_id_keystr, TARGET_COLUMN]

                if '-1' in mask_data:
                    Y_train[n] = np.zeros((1024, 1024, 1))
                else:
                    if type(mask_data) == str:
                        Y_train[n] = np.expand_dims(
                            rle2mask(df.loc[_id_keystr, TARGET_COLUMN], 1024, 1024).T, axis=2)
                    else:
                        Y_train[n] = np.zeros((1024, 1024, 1))
                        for x in mask_data:
                            Y_train[n] = Y_train[n] + np.expand_dims(rle2mask(x, 1024, 1024).T, axis=2)
            except KeyError:
                print(f"Key {_id.split('/')[-1][:-4]} without mask, assuming healthy patient.")
                Y_train[n] = np.zeros((1024, 1024, 1))  # Assume missing masks are empty masks.

        print('Done data preprocessing as numpy array!')

        return X_train, Y_train

    @staticmethod
    def _PS_data_preprocess(fns, df, tf=False):
        """
        for tf=True, need to use TF2.0

        :param fns:
        :param df:
        :param tf:
        :return:
        """
        TARGET_COLUMN = ' EncodedPixels'
        im_height = 1024
        im_width = 1024
        im_chan = 1
        # Get train images and masks
        if not tf:
            return PS._PS_data_preprocess_np(fns, df, TARGET_COLUMN, im_height, im_width, im_chan)
        else:
            return PS._PS_data_preprocess_tf(fns, df, TARGET_COLUMN, im_height, im_width, im_chan)

    @staticmethod
    def _PS_data_preprocess_tf(fns, df, TARGET_COLUMN, im_height, im_width, im_chan):
        tf_data_handler = utils.PS_TF_DataHandler()
        return tf_data_handler.to_tf_from_disk(fns, df, TARGET_COLUMN, im_height, im_width, im_chan)



    @staticmethod
    def img_build_patches(imgs):
        im_height = 128
        im_width = 128
        imgs = imgs.reshape((-1, im_height, im_width, 1))
        return imgs

    def prepare_train_dev_data(self):
        self._prepare_train_data_as_np()

    def _prepare_train_data_as_tf(self):  # just prepare train/dev here together
        # getting path of all the train and test images
        train_data_wildcard = self.DATA_PATH_BASE+'/dicom-images-train/*/*/*.dcm'

        if self.developing:
            train_fns = sorted(glob(train_data_wildcard))
        else:
            train_fns = sorted(glob(train_data_wildcard))

        utils.logger.debug(f'train & dev counts: {len(train_fns)}')
        df_full = pd.read_csv(self.DATA_PATH_BASE+'/train-rle.csv',
                              index_col='ImageId')
        self.ds = PS._PS_data_preprocess(train_fns, df_full, tf=True)

    def _prepare_train_data_as_np(self):  # just prepare train/dev here together
        # getting path of all the train and test images
        train_data_wildcard = self.DATA_PATH_BASE+'/dicom-images-train/*/*/*.dcm'

        train_fns = sorted(glob(train_data_wildcard))

        utils.logger.debug(f'train & dev counts: {len(train_fns)}')
        df_full = pd.read_csv(self.DATA_PATH_BASE+'/train-rle.csv',
                              index_col='ImageId')

        train_fns_splits = []
        idx = 0
        LEN=1024
        for idx in range(len(train_fns)//LEN+1):
            train_fns_splits.append(train_fns[idx*LEN:idx*LEN+LEN])

        for i, fns in enumerate(train_fns_splits):
            images, mask_e = PS._PS_data_preprocess(fns, df_full)

            splits = list(KFold(n_splits=5, random_state=2019, shuffle=True).split(images))  # just use sklearn split to get id and it is fine. For text thing,
            tr_ind, val_ind = splits[0]  # just do 1 fold, later we can add them all back

            self.train_X = images[tr_ind]
            self.train_Y = mask_e[tr_ind]

            self.dev_X = images[val_ind]
            self.dev_Y = mask_e[val_ind]

            utils.logger.debug(f'size: train_X {self.train_X.shape}, train_Y {self.train_Y.shape}')

            utils.logger.debug(sorted(glob('*')))
            self.save_data_tf(file_name=f'train_dev.{i}.tfrec')
            del self.train_X
            del self.train_Y
            del self.dev_X
            del self.dev_Y
            del images
            del mask_e
            gc.collect()

    def _clone_data(self, src):
        self.model = src.model
        self.model_metrics = src.model_metrics
        self.model_loss = src.model_loss

        self.train_X = src.train_X
        self.train_Y = src.train_Y

        self.dev_X = src.dev_X
        self.dev_Y = src.dev_Y

        self.test_X = src.test_X

        self.result_analyzer = src.result_analyzer   # for analyze the result

        self._stage = src._stage

    def save_data_tf(self, file_name='train_dev.tfrec'):
        assert self.train_X is not None

        all_X = np.concatenate((self.train_X, self.dev_X), axis=0)
        all_Y = np.concatenate((self.train_Y, self.dev_Y), axis=0)

        ds = utils.PS_TF_DataHandler.get_train_dataset(all_X, all_Y)

        #def train_input_fn_bt(features, labels, batch_size, cv, split_id=None, n_splits=None, ds=None):
        assert len(all_X) == len(all_Y)
        try:
            self.BS
        except AttributeError:
            self._PS_init_()
        split = 5
        #self.ds = utils.PS_TF_DataHandler.train_input_fn_bt(None, None, BS, cv=True, split_id=0, n_splits=split, ds=ds, ds_len=math.floor(len(all_X)*(1-1/split)))
        self.ds = ds
        self.ds_len = math.floor(len(all_X)*(1-1/split))
        #self.dev_ds = utils.PS_TF_DataHandler.train_input_fn_bt(None, None, BS, cv=True, cv_train=False, split_id=0, n_splits=split, ds=ds, ds_len=math.floor(len(all_X) * (1 / split)))
        self.dev_ds_len = math.floor(len(all_X) * (1 / split))

        utils.logger.debug(f'dev set counts: {self.dev_ds_len} / {len(all_X)}')

        utils.PS_TF_DataHandler.to_tfrecord(self.ds, file_name=file_name)

    def after_prepare_data_hook(self):
        self.analyze_data()

    def train_model(self):
        #self.model.fit(self.train_X, self.train_Y, validation_data=(self.dev_X, self.dev_Y), batch_size=256, epochs=5)  # can run now...
        v_s = math.floor(self.dev_ds_len / self.BS)

        def split_train_dev(ds, n_splits, split_id):
            ds_shards = [ds.shard(n_splits, i) for i in range(n_splits)]

            shards_cross = [ds_shards[val_id] for val_id in range(n_splits) if val_id != split_id]

            ds_train = shards_cross[0]
            for t in shards_cross[1:]:
                ds_train = ds.concatenate(t)
            return ds, ds.shard(n_splits, split_id)

        def mask_to_binary(a, b, threshold=0.5):
            #b = tf.transpose(b, [1,0,2])
            threshold = math_ops.cast(threshold, b.dtype)
            b = math_ops.cast(b > threshold, b.dtype)
            return a, b

        def ds_prepare():
            ds = self.ds
            ds = ds.map(mask_to_binary)
            ds, ds_dev = split_train_dev(ds, 5, 0)
            ds = ds.shuffle(buffer_size=1024).repeat().batch(self.BS).prefetch(10)
            ds_dev = ds_dev.batch(self.BS).prefetch(10)
            self.ds = ds
            self.dev_ds = ds_dev

        ds_prepare()
        self.model.fit(self.ds, steps_per_epoch=1024, epochs=1,
                       validation_data=self.dev_ds, validation_steps=v_s if v_s > 0 else 1, verbose=1)

    def prepare_test_data(self):
        test_data_wildcard = self.DATA_PATH_BASE+'/dicom-images-test/*/*/*.dcm'
        if self.developing:
            test_fns = sorted(glob(test_data_wildcard))
        else:
            test_fns = sorted(glob(test_data_wildcard))
        utils.logger.debug(f'test counts: {len(test_fns)}')

    def build_and_set_model(self):
        def build_model(input_layer, start_neurons):
            # ref: https://www.kaggle.com/phoenigs/u-net-dropout-augmentation-stratification
            # 128 -> 64
            conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
            conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
            pool1 = MaxPooling2D((2, 2))(conv1)
            pool1 = Dropout(0.25)(pool1)

            # 64 -> 32
            conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
            conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
            pool2 = MaxPooling2D((2, 2))(conv2)
            pool2 = Dropout(0.5)(pool2)

            # 32 -> 16
            conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
            conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
            pool3 = MaxPooling2D((2, 2))(conv3)
            pool3 = Dropout(0.5)(pool3)

            # 16 -> 8
            conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
            conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
            pool4 = MaxPooling2D((2, 2))(conv4)
            pool4 = Dropout(0.5)(pool4)

            # Middle
            convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
            convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

            # 8 -> 16
            deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
            uconv4 = concatenate([deconv4, conv4])
            uconv4 = Dropout(0.5)(uconv4)
            uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
            uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

            # 16 -> 32
            deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
            uconv3 = concatenate([deconv3, conv3])
            uconv3 = Dropout(0.5)(uconv3)
            uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
            uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

            # 32 -> 64
            deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
            uconv2 = concatenate([deconv2, conv2])
            uconv2 = Dropout(0.5)(uconv2)
            uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
            uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

            # 64 -> 128
            deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
            uconv1 = concatenate([deconv1, conv1])
            uconv1 = Dropout(0.5)(uconv1)
            uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
            uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

            # uconv1 = Dropout(0.5)(uconv1)
            output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

            return output_layer

        img_size = 256
        input_layer = Input((img_size, img_size, 1))
        output_layer = build_model(input_layer, 16)
        model = Model(input_layer, output_layer)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", utils.dice_coef])


        self.model = model


    def set_result_analyzer(self):
        self.result_analyzer = PS_result_analyzer()

    @staticmethod
    def _check_image_data(ds):
        cnt = 0
        for image, mask in ds:
            print(cnt)
            cnt += 1
            m = mask.numpy()
            if (m.max() <= 0):
                continue
            m = np.reshape(m, (1024, 1024))
            img = image.numpy()
            img = np.reshape(img, (1024, 1024))
            #plt.imshow(img)
            plt.imshow(m)
            set_trace()
            break

    def _recover_from_tf(self):
        self.ds = utils.PS_TF_DataHandler.from_tfrecord()
        PS._check_image_data(self.ds)

class PS_result_analyzer():  # todo maybe add other good analyze functions
    def dev_set_performance(self, y_true, y_pred):
        return utils.dice_coef(y_true, y_pred)

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
            currentPixel+=1

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)
