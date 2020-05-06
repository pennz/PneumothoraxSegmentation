#!/usr/bin/env python
# coding: utf-8

# ### Overview
# The primary goal of this competition is identification and segmentation of chest radiographic images with pneumothorax. In this kernel a U-net based approach is used, which provides end-to-end framework for image segmentation. In prior image segmentation competitions ([Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection/discussion) and [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge)), U-net based model architecture has demonstrated supperior performence, and top solutions are based on it. The current competition is similar to TGS Salt Identification Challenge in terms of identifying the correct mask based on visual inspection of images. Therefore, I have tried a technique that was extremely effective in Salt competition - [Hypercolumns](https://towardsdatascience.com/review-hypercolumn-instance-segmentation-367180495979).
#
# As a starting point [this public kernel](https://www.kaggle.com/mnpinto/pneumothorax-fastai-u-net) is used, and the following things are added (see text below for more details):
# * Hypercolumns * Gradient accumulation
# * TTA based on horizontal flip
# * Noise removal (if the predicted mask contains too few pixels, it is assumed to be empty)
# * Image equilibration
# * we can add mixup

import gc
import sys

import fastai
import torchvision
from fastai.callbacks import SaveModelCallback
from fastai.callbacks.hooks import (
    Hook,
    _hook_inner,
    dummy_eval,
    hook_outputs,
    model_sizes,
)
from fastai.vision import *
from fastai.vision.learner import cnn_config, create_head, num_features_model
from fastai.vision.models.unet import UnetBlock, _get_sfs_idxs
from IPython import get_ipython
from IPython.core.debugger import set_trace
from PIL import Image
from sklearn.model_selection import KFold

# In[ ]:
import torchsnooper
import utils
from kernel import KaggleKernel
from mask_functions import *

get_ipython().run_line_magic("reload_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
get_ipython().run_line_magic("matplotlib", "inline")


sys.path.insert(0, "../input/siim-acr-pneumothorax-segmentation")


# fastai.__version__

# copy pretrained weights for resnet34 to the folder fastai will search by default
Path("/tmp/.cache/torch/checkpoints/").mkdir(exist_ok=True, parents=True)
get_ipython().system(
    "cp '../input/resnet34/resnet34.pth' '/tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth'"
)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # tf.set_random_seed(seed)


# Accumulation of gradients to overcome the problem of too small batches. The code is mostly based on [this post](https://forums.fast.ai/t/accumulating-gradients/33219/25) with slight adjustment to work with mean reduction.
class AccumulateOptimWrapper(OptimWrapper):
    def step(self):
        pass

    def zero_grad(self):
        pass

    def real_step(self):
        super().step()

    def real_zero_grad(self):
        super().zero_grad()


def acc_create_opt(self, lr: Floats, wd: Floats = 0.0):
    "Create optimizer with `lr` learning rate and `wd` weight decay."
    self.opt = AccumulateOptimWrapper.create(
        self.opt_func,
        lr,
        self.layer_groups,
        wd=wd,
        true_wd=self.true_wd,
        bn_wd=self.bn_wd,
    )


Learner.create_opt = acc_create_opt

# Setting transformations on masks to False on test set


def transform(self, tfms: Optional[Tuple[TfmList, TfmList]] = (None, None), **kwargs):
    if not tfms:
        tfms = (None, None)
    assert is_listy(tfms) and len(tfms) == 2
    self.train.transform(tfms[0], **kwargs)
    self.valid.transform(tfms[1], **kwargs)
    kwargs["tfm_y"] = False  # Test data has no labels
    if self.test:
        self.test.transform(tfms[1], **kwargs)
    return self


fastai.data_block.ItemLists.transform = transform


class unetKernel(KaggleKernel):
    def __init__(self, stage="dev"):
        super(unetKernel, self).__init__()
        # The original images, provided in this competition, have 1024x1024 resolution. To prevent additional overhead on image loading, the datasets composed of 128x128 and 256x256 scaled down images are prepared separately and used as an input. Check [this keknel](https://www.kaggle.com/iafoss/data-repack-and-image-statistics) for more details on image rescaling and mask generation. Also In that kernel I apply image normalization based on histograms (exposure.equalize_adapthist) that provides some improvement of image appearance as well as a small boost of the model performance. The corresponding pixel statistics are computed in the kernel.
        self.sz = 256
        self.bs = 16
        self.n_acc = 64 // self.bs  # gradinet accumulation steps
        self.nfolds = 4

        self.SEED = 2019
        self.stage = stage

        # eliminate all predictions with a few (self.noise_th) pixesls
        # threshold for the number of predicted pixels
        self.noise_th = 75.0 * (self.sz / 128.0) ** 2
        self.best_thr0 = (
            0.2  # preliminary value of the threshold for metric calculation
        )

        if self.sz == 256:
            self.stats = ([0.540, 0.540, 0.540], [0.264, 0.264, 0.264])
            self.TRAIN = "../input/siimacr-pneumothorax-segmentation-data-256/train"
            self.TEST = "../input/siimacr-pneumothorax-segmentation-data-256/test"
            self.MASKS = "../input/siimacr-pneumothorax-segmentation-data-256/masks"
        elif self.sz == 128:
            self.stats = ([0.615, 0.615, 0.615], [0.291, 0.291, 0.291])
            self.TRAIN = "../input/siimacr-pneumothorax-segmentation-data-128/train"
            self.TEST = "../input/siimacr-pneumothorax-segmentation-data-128/test"
            self.MASKS = "../input/siimacr-pneumothorax-segmentation-data-128/masks"

        self.best_thr = 0.21

    def pre_prepare_data_hook(self):
        if self.stage == "dev":
            seed_everything(self.SEED)

    def _get_fold_data(self, fold):
        kf = KFold(n_splits=self.nfolds, shuffle=True, random_state=self.SEED)
        valid_idx = list(kf.split(list(range(len(Path(self.TRAIN).ls())))))[
            fold][1]
        # Create databunch
        data = (
            SegmentationItemList.from_folder(self.TRAIN)
            .split_by_idx(valid_idx)
            .label_from_func(lambda x: str(x).replace("train", "masks"), classes=[0, 1])
            .add_test(Path(self.TEST).ls(), label=None)
            .transform(get_transforms(), size=self.sz, tfm_y=True)
            .databunch(path=Path("."), bs=self.bs)
            .normalize(self.stats)
        )
        return data

    # A slight modification of the default dice metric to make it comparable with the competition metric: dice is computed for each image independently, and dice of empty image with zero prediction is 1. Also I use noise removal and similar threshold as in my prediction pipline.
    def _dice(
        self, input: Tensor, targs: Tensor, iou: bool = False, eps: float = 1e-8
    ) -> Rank0Tensor:
        n = targs.shape[0]
        input = torch.softmax(input, dim=1)[:, 1, ...].view(n, -1)
        input = (input > self.best_thr0).long()
        input[input.sum(-1) < self.noise_th, ...] = 0.0
        # input = input.argmax(dim=1).view(n,-1)
        targs = targs.view(n, -1)
        intersect = (input * targs).sum(-1).float()
        union = (input + targs).sum(-1).float()
        if not iou:
            return ((2.0 * intersect + eps) / (union + eps)).mean()
        else:
            return ((intersect + eps) / (union - intersect + eps)).mean()

    def _set_BN_momentum(self, model, momentum=None):
        if momentum is None:
            momentum = 0.1 * self.bs / 64
        for i, (name, layer) in enumerate(model.named_modules()):
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                layer.momentum = momentum

    def prepare_train_dev_data(self):
        data = utils.get_obj_or_dump("data0.bin")
        if data is None:
            data = self._get_fold_data(0)
            utils.dump_obj(data, "data0.bin")
        self.data0 = data

    def train_model(self):
        self.scores, self.best_thrs = [], []

        fold_run_cnt = 0
        for fold in range(self.nfolds):
            print("fold: ", fold)
            if hasattr(self, "data0") and fold == 0:
                data = self.data0
            else:
                data = self._get_fold_data(fold)

            learn: Learner = unet_learner(
                data,
                torchvision.models.resnext50_32x4d,
                callback_fns=partial(
                    utils.CSVLoggerBufferCustomized, append=True, filename="metric.log"
                ),
                cut=7,
                metrics=[self._dice],
                self_attention=True,
                bottle=True,
            )
            learn.clip_grad(1.0)
            self._set_BN_momentum(learn.model)

            self.learn = learn

            if fold == 0:
                print(learn.model)

            lr = 1e-3
            # fit the decoder part of the model keeping the encode frozen
            just_predict = False
            if Path("models/" + "fold" + str(fold) + "more" + ".pth").exists():
                learn.load("fold" + str(fold) + "more")
                lr /= 2
                just_predict = True
            elif Path("models/" + "fold" + str(fold) + ".pth").exists():
                learn.load("fold" + str(fold))
            else:
                learn.freeze()
                learn.fit_one_cycle(
                    4, lr, callbacks=[AccumulateStep(learn, self.n_acc)]
                )

                # fit entire model with saving on the best epoch
                learn.unfreeze()
                learn.fit_one_cycle(
                    4,
                    slice(lr / 80, lr / 2),
                    callbacks=[AccumulateStep(learn, self.n_acc)],
                )
                learn.save("fold" + str(fold))

            if not just_predict:
                learn.freeze_to(7)  # [6] won't be freezed, [:6] will
                lr = lr / 2
                learn.fit_one_cycle(
                    4, lr, callbacks=[AccumulateStep(learn, self.n_acc)]
                )
                learn.unfreeze()
                learn.freeze_to(2)  # backbone won't be trained...
                learn.fit_one_cycle(
                    4,
                    slice(lr / 80, lr / 2),
                    callbacks=[AccumulateStep(learn, self.n_acc)],
                )
                learn.save("fold" + str(fold) + "more")

            # prediction on val and test sets
            print("pred_with_flip")
            self.preds, self.ys = pred_with_flip(learn)
            pt, _ = pred_with_flip(learn, DatasetType.Test)

            if fold == 0:
                self.preds_test = pt
            else:
                self.preds_test += pt

            if hasattr(self, "best_thr"):
                break

            # convert predictions to byte type and save
            preds_save = (self.preds * 255.0).byte()
            torch.save(preds_save, "preds_fold" + str(fold) + ".pt")
            np.save("items_fold" + str(fold), data.valid_ds.items)

            # remove noise
            self.preds[
                self.preds.view(
                    self.preds.shape[0], -1).sum(-1) < self.noise_th, ...
            ] = 0.0

            # optimal threshold
            # The best way would be collecting all oof predictions followed by a single threshold
            # calculation. However, it requires too much RAM for high image resolution
            self.dices = []
            self.thrs = np.arange(0.01, 1, 0.01)
            for th in progress_bar(self.thrs):
                preds_m = (self.preds > th).long()
                self.dices.append(dice_overall(preds_m, self.ys).mean())
            self.dices = np.array(self.dices)
            self.scores.append(self.dices.max())
            self.best_thrs.append(self.thrs[self.dices.argmax()])

            # if fold != self.nfolds - 1: del self.preds, self.ys, preds_save
            gc.collect()
            torch.cuda.empty_cache()

            fold_run_cnt += 1
            break  # only run one fold

        self.preds_test /= fold_run_cnt

    def predict_test(self):
        print("pred_with_flip")
        self.preds_test, _ = pred_with_flip(self.learn, DatasetType.Test)
        self.predict_on_test(thr=0.9)

    # only dataloader ... others cannot dumped
    def dump_state(self, exec_flag=False, force=True):
        utils.logger.debug(f"state {self._stage}")
        if exec_flag:
            utils.logger.debug(f"dumping state {self._stage}")
            self_data = vars(self)

            names_to_exclude = {"model"}
            # adding preds, ys preds_test, 2.8GB....

            data_to_save = {
                k: v for k, v in self_data.items() if k not in names_to_exclude
            }

            utils.dump_obj(
                data_to_save, f"run_state_{self._stage}.pkl", force=False)

    def load_state_data_only(self, stage, file_name="run_state.pkl"):
        if stage is not None:
            file_name = f"run_state_{stage}.pkl"
        utils.logger.debug(f"restore from {file_name}")
        self_data = utils.get_obj_or_dump(filename=file_name)

        for k, v in self_data.items():
            setattr(self, k, v)

    def _print_scores_related(self):
        print("scores: ", self.scores)
        print("mean score: ", np.array(self.scores).mean())
        print("thresholds: ", self.best_thrs)
        self.best_thr = np.array(self.best_thrs).mean()
        print("best threshold: ", self.best_thr)

    def _analyze_dice_with_threshold(self):
        best_dice = self.dices.max()
        plt.figure(figsize=(8, 4))
        plt.plot(self.thrs, self.dices)
        plt.vlines(x=self.best_thrs[-1],
                   ymin=self.dices.min(), ymax=self.dices.max())
        plt.text(
            self.best_thrs[-1] + 0.03,
            best_dice - 0.01,
            f"DICE = {best_dice:.3f}",
            fontsize=14,
        )
        plt.show()

    def _plot_some_samples(self):
        rows = 10
        plot_idx = self.ys.sum((1, 2)).sort(descending=True).indices[:rows]
        for idx in plot_idx:
            fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))
            ax0.imshow(data.valid_ds[idx][0].data.numpy().transpose(1, 2, 0))
            ax1.imshow(self.ys[idx], vmin=0, vmax=1)
            ax2.imshow(self.preds[idx], vmin=0, vmax=1)
            ax1.set_title("Targets")
            ax2.set_title("Predictions")

    def predict_on_test(self, thr=None):
        # ### Submission
        # convert predictions to byte type and save
        preds_save = (self.preds_test * 255.0).byte()
        torch.save(preds_save, "preds_test.pt")

        # If any pixels are predicted for an empty mask, the corresponding image gets zero score during evaluation. While prediction of no pixels for an empty mask gives perfect score. Because of this penalty it is resonable to set masks to zero if the number of predicted pixels is small. This trick was quite efective in [Airbus Ship Detection Challenge](https://www.kaggle.com/iafoss/unet34-submission-tta-0-699-new-public-lb).
        self.preds_test[
            self.preds_test.view(
                self.preds_test.shape[0], -1).sum(-1) < self.noise_th,
            ...,
        ] = 0.0

        # Generate rle encodings (images are first converted to the original size)
        thr = self.best_thr if thr is None else thr
        self.preds_test = (self.preds_test > thr).long().numpy()
        rles = []
        for p in progress_bar(self.preds_test):
            im = PIL.Image.fromarray(
                (p.T * 255).astype(np.uint8)).resize((1024, 1024))
            im = np.asarray(im)
            rles.append(mask2rle(im, 1024, 1024))

        ids = [o.stem for o in self.learn.data.test_ds.items]
        sub_df = pd.DataFrame({"ImageId": ids, "EncodedPixels": rles})
        sub_df.loc[sub_df.EncodedPixels == "", "EncodedPixels"] = "-1"
        sub_df.to_csv("submission.csv", index=False)
        print(sub_df.head())

    def after_train(self):
        pass
        # self.print_scores_related()

        # self.analyze_dice_with_threshold()

        # self.plot_some_samples()


# ### Model

# The model used in this kernel is based on U-net like architecture with ResNet34 encoder. To boost the model performance, Hypercolumns are incorporated into DynamicUnet fast.ai class (see code below). The idea of Hypercolumns is schematically illustrated in the following figure. ![](https://i.ibb.co/3y7f8rj/Hypercolumns1.png)
# Each upscaling block is connected to the output layer through linear resize to the original image size. So the final image is produced based on concatenation of U-net output with resized outputs of intermediate layers. These skip-connections provide a shortcut for gradient flow improving model performance and convergence speed. Since intermediate layers have many channels, their upscaling and use as an input for the final layer would introduce a significant overhead in terms the computational time and memory. Therefore, 3x3 convolutions are applied (factorization) before the resize to reduce the number of channels.
# Further details on Hypercolumns can be found [here](http://home.bharathh.info/pubs/pdfs/BharathCVPR2015.pdf) and [here](https://towardsdatascience.com/review-hypercolumn-instance-segmentation-367180495979). Below the fast.ai code modified to incorporate Hypercolumns.


class Hcolumns(nn.Module):
    def __init__(self, hooks: Collection[Hook], nc: Collection[int] = None):
        super(Hcolumns, self).__init__()
        self.hooks = hooks
        self.n = len(self.hooks)
        self.factorization = None
        if nc is not None:
            self.factorization = nn.ModuleList()
            for i in range(self.n):
                # nc = [2048, 2048, 1536, 1024, 288, 291]
                self.factorization.append(
                    nn.Sequential(
                        conv2d(nc[i], nc[-1] // 2, 1, padding=0, bias=True),
                        conv2d(nc[-1] // 2, nc[-1] // 2,
                               3, padding=1, bias=True),
                        conv2d(nc[-1] // 2, nc[-1], 1, padding=0, bias=True),
                    )
                )
                # self.factorization.append(conv2d(nc[i],nc[-1],3,padding=1,bias=True))

    def forward(self, x: Tensor):
        n = len(self.hooks)
        out = [
            F.interpolate(
                self.hooks[i].stored
                if self.factorization is None
                else self.factorization[i](self.hooks[i].stored),
                scale_factor=2 ** (self.n - i),
                mode="bilinear",
                align_corners=False,
            )
            for i in range(self.n)
        ] + [x]
        return torch.cat(out, dim=1)


class DynamicUnet_Hcolumns(SequentialEx):
    "Create a U-Net from a given architecture."

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        blur: bool = False,
        blur_final=True,
        self_attention: bool = False,
        y_range: Optional[Tuple[float, float]] = None,
        last_cross: bool = True,
        bottle: bool = False,
        small=True,
        **kwargs,
    ):
        imsize = (256, 256)
        # for resnet50 ... but memory not enough...
        # sfs_szs = [(1, 64, 128, 128), (1, 64, 128, 128), (1, 64, 1...512, 32, 32), (1, 1024, 16, 16), (1, 2048, 8, 8)]
        # sfs_idxs = [6, 5, 4, 2]  #? 3?
        sfs_szs = model_sizes(encoder, size=imsize)
        # for resnext50_32x4d
        # [torch.Size([1, 64, 64, 64]), torch.Size([1, 64, 64, 64]), torch.Size([1, 64, 64, 64]), torch.Size([1, 64, 32, 32]), torch.Size([1, 256, 32, 32]), torch.Size([1, 512, 16, 16]), torch.Size([1, 1024, 8, 8]), torch.Size([1, 2048, 4, 4])]
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        # if small: sfs_idxs = sfs_idxs[-3:] (need to do double upscale)
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        if small:
            middle_conv_size_down_scale = 2
            middle_conv = conv_layer(
                ni, ni // middle_conv_size_down_scale, **kwargs
            ).eval()
        else:
            middle_conv_size_scale = 2
            middle_conv = nn.Sequential(
                conv_layer(ni, ni * middle_conv_size_scale, **kwargs),
                conv_layer(ni * middle_conv_size_scale, ni, **kwargs),
            ).eval()
        x = middle_conv(x)
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        if small:
            self.hc_hooks = []
            hc_c = []
        else:
            self.hc_hooks = [Hook(layers[-1], _hook_inner, detach=False)]
            hc_c = [x.shape[1]]

        for i, idx in enumerate(sfs_idxs):
            final_unet_flag = i == len(sfs_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            do_blur = blur and (final_unet_flag or blur_final)
            sa = self_attention and (i == len(sfs_idxs) - 3)
            unet_block_class = UnetBlockSmall if small else UnetBlock
            unet_block = unet_block_class(
                up_in_c,
                x_in_c,
                self.sfs[i],
                final_div=final_unet_flag,
                blur=blur,
                self_attention=sa,
                **kwargs,
            ).eval()
            print(unet_block)
            layers.append(unet_block)
            x = unet_block(x)
            # added for hypercolumns, two line
            self.hc_hooks.append(Hook(layers[-1], _hook_inner, detach=False))
            hc_c.append(x.shape[1])

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, **kwargs))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, **kwargs))
        # added for hypercolumns, two line
        hc_c.append(ni)
        layers.append(Hcolumns(self.hc_hooks, hc_c))
        layers += [
            conv_layer(ni * len(hc_c), n_classes,
                       ks=1, use_activ=False, **kwargs)
        ]
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"):
            self.sfs.remove()


class UnetBlockSmall(UnetBlock):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

    def __init__(
        self,
        up_in_c: int,
        x_in_c: int,
        hook: Hook,
        final_div: bool = True,
        blur: bool = False,
        leaky: float = None,
        self_attention: bool = False,
        **kwargs,
    ):
        self.hook = hook
        # need //2, as in forward path????
        self.shuf = PixelShuffle_ICNR(
            up_in_c, up_in_c // 2, blur=blur, leaky=leaky, **kwargs
        )
        self.bn = batchnorm_2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = ni if final_div else ni // 4
        self.conv1 = conv_layer(ni, nf, leaky=leaky, **kwargs)
        self.conv2 = conv_layer(
            nf, nf, leaky=leaky, self_attention=self_attention, **kwargs
        )
        self.relu = relu(leaky=leaky)


def unet_learner(
    data: DataBunch,
    arch: Callable,
    pretrained: bool = True,
    blur_final: bool = True,
    norm_type: Optional[NormType] = NormType,
    split_on: Optional[SplitFuncOrIdxList] = None,
    blur: bool = False,
    self_attention: bool = False,
    y_range: Optional[Tuple[float, float]] = None,
    last_cross: bool = True,
    bottle: bool = False,
    cut: Union[int, Callable] = None,
    hypercolumns=True,
    **learn_kwargs: Any,
) -> Learner:
    "Build Unet learner from `data` and `arch`."
    meta = cnn_config(arch)
    body = create_body(arch, pretrained, cut)
    M = DynamicUnet_Hcolumns if hypercolumns else models.unet.DynamicUnet
    model = to_device(
        M(
            body,
            n_classes=data.c,
            blur=blur,
            blur_final=blur_final,
            self_attention=self_attention,
            y_range=y_range,
            norm_type=norm_type,
            last_cross=last_cross,
            bottle=bottle,
        ),
        data.device,
    )
    learn = Learner(data, model, **learn_kwargs)
    learn.split(ifnone(split_on, meta["split"]))
    if pretrained:
        learn.freeze()
    apply_init(model[1], nn.init.kaiming_normal_)
    return learn


@dataclass
class AccumulateStep(LearnerCallback):
    """
    Does accumlated step every nth step by accumulating gradients
    """

    def __init__(self, learn: Learner, n_step: int = 1):
        super().__init__(learn)
        self.n_step = n_step

    def on_epoch_begin(self, **kwargs):
        "init samples and batches, change optimizer"
        self.acc_batches = 0

    def on_batch_begin(self, last_input, last_target, **kwargs):
        "accumulate samples and batches"
        self.acc_batches += 1

    def on_backward_end(self, **kwargs):
        "step if number of desired batches accumulated, reset samples"
        if (self.acc_batches % self.n_step) == self.n_step - 1:
            for p in self.learn.model.parameters():
                if p.requires_grad:
                    p.grad.div_(self.acc_batches)

            self.learn.opt.real_step()
            self.learn.opt.real_zero_grad()
            self.acc_batches = 0

    def on_epoch_end(self, **kwargs):
        "step the rest of the accumulated grads"
        if self.acc_batches > 0:
            for p in self.learn.model.parameters():
                if p.requires_grad:
                    p.grad.div_(self.acc_batches)
            self.learn.opt.real_step()
            self.learn.opt.real_zero_grad()
            self.acc_batches = 0


# dice for threshold selection
def dice_overall(preds, targs):
    n = preds.shape[0]
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    intersect = (preds * targs).sum(-1).float()
    union = (preds + targs).sum(-1).float()
    u0 = union == 0
    intersect[u0] = 1
    union[u0] = 2
    return 2.0 * intersect / union


# The following function generates predictions with using flip TTA (average the result for the original image and a flipped one).
# Prediction with flip TTA
def pred_with_flip(
    learn: fastai.basic_train.Learner,
    ds_type: fastai.basic_data.DatasetType = DatasetType.Valid,
):
    # get prediction
    preds, ys = learn.get_preds(ds_type)
    preds = preds[:, 1, ...]
    # add fiip to dataset and get prediction
    learn.data.dl(ds_type).dl.dataset.tfms.append(flip_lr())
    preds_lr, ys = learn.get_preds(ds_type)
    del learn.data.dl(ds_type).dl.dataset.tfms[-1]
    preds_lr = preds_lr[:, 1, ...]
    ys = ys.squeeze()
    preds = 0.5 * (preds + torch.flip(preds_lr, [-1]))
    del preds_lr
    gc.collect()
    torch.cuda.empty_cache()
    return preds, ys


# Setting div=True in open_mask
class SegmentationLabelList(SegmentationLabelList):
    def open(self, fn):
        return open_mask(fn, div=True)


class SegmentationItemList(SegmentationItemList):
    _label_cls = SegmentationLabelList
