# todo
1. add attention
2. add groupNorm
3. PReLU
4. refine ...
5. loss, multi-layer, just force, maybe training more easily
```sh
# https://www.kaggle.com/iafoss/hypercolumns-pneumothorax-fastai-0-819-lb

# Know about how to do it right. do research first!!! -> In prior image segmentation competitions (Airbus Ship Detection Challenge and TGS Salt Identification Challenge), U-net based model architecture has demonstrated supperior performence, and top solutions are based on it.

# to know about gradient accumulation

# to check Vanishing / Exploding Gradients (how the calculation is done)

# Hypercolumn (for Instance Segmentation)
# A hyptercolumn at ONE LOCATION is one long vector concatenated the features from some or all the feature maps in the network
```
```
DynamicUnet_Hcolumns(
  (layers): ModuleList(
    (0): Sequential(
      (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
      (2): ReLU(inplace)
      (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (4): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (downsample): Sequential(
            (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
        (2): Bottleneck(
          (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
      )
      (5): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
        (2): Bottleneck(
          (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
        (3): Bottleneck(
          (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
      )
      (6): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (downsample): Sequential(
            (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(1024, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
        (2): Bottleneck(
          (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
        (3): Bottleneck(
          (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
        (4): Bottleneck(
          (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
        (5): Bottleneck(
          (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
      )
      (7): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(1024, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(1024, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv3): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (downsample): Sequential(
            (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(2048, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(1024, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(1024, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv3): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
        (2): Bottleneck(
          (conv1): Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(1024, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (bn2): BatchNorm2d(1024, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (conv3): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
        )
      )
    )
    (1): BatchNorm2d(2048, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Sequential(
      (0): Sequential(
        (0): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
      )
      (1): Sequential(
        (0): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
      )
    )
    (4): UnetBlock(
      (shuf): PixelShuffle_ICNR(
        (conv): Sequential(
          (0): Conv2d(2048, 4096, kernel_size=(1, 1), stride=(1, 1))
        )
        (shuf): PixelShuffle(upscale_factor=2)
        (pad): ReplicationPad2d((1, 0, 1, 0))
        (blur): AvgPool2d(kernel_size=2, stride=1, padding=0)
        (relu): ReLU(inplace)
      )
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
      (conv1): Sequential(
        (0): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
        (2): SelfAttention(
          (query): Conv1d(1536, 192, kernel_size=(1,), stride=(1,), bias=False)
          (key): Conv1d(1536, 192, kernel_size=(1,), stride=(1,), bias=False)
          (value): Conv1d(1536, 1536, kernel_size=(1,), stride=(1,), bias=False)
        )
      )
      (relu): ReLU()
    )
    (5): UnetBlock(
      (shuf): PixelShuffle_ICNR(
        (conv): Sequential(
          (0): Conv2d(1536, 3072, kernel_size=(1, 1), stride=(1, 1))
        )
        (shuf): PixelShuffle(upscale_factor=2)
        (pad): ReplicationPad2d((1, 0, 1, 0))
        (blur): AvgPool2d(kernel_size=2, stride=1, padding=0)
        (relu): ReLU(inplace)
      )
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
      (conv1): Sequential(
        (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
      )
      (relu): ReLU()
    )
    (6): UnetBlock(
      (shuf): PixelShuffle_ICNR(
        (conv): Sequential(
          (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1))
        )
        (shuf): PixelShuffle(upscale_factor=2)
        (pad): ReplicationPad2d((1, 0, 1, 0))
        (blur): AvgPool2d(kernel_size=2, stride=1, padding=0)
        (relu): ReLU(inplace)
      )
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.00625, affine=True, track_running_stats=True)
      (conv1): Sequential(
        (0): Conv2d(576, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
      )
      (conv2): Sequential(
        (0): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
      )
      (relu): ReLU()
    )
    (7): PixelShuffle_ICNR(
      (conv): Sequential(
        (0): Conv2d(288, 1152, kernel_size=(1, 1), stride=(1, 1))
      )
      (shuf): PixelShuffle(upscale_factor=2)
      (pad): ReplicationPad2d((1, 0, 1, 0))
      (blur): AvgPool2d(kernel_size=2, stride=1, padding=0)
      (relu): ReLU(inplace)
    )
    (8): MergeLayer()
    (9): SequentialEx(
      (layers): ModuleList(
        (0): Sequential(
          (0): Conv2d(291, 145, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU(inplace)
        )
        (1): Sequential(
          (0): Conv2d(145, 291, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU(inplace)
        )
        (2): MergeLayer()
      )
    )
    (10): Hcolumns(
      (factorization): ModuleList(
        (0): Sequential(
          (0): Conv2d(1536, 145, kernel_size=(1, 1), stride=(1, 1))
          (1): Conv2d(145, 145, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (2): Conv2d(145, 291, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): Sequential(
          (0): Conv2d(1024, 145, kernel_size=(1, 1), stride=(1, 1))
          (1): Conv2d(145, 145, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (2): Conv2d(145, 291, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): Sequential(
          (0): Conv2d(288, 145, kernel_size=(1, 1), stride=(1, 1))
          (1): Conv2d(145, 145, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (2): Conv2d(145, 291, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    (11): Sequential(
      (0): Conv2d(1164, 2, kernel_size=(1, 1), stride=(1, 1))
    )
  )
)
```
tarting var:.. self = DynamicUnet_Hcolumns()
Starting var:.. encoder = Sequential(  (0): Conv2d(3, 64, kernel_size=(7, ...g_stats=True)      (relu): ReLU(inplace)    )  ))
Starting var:.. n_classes = 2
Starting var:.. blur = False
Starting var:.. blur_final = True
Starting var:.. self_attention = True
Starting var:.. y_range = None
Starting var:.. last_cross = True
Starting var:.. bottle = True
Starting var:.. small = True
Starting var:.. kwargs = {'norm_type': <enum 'NormType'>}
Starting var:.. __class__ = <class 'hc_unet.DynamicUnet_Hcolumns'>
14:05:53.883935 call       313     def __init__(self, encoder: nn.Module, n_classes: int, blur: bool = False, blur_final=True,
14:05:53.885016 line       317         imsize = (128, 128)
New var:....... imsize = (128, 128)
14:05:53.886027 line       321         sfs_szs = model_sizes(encoder, size=imsize)
New var:....... sfs_szs = [(1, 64, 64, 64), (1, 64, 64, 64), (1, 64, 64, 6... 256, 32, 32), (1, 512, 16, 16), (1, 1024, 8, 8)]
14:05:54.001826 line       324         sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
New var:....... sfs_idxs = [5, 4, 2]
14:05:54.003075 line       326         self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
14:05:54.004282 line       327         x = dummy_eval(encoder, imsize).detach()
New var:....... x = tensor<(1, 1024, 8, 8), float32, cpu>
14:05:54.112064 line       329         ni = sfs_szs[-1][1]
New var:....... ni = 1024
14:05:54.114533 line       330         middle_conv_size_scale = 2
New var:....... middle_conv_size_scale = 2
14:05:54.116317 line       331         if small:
14:05:54.118268 line       332             middle_conv_size_scale = 1
Modified var:.. middle_conv_size_scale = 1
14:05:54.120035 line       333         middle_conv = nn.Sequential(conv_layer(ni, ni * middle_conv_size_scale, **kwargs),
14:05:54.298226 line       334                                     conv_layer(ni * middle_conv_size_scale, ni, **kwargs)).eval()
New var:....... middle_conv = Sequential(  (0): Sequential(    (0): Conv2d(102...(1, 1), padding=(1, 1))    (1): ReLU(inplace)  ))
14:05:54.477163 line       335         x = middle_conv(x)
Modified var:.. x = tensor<(1, 1024, 8, 8), float32, cpu, grad>
14:05:54.575460 line       336         layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]
New var:....... layers = [Sequential(  (0): Conv2d(3, 64, kernel_size=(7,...1, 1), padding=(1, 1))    (1): ReLU(inplace)  ))]
14:05:54.579321 line       338         if small:
14:05:54.582164 line       339             self.hc_hooks = []
14:05:54.584819 line       340             hc_c = []
New var:....... hc_c = []
14:05:54.587605 line       345         for i, idx in enumerate(sfs_idxs):
New var:....... i = 0
New var:....... idx = 5
14:05:54.590452 line       346             not_final = i != len(sfs_idxs) - 1
New var:....... not_final = True
14:05:54.593179 line       347             up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
New var:....... up_in_c = 1024
New var:....... x_in_c = 512
14:05:54.595930 line       348             do_blur = blur and (not_final or blur_final)
New var:....... do_blur = False
14:05:54.598838 line       349             sa = self_attention and (i == len(sfs_idxs) - 3)
New var:....... sa = True
14:05:54.601474 line       350             unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final,
14:05:54.604065 line       351                                    blur=blur, self_attention=sa, **kwargs).eval()
New var:....... unet_block = UnetBlock(  (shuf): PixelShuffle_ICNR(    (conv)...stride=(1,), bias=False)    )  )  (relu): ReLU())
14:05:55.046451 line       352             layers.append(unet_block)
Modified var:.. layers = [Sequential(  (0): Conv2d(3, 64, kernel_size=(7,...tride=(1,), bias=False)    )  )  (relu): ReLU())]
14:05:55.050323 line       353             x = unet_block(x)
Modified var:.. x = tensor<(1, 1024, 16, 16), float32, cpu, grad>
14:05:55.425198 line       355             self.hc_hooks.append(Hook(layers[-1], _hook_inner, detach=False))
14:05:55.429178 line       356             hc_c.append(x.shape[1])
Modified var:.. hc_c = [1024]
14:05:55.433007 line       345         for i, idx in enumerate(sfs_idxs):
Modified var:.. i = 1
Modified var:.. idx = 4
14:05:55.436825 line       346             not_final = i != len(sfs_idxs) - 1
14:05:55.440832 line       347             up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
Modified var:.. x_in_c = 256
14:05:55.444602 line       348             do_blur = blur and (not_final or blur_final)
14:05:55.448379 line       349             sa = self_attention and (i == len(sfs_idxs) - 3)
Modified var:.. sa = False
14:05:55.452148 line       350             unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final,
14:05:55.456079 line       351                                    blur=blur, self_attention=sa, **kwargs).eval()
Modified var:.. unet_block = UnetBlock(  (shuf): PixelShuffle_ICNR(    (conv)...(1, 1))    (1): ReLU(inplace)  )  (relu): ReLU())
14:05:55.697351 line       352             layers.append(unet_block)
Modified var:.. layers = [Sequential(  (0): Conv2d(3, 64, kernel_size=(7,...1, 1))    (1): ReLU(inplace)  )  (relu): ReLU())]
14:05:55.701585 line       353             x = unet_block(x)
Modified var:.. x = tensor<(1, 768, 32, 32), float32, cpu, grad>
14:05:55.996809 line       355             self.hc_hooks.append(Hook(layers[-1], _hook_inner, detach=False))
14:05:56.003463 line       356             hc_c.append(x.shape[1])
Modified var:.. hc_c = [1024, 768]
14:05:56.010498 line       345         for i, idx in enumerate(sfs_idxs):
Modified var:.. i = 2
Modified var:.. idx = 2
14:05:56.016782 line       346             not_final = i != len(sfs_idxs) - 1
Modified var:.. not_final = False
14:05:56.023233 line       347             up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
Modified var:.. up_in_c = 768
Modified var:.. x_in_c = 64
14:05:56.029500 line       348             do_blur = blur and (not_final or blur_final)
14:05:56.036455 line       349             sa = self_attention and (i == len(sfs_idxs) - 3)
14:05:56.042908 line       350             unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final,
14:05:56.049055 line       351                                    blur=blur, self_attention=sa, **kwargs).eval()
14:05:56.145423 line       352             layers.append(unet_block)
14:05:56.167285 line       353             x = unet_block(x)
Modified var:.. x = tensor<(1, 224, 64, 64), float32, cpu, grad>
14:05:56.409598 line       355             self.hc_hooks.append(Hook(layers[-1], _hook_inner, detach=False))
14:05:56.417376 line       356             hc_c.append(x.shape[1])
Modified var:.. hc_c = [1024, 768, 224]
14:05:56.424618 line       345         for i, idx in enumerate(sfs_idxs):
14:05:56.431524 line       358         ni = x.shape[1]
Modified var:.. ni = 224
14:05:56.438593 line       359         if imsize != sfs_szs[0][-2:]: layers.append(PixelShuffle_ICNR(ni, **kwargs))
Modified var:.. layers = [Sequential(  (0): Conv2d(3, 64, kernel_size=(7,...=2, stride=1, padding=0)  (relu): ReLU(inplace))]
14:05:56.452482 line       360         if last_cross:
14:05:56.459827 line       361             layers.append(MergeLayer(dense=True))
Modified var:.. layers = [Sequential(  (0): Conv2d(3, 64, kernel_size=(7,...padding=0)  (relu): ReLU(inplace)), MergeLayer()]
14:05:56.466979 line       362             ni += in_channels(encoder)
Modified var:.. ni = 227
14:05:56.486477 line       363             layers.append(res_block(ni, bottle=bottle, **kwargs))
Modified var:.. layers = [Sequential(  (0): Conv2d(3, 64, kernel_size=(7,...(1): ReLU(inplace)    )    (2): MergeLayer()  ))]
14:05:56.503463 line       365         hc_c.append(ni)
Modified var:.. hc_c = [1024, 768, 224, 227]
14:05:56.510710 line       366         layers.append(Hcolumns(self.hc_hooks, hc_c))
Modified var:.. layers = [Sequential(  (0): Conv2d(3, 64, kernel_size=(7,...227, kernel_size=(1, 1), stride=(1, 1))    )  ))]
14:05:56.532687 line       367         layers += [conv_layer(ni * len(hc_c), n_classes, ks=1, use_activ=False, **kwargs)]
Modified var:.. layers = [Sequential(  (0): Conv2d(3, 64, kernel_size=(7,...nv2d(908, 2, kernel_size=(1, 1), stride=(1, 1)))]
14:05:56.541413 line       368         if y_range is not None: layers.append(SigmoidRange(*y_range))
14:05:56.548957 line       369         super().__init__(*layers)
Modified var:.. self = DynamicUnet_Hcolumns(  (layers): ModuleList(    ...8, 2, kernel_size=(1, 1), stride=(1, 1))    )  ))
14:05:56.557870 return     369         super().__init__(*layers)
Return value:.. None



Starting var:.. self = DynamicUnet_Hcolumns()
Starting var:.. encoder = Sequential(  (0): Conv2d(3, 64, kernel_size=(7, ...g_stats=True)      (relu): ReLU(inplace)    )  ))
Starting var:.. n_classes = 2
Starting var:.. blur = False
Starting var:.. blur_final = True
Starting var:.. self_attention = True
Starting var:.. y_range = None
Starting var:.. last_cross = True
Starting var:.. bottle = True
Starting var:.. small = True
Starting var:.. kwargs = {'norm_type': <enum 'NormType'>}
Starting var:.. __class__ = <class 'hc_unet.DynamicUnet_Hcolumns'>
14:39:29.296546 call       313     def __init__(self, encoder: nn.Module, n_classes: int, blur: bool = False, blur_final=True,
14:39:29.297474 line       317         imsize = (128, 128)
New var:....... imsize = (256, 256)
14:39:29.298386 line       321         sfs_szs = model_sizes(encoder, size=imsize)
New var:....... sfs_szs = [(1, 64, 128, 128), (1, 64, 128, 128), (1, 64, 1...56, 64, 64), (1, 512, 32, 32), (1, 1024, 16, 16)]
14:39:29.560884 line       324         sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
New var:....... sfs_idxs = [5, 4, 2]
14:39:29.562043 line       326         self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
14:39:29.563111 line       327         x = dummy_eval(encoder, imsize).detach()
New var:....... x = tensor<(1, 1024, 16, 16), float32, cpu>
14:39:29.802480 line       329         ni = sfs_szs[-1][1]
New var:....... ni = 1024
14:39:29.805044 line       330         middle_conv_size_scale = 2
New var:....... middle_conv_size_scale = 2
14:39:29.807557 line       331         if small:
14:39:29.809937 line       332             middle_conv_size_scale = 1
Modified var:.. middle_conv_size_scale = 1
14:39:29.812415 line       333         middle_conv = nn.Sequential(conv_layer(ni, ni * middle_conv_size_scale, **kwargs),
14:39:29.967995 line       334                                     conv_layer(ni * middle_conv_size_scale, ni, **kwargs)).eval()
New var:....... middle_conv = Sequential(  (0): Sequential(    (0): Conv2d(102...(1, 1), padding=(1, 1))    (1): ReLU(inplace)  ))
14:39:30.134751 line       335         x = middle_conv(x)
Modified var:.. x = tensor<(1, 1024, 16, 16), float32, cpu, grad>
14:39:30.334625 line       336         layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]
New var:....... layers = [Sequential(  (0): Conv2d(3, 64, kernel_size=(7,...1, 1), padding=(1, 1))    (1): ReLU(inplace)  ))]
14:39:30.340356 line       338         if small:
14:39:30.344191 line       339             self.hc_hooks = []
14:39:30.348036 line       340             hc_c = []
New var:....... hc_c = []
14:39:30.351826 line       345         for i, idx in enumerate(sfs_idxs):
New var:....... i = 0
New var:....... idx = 5
14:39:30.355548 line       346             not_final = i != len(sfs_idxs) - 1
New var:....... not_final = True
14:39:30.359710 line       347             up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
New var:....... up_in_c = 1024
New var:....... x_in_c = 512
14:39:30.363171 line       348             do_blur = blur and (not_final or blur_final)
New var:....... do_blur = False
14:39:30.366705 line       349             sa = self_attention and (i == len(sfs_idxs) - 3)
New var:....... sa = True
14:39:30.370180 line       350             unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final,
New var:....... unet_block_class = <class 'hc_unet.UnetBlockSmall'>
14:39:30.373784 line       351                                    blur=blur, self_attention=sa, **kwargs).eval()
14:39:30.377306 line       352             layers.append(unet_block)
New var:....... unet_block = UnetBlockSmall(  (shuf): PixelShuffle_ICNR(    (...stride=(1,), bias=False)    )  )  (relu): ReLU())
14:39:30.787862 line       353             x = unet_block(x)
Modified var:.. layers = [Sequential(  (0): Conv2d(3, 64, kernel_size=(7,...tride=(1,), bias=False)    )  )  (relu): ReLU())]
14:39:30.791989 line       354             # added for hypercolumns, two line
14:39:30.818228 exception  354             # added for hypercolumns, two line
RuntimeError: Given groups=1, weight of size 1024 1024 3 3, expected input[1, 768, 32, 32] to have 1024 channels, but got 768 channels instead
Call ended by exception


 Starting var:.. kwargs = {'norm_type': <enum 'NormType'>}
    14:44:27.657667 call       377                  blur: bool = False, self_attention: bool = False, y_range: Optional[Tuple[float, float]] = None,
    14:44:27.657791 line       380     "Build Unet learner from `data` and `arch`."
    14:44:27.657916 line       381     meta = cnn_config(arch)
    Modified var:.. self = UnetBlockSmall(  (shuf): PixelShuffle_ICNR(    (...stride=1, padding=0)    (relu): ReLU(inplace)  ))
    14:44:27.686388 line       382     body = create_body(arch, pretrained, cut)
    Modified var:.. self = UnetBlockSmall(  (shuf): PixelShuffle_ICNR(    (...ntum=0.1, affine=True, track_running_stats=True))
    14:44:27.687293 line       383     M = DynamicUnet_Hcolumns if hypercolumns else models.unet.DynamicUnet
    New var:....... ni = 1024
    14:44:27.687524 line       384     model = to_device(M(body, n_classes=data.c, blur=blur, blur_final=blur_final,
    New var:....... nf = 1024
    14:44:27.687720 line       385                         self_attention=self_attention, y_range=y_range, norm_type=norm_type,
    Modified var:.. self = UnetBlockSmall(  (shuf): PixelShuffle_ICNR(    (...(1, 1), padding=(1, 1))    (1): ReLU(inplace)  ))
    14:44:27.842509 line       386                         last_cross=last_cross, bottle=bottle), data.device)
    Modified var:.. self = UnetBlockSmall(  (shuf): PixelShuffle_ICNR(    (...rnel_size=(1,), stride=(1,), bias=False)    )  ))
    14:44:28.031905 line       387     learn = Learner(data, model, **learn_kwargs)
    Modified var:.. self = UnetBlockSmall(  (shuf): PixelShuffle_ICNR(    (...stride=(1,), bias=False)    )  )  (relu): ReLU())
    14:44:28.032280 return     387     learn = Learner(data, model, **learn_kwargs)
    Return value:.. None
New var:....... unet_block = UnetBlockSmall(  (shuf): PixelShuffle_ICNR(    (...stride=(1,), bias=False)    )  )  (relu): ReLU())
14:44:28.036791 line       353             x = unet_block(x)
Modified var:.. layers = [Sequential(  (0): Conv2d(3, 64, kernel_size=(7,...tride=(1,), bias=False)    )  )  (relu): ReLU())]
14:44:28.040609 line       354             # added for hypercolumns, two line
14:44:28.063340 exception  354             # added for hypercolumns, two line
RuntimeError: Given groups=1, weight of size 1024 1024 3 3, expected input[1, 768, 32, 32] to have 1024 channels, but got 768 channels instead
Call ended by exception
E