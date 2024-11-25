import torch.nn as nn
import torch
from torch.cuda.amp import autocast, GradScaler

class InferUVTRSSL():
    """As stated in https://github.com/huggingface/accelerate/issues/668, we
    should not use accumulate provided by accelerator, but create a wrapper to
    two modules.
    """

    def __init__(self, UVTRSSL_Wrapper) -> None:
        super().__init__()
        # self.controlnet = controlnet
        # self.unet = unet
        # self.weight_dtype = weight_dtype
        # self.unet_in_fp16 = unet_in_fp16

        self.UVTRSSL_Wrapper = UVTRSSL_Wrapper

    @torch.no_grad()
    def __call__(self, batch):

        with autocast():
            controlnet_image = self.UVTRSSL_Wrapper(False, batch)

        return controlnet_image