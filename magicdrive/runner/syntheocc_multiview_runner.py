import logging
import os
import contextlib
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers import (
    ModelMixin,
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers.optimization import get_scheduler

from ..misc.common import load_module, convert_outputs_to_fp16, move_to
from .base_runner import BaseRunner
from .utils import smart_param_count

import numpy as np

# from .multiview_runner import ControlnetUnetWrapper, MultiviewRunner
from .multiview_runner import MultiviewRunner
from ..networks.UniPAD.uvtr_ssl import UVTRSSL
from mmcv import Config, DictAction

from ..networks.cldm.volume_transform import VolumeTransform
from diffusers.configuration_utils import register_to_config, ConfigMixin
from magicdrive.dataset.syntheocc_utils import collate_fn
from functools import partial


class SyntheOccControlnetUnetWrapper(ModelMixin):
    """As stated in https://github.com/huggingface/accelerate/issues/668, we
    should not use accumulate provided by accelerator, but create a wrapper to
    two modules.
    """

    def __init__(self, controlnet, unet, weight_dtype=torch.float32,
                 unet_in_fp16=True) -> None:
        super().__init__()
        self.controlnet = controlnet
        self.unet = unet
        self.weight_dtype = weight_dtype
        self.unet_in_fp16 = unet_in_fp16

    def forward(self, noisy_latents, timesteps, camera_param,
                encoder_hidden_states, encoder_hidden_states_uncond,
                controlnet_image, **kwargs):
        # N_cam = noisy_latents.shape[1]
        N_cam = camera_param.shape[1]
        kwargs = move_to(
            kwargs, self.weight_dtype, lambda x: x.dtype == torch.float32)

        # fmt: off
        # down_block_res_samples, mid_block_res_sample, \
        # encoder_hidden_states_with_cam = self.controlnet(
        #     noisy_latents,  # b, N_cam, 4, H/8, W/8
        #     timesteps,  # b
        #     camera_param=camera_param,  # b, N_cam, 189
        #     encoder_hidden_states=encoder_hidden_states,  # b, len, 768
        #     encoder_hidden_states_uncond=encoder_hidden_states_uncond,  # 1, len, 768
        #     controlnet_cond=controlnet_image,  # b, 26, 200, 200
        #     return_dict=False,
        #     **kwargs,
        # )
        # fmt: on


        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,  # b, N_cam, 4, H/8, W/8
            timesteps,  # b
            encoder_hidden_states=encoder_hidden_states,  # b, len, 768
            controlnet_cond=controlnet_image,  # b, 26, 200, 200
            return_dict=False,
        )


        # starting from here, we use (B n) as batch_size
        # noisy_latents = rearrange(noisy_latents, "b n ... -> (b n) ...")
        # if timesteps.ndim == 1:
        #     timesteps = repeat(timesteps, "b -> (b n)", n=N_cam)

        # Predict the noise residual
        # NOTE: Since we fix most of the model, we cast the model to fp16 and
        # disable autocast to prevent it from falling back to fp32. Please
        # enable autocast on your customized/trainable modules.
        context = contextlib.nullcontext
        context_kwargs = {}
        if self.unet_in_fp16:
            context = torch.cuda.amp.autocast
            context_kwargs = {"enabled": False}
        with context(**context_kwargs):
            model_pred = self.unet(
                noisy_latents,  # b x n, 4, H/8, W/8
                timesteps.reshape(-1),  # b x n
                encoder_hidden_states=encoder_hidden_states.to(
                    dtype=self.weight_dtype
                ),  # b x n, len + 1, 768
                # TODO: during training, some camera param are masked.
                down_block_additional_residuals=[
                    sample.to(dtype=self.weight_dtype)
                    for sample in down_block_res_samples
                ],  # all intermedite have four dims: b x n, c, h, w
                mid_block_additional_residual=mid_block_res_sample.to(
                    dtype=self.weight_dtype
                ),  # b x n, 1280, h, w. we have 4 x 7 as mid_block_res
            ).sample

        # model_pred = rearrange(model_pred, "(b n) ... -> b n ...", n=N_cam)
        return model_pred




class UVTRSSL_Wrapper(ModelMixin, ConfigMixin):
    def __init__(self, ) -> None:
        super().__init__()
        cfg = Config.fromfile('magicdrive/networks/UniPAD/configs/uvtr_cam_vs0.075_pretrain.py')
        args = cfg.model.copy()
        obj_type = args.pop('type')
        self.UVTRSSL = UVTRSSL(**args)



        # # origin_occ_shape = (5, 180, 180)
        # origin_occ_shape = (20, 180, 180)
        # input_size = (224, 400)
        # down_sample = 8
        # grid_config = dict(
        #     x_bound = [-54.0, 54.0, 0.6],
        #     y_bound = [-54.0, 54.0, 0.6],
        #     # z_bound = [-5.0, 3.0, 1.6],
        #     z_bound = [-5.0, 3.0, 0.4],
        #     d_bound = [0.5, 48.5, 1.0],
        # )

        # origin_occ_shape = (5, 180, 180)
        origin_occ_shape = (2, 18, 18)
        input_size = (224, 400)
        down_sample = 8
        grid_config = dict(
            x_bound = [-54.0, 54.0, 6],
            y_bound = [-54.0, 54.0, 6],
            # z_bound = [-5.0, 3.0, 1.6],
            z_bound = [-5.0, 3.0, 4],
            d_bound = [0.5, 48.5, 1.0],
        )

        self.VT = VolumeTransform(with_DSE=True, 
                                  origin_occ_shape=origin_occ_shape, 
                                  input_size=input_size,
                                  down_sample=down_sample,
                                  grid_config=grid_config,
                                  )



    def forward(self, return_loss=True, batch=None):
        # TODO
        # with torch.no_grad():
        volume_feats = self.UVTRSSL(return_loss, **batch['unipad'])

        B = volume_feats.shape[0]
        N = 6
        # cyc = lambda i:(i+N)%N
        # # t = t.repeat(3)
        # target_index = torch.randint(0, 6, (B, 1), device=volume_feats.device)
        # left_index, right_index = cyc(target_index - 1), cyc(target_index + 1)
        # index = torch.cat([target_index, left_index, right_index], dim=-1)

        # T, K = hint['T'], hint['K']
        # lidar2img = []
        # for cur_img_metas in batch['img_metas']:
        #     cur_lidar2img = np.array(cur_img_metas['lidar2img'])
        #     lidar2img.append(cur_lidar2img)
        # lidar2img = np.array(lidar2img)
        # lidar2img = torch.tensor(lidar2img, device=volume_feats.device)
        # lidar2img = lidar2img.squeeze(2)



        # cam_intrinsic, lidar2cam = [], []
        # for cur_img_metas in batch['img_metas']:
        #     cur_cam_intrinsic = np.array(cur_img_metas['cam_intrinsic'])
        #     cam_intrinsic.append(cur_cam_intrinsic)
        #     cur_lidar2cam = np.array(cur_img_metas['lidar2cam'])
        #     lidar2cam.append(cur_lidar2cam)
        # cam_intrinsic = np.array(cam_intrinsic)
        # cam_intrinsic = torch.tensor(cam_intrinsic, dtype=volume_feats.dtype, device=volume_feats.device)
        # cam_intrinsic = cam_intrinsic.squeeze(2)

        # lidar2cam = np.array(lidar2cam)
        # lidar2cam = torch.tensor(lidar2cam, dtype=volume_feats.dtype, device=volume_feats.device)
        # lidar2cam = lidar2cam.squeeze(2)

        # T = lidar2cam
        # K = cam_intrinsic[..., :3, :3]




        cam_intrinsic, cam2lidar = [], []
        for cur_b in range(B):
            cur_cam_param = batch['camera_param'][cur_b]
            cam_intrinsic.append(cur_cam_param[:, :, :3])
            cam2lidar.append(cur_cam_param[:, :, 3:])
        cam_intrinsic = torch.stack(cam_intrinsic)
        resize = 0.25
        crop = torch.Tensor([0, 1]).to(cam_intrinsic)
        cam_intrinsic[:, :, :2, :2] *= resize
        cam_intrinsic[:, :, :2, 2] -= crop

        cam2lidar = torch.stack(cam2lidar)
        padcam2lidar = repeat(torch.eye(4).to(cam2lidar), 'p q -> bs N p q', bs=B, N=N)
        padcam2lidar[:, :, :3] = cam2lidar

        T = padcam2lidar
        K = cam_intrinsic

        # # camera to lidar transform
        # camera2lidar = np.eye(4).astype(np.float32)
        # camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
        # camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
        # data["camera2lidar"].append(camera2lidar)

        # # fmt: off
        # camera_param = torch.stack([torch.cat([
        #     example["camera_intrinsics"].data[:, :3, :3],  # 3x3 is enough
        #     example["camera2lidar"].data[:, :3],  # only first 3 rows meaningful
        # ], dim=-1) for example in examples], dim=0)
        # # fmt: on



        # msk, msk_txt = hint['msk'], hint['msk_txt']
        # msk_txt = msk_txt.repeat(3,1,1,1)  # 3,6,77,1024
        # msk[:,[3,5],...] = msk[:,[5,3],...]

            

        # T = T[torch.arange(B)[:,None], index, ...].reshape(B*3, T.shape[2], T.shape[3])  # B,3,4,4
        # K = K[torch.arange(B)[:,None], index, ...].reshape(B*3, K.shape[2], K.shape[3])  # B*3,3,3
        # occ_feat = occ_feat['x']
        # occ_feat = occ_feat.unsqueeze(1).repeat(1,3,1,1,1,1)\
        #             .reshape(B*3, occ_feat.shape[1], occ_feat.shape[2], occ_feat.shape[3], occ_feat.shape[4])  # B*3,C,H,W,D
        # volume_feats = self.VT(occ_feat, K, T, target_index)  # B*3,C,H,W


        # # lidar2img = lidar2img[torch.arange(B)[:,None], index, ...].reshape(B*3, lidar2img.shape[2], lidar2img.shape[3])  # B,3,4,4
        # T = T[torch.arange(B)[:,None], index, ...].reshape(B*3, T.shape[2], T.shape[3])  # B,3,4,4
        # K = K[torch.arange(B)[:,None], index, ...].reshape(B*3, K.shape[2], K.shape[3])  # B*3,3,3
        # volume_feats = volume_feats.unsqueeze(1).repeat(1,3,1,1,1,1)\
        #             .reshape(B*3, volume_feats.shape[1], volume_feats.shape[2], volume_feats.shape[3], volume_feats.shape[4])  # B*3,C,H,W,D
        # volume_feats = self.VT(volume_feats, K, T, target_index)  # B*3,C,H,W


        T = T.reshape(B*6, T.shape[2], T.shape[3])  # B,3,4,4
        K = K.reshape(B*6, K.shape[2], K.shape[3])  # B*3,3,3
        volume_feats = volume_feats.unsqueeze(1).repeat(1,6,1,1,1,1)\
                    .reshape(B*6, volume_feats.shape[1], volume_feats.shape[2], volume_feats.shape[3], volume_feats.shape[4])  # B*3,C,H,W,D
        volume_feats = self.VT(volume_feats, K, T)  # B*3,C,H,W

        return volume_feats


class UniPADControlnetUnetWrapper(ModelMixin):
    """As stated in https://github.com/huggingface/accelerate/issues/668, we
    should not use accumulate provided by accelerator, but create a wrapper to
    two modules.
    """

    def __init__(self, UVTRSSL_Wrapper, controlnet_unet) -> None:
        super().__init__()
        # self.controlnet = controlnet
        # self.unet = unet
        # self.weight_dtype = weight_dtype
        # self.unet_in_fp16 = unet_in_fp16

        self.UVTRSSL_Wrapper = UVTRSSL_Wrapper
        self.controlnet_unet = controlnet_unet

    def forward(self, batch, noisy_latents, timesteps, camera_param,
                encoder_hidden_states, encoder_hidden_states_uncond,
                **kwargs):

        # with torch.no_grad():
        controlnet_image = self.UVTRSSL_Wrapper(batch)

        model_pred = self.controlnet_unet(
            noisy_latents, timesteps, camera_param, encoder_hidden_states,
            encoder_hidden_states_uncond, controlnet_image,
            **kwargs,
        )

        return model_pred


class SyntheOccMultiviewRunner(MultiviewRunner):
    def __init__(self, cfg, accelerator, train_set, val_set) -> None:
        super().__init__(cfg, accelerator, train_set, val_set)

        # import pickle5 as pickle
        # with open('/home/yzhu/BEV_Diffusion/train_extracted_bev_feature_5000.pickle', 'rb') as handle:
        #     self.train_extracted_bev_feature = pickle.load(handle)


    def _set_dataset_loader(self):
        # dataset
        collate_fn_param = {
            "tokenizer": self.tokenizer,
            "template": self.cfg.dataset.template,
            "bbox_mode": self.cfg.model.bbox_mode,
            "bbox_view_shared": self.cfg.model.bbox_view_shared,
            "bbox_drop_ratio": self.cfg.runner.bbox_drop_ratio,
            "bbox_add_ratio": self.cfg.runner.bbox_add_ratio,
            "bbox_add_num": self.cfg.runner.bbox_add_num,
        }

        if self.train_dataset is not None:
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset, shuffle=True,
                collate_fn=partial(
                    collate_fn, is_train=True, **collate_fn_param),
                batch_size=self.cfg.runner.train_batch_size,
                num_workers=self.cfg.runner.num_workers, pin_memory=True,
                prefetch_factor=self.cfg.runner.prefetch_factor,
                persistent_workers=True,
            )
        if self.val_dataset is not None:
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset, shuffle=False,
                collate_fn=partial(
                    collate_fn, is_train=False, **collate_fn_param),
                batch_size=self.cfg.runner.validation_batch_size,
                num_workers=self.cfg.runner.num_workers,
                prefetch_factor=self.cfg.runner.prefetch_factor,
            )

    def _init_fixed_models(self, cfg):
        # fmt: off
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="vae")
        self.noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
        # fmt: on

    def _init_trainable_models(self, cfg):
        # fmt: off
        unet = UNet2DConditionModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="unet")
        # fmt: on

        model_cls = load_module(cfg.model.unet_module)
        unet_param = OmegaConf.to_container(self.cfg.model.unet, resolve=True)
        self.unet = model_cls.from_unet_2d_condition(unet, **unet_param)

        # model_cls = load_module(cfg.model.model_module)
        # controlnet_param = OmegaConf.to_container(
        #     self.cfg.model.controlnet, resolve=True)
        # self.controlnet = model_cls.from_unet(unet, **controlnet_param)

        model_cls = load_module(cfg.model.model_module)
        # self.controlnet = model_cls.from_unet(unet, conditioning_channels=256)
        self.controlnet = model_cls.from_unet(unet, conditioning_channels=257)

        # self.UVTRSSL_Wrapper = UVTRSSL_Wrapper()

    def _set_model_trainable_state(self, train=True):
        # set trainable status
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controlnet.train(train)
        self.unet.requires_grad_(False)
        for name, mod in self.unet.trainable_module.items():
            logging.debug(
                f"[MultiviewRunner] set {name} to requires_grad = True")
            mod.requires_grad_(train)

        # self.UVTRSSL_Wrapper.train(train)
        # self.UVTRSSL_Wrapper.UVTRSSL.init_weights()


    def set_optimizer_scheduler(self):
        # optimizer and lr_schedulers
        if self.cfg.runner.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        params_to_optimize = list(self.controlnet.parameters())
        unet_params = self.unet.trainable_parameters
        param_count = smart_param_count(unet_params)
        logging.info(
            f"[MultiviewRunner] add {param_count} params from unet to optimizer.")
        params_to_optimize += unet_params

        # UVTRSSL_Wrapper_params = list(self.UVTRSSL_Wrapper.parameters())
        # params_to_optimize += UVTRSSL_Wrapper_params

        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=self.cfg.runner.learning_rate,
            betas=(self.cfg.runner.adam_beta1, self.cfg.runner.adam_beta2),
            weight_decay=self.cfg.runner.adam_weight_decay,
            eps=self.cfg.runner.adam_epsilon,
        )

        # lr scheduler
        self._calculate_steps()
        # fmt: off
        self.lr_scheduler = get_scheduler(
            self.cfg.runner.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.runner.lr_warmup_steps * self.cfg.runner.gradient_accumulation_steps,
            num_training_steps=self.cfg.runner.max_train_steps * self.cfg.runner.gradient_accumulation_steps,
            num_cycles=self.cfg.runner.lr_num_cycles,
            power=self.cfg.runner.lr_power,
        )
        # fmt: on

    def prepare_device(self):
        # self.controlnet_unet = ControlnetUnetWrapper(self.controlnet, self.unet)
        self.controlnet_unet = SyntheOccControlnetUnetWrapper(self.controlnet, self.unet)
        # self.unipad_controlnet_unet = UniPADControlnetUnetWrapper(self.UVTRSSL_Wrapper, self.controlnet_unet)
        # accelerator
        ddp_modules = (
            self.controlnet_unet,
            # self.unipad_controlnet_unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )
        ddp_modules = self.accelerator.prepare(*ddp_modules)
        (
            self.controlnet_unet,
            # self.unipad_controlnet_unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = ddp_modules

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.cfg.runner.unet_in_fp16 and self.weight_dtype == torch.float16:
            self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
            # move optimized params to fp32. TODO: is this necessary?
            if self.cfg.model.use_fp32_for_unet_trainable:
                for name, mod in self.unet.trainable_module.items():
                    logging.debug(f"[MultiviewRunner] set {name} to fp32")
                    mod.to(dtype=torch.float32)
                    mod._original_forward = mod.forward
                    # autocast intermediate is necessary since others are fp16
                    mod.forward = torch.cuda.amp.autocast(
                        dtype=torch.float16)(mod.forward)
                    # we ensure output is always fp16
                    mod.forward = convert_outputs_to_fp16(mod.forward)
            else:
                raise TypeError(
                    "There is an error/bug in accumulation wrapper, please "
                    "make all trainable param in fp32.")
        controlnet_unet = self.accelerator.unwrap_model(self.controlnet_unet)
        controlnet_unet.weight_dtype = self.weight_dtype
        controlnet_unet.unet_in_fp16 = self.cfg.runner.unet_in_fp16

        # unipad_controlnet_unet = self.accelerator.unwrap_model(self.unipad_controlnet_unet)
        # unipad_controlnet_unet.controlnet_unet.weight_dtype = self.weight_dtype
        # unipad_controlnet_unet.controlnet_unet.unet_in_fp16 = self.cfg.runner.unet_in_fp16

        # with torch.no_grad():
        #     self.accelerator.unwrap_model(self.controlnet).prepare(
        #         self.cfg,
        #         tokenizer=self.tokenizer,
        #         text_encoder=self.text_encoder
        #     )

        # We need to recalculate our total training steps as the size of the
        # training dataloader may have changed.
        self._calculate_steps()

    def _save_model(self, root=None):
        if root is None:
            root = self.cfg.log_root
        # if self.accelerator.is_main_process:
        controlnet = self.accelerator.unwrap_model(self.controlnet)
        controlnet.save_pretrained(
            os.path.join(root, self.cfg.model.controlnet_dir))
        unet = self.accelerator.unwrap_model(self.unet)
        unet.save_pretrained(os.path.join(root, self.cfg.model.unet_dir))

        # UVTRSSL_Wrapper = self.accelerator.unwrap_model(self.UVTRSSL_Wrapper)
        # UVTRSSL_Wrapper.save_pretrained(os.path.join(root, 'UVTRSSL_Wrapper'))

        logging.info(f"Save your model to: {root}")

    def _train_one_stop(self, batch):
        # self.controlnet_unet.train()
        # self.unipad_controlnet_unet.train()
        with self.accelerator.accumulate(self.controlnet_unet):
        # with self.accelerator.accumulate(self.unipad_controlnet_unet):
            N_cam = batch["pixel_values"].shape[1]

            # Convert images to latent space
            latents = self.vae.encode(
                rearrange(batch["pixel_values"], "b n c h w -> (b n) c h w").to(
                    dtype=self.weight_dtype
                )
            ).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            # latents = rearrange(latents, "(b n) c h w -> b n c h w", n=N_cam)

            # embed camera params, in (B, 6, 3, 7), out (B, 6, 189)
            # camera_emb = self._embed_camera(batch["camera_param"])
            camera_param = batch["camera_param"].to(self.weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            # make sure we use same noise for different views, only take the
            # first
            if self.cfg.model.train_with_same_noise:
                noise = repeat(noise[:, 0], "b ... -> b r ...", r=N_cam)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            if self.cfg.model.train_with_same_t:
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
            else:
                timesteps = torch.stack([torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ) for _ in range(N_cam)], dim=1)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self._add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
            encoder_hidden_states_uncond = self.text_encoder(
                batch
                ["uncond_ids"])[0]
            
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(-1, N_cam, -1, -1)
            encoder_hidden_states_uncond = encoder_hidden_states_uncond.unsqueeze(1).expand(-1, N_cam, -1, -1)
            encoder_hidden_states = rearrange(encoder_hidden_states, "b n n_token d -> (b n) n_token d")
            encoder_hidden_states_uncond = rearrange(encoder_hidden_states_uncond, "b n n_token d -> (b n) n_token d")

            # controlnet_image = batch["bev_map_with_aux"].to(
            #     dtype=self.weight_dtype)

            # controlnet_image = torch.load('/home/yzhu/BEV_Diffusion/extracted_bev_feature.pth')
            # controlnet_image = controlnet_image.to(device=latents.device, dtype=self.weight_dtype)
            # controlnet_image = controlnet_image.expand([bsz, -1, -1, -1])

            # controlnet_image_list = []
            # for idx_batch in range(bsz):
            #     cur_sample_idx = batch['meta_data']['metas'][idx_batch].data['token']
            #     controlnet_image_list.append(torch.load(f'./train_extracted_bev_feature/{cur_sample_idx}.bin'))
            # controlnet_image = torch.cat(controlnet_image_list)
            # controlnet_image = controlnet_image.to(device=latents.device, dtype=self.weight_dtype)
            # controlnet_image = rearrange(controlnet_image, "b c v_h h w -> b (c v_h) h w")

            controlnet_image = batch['syntheocc']['ctrl_img'].to(device=latents.device, dtype=self.weight_dtype)
            controlnet_image = rearrange(controlnet_image, "b n c h w -> (b n) c h w")

            model_pred = self.controlnet_unet(
                noisy_latents, timesteps, camera_param, encoder_hidden_states,
                encoder_hidden_states_uncond, controlnet_image,
                **batch['kwargs'],
            )

            # model_pred = self.unipad_controlnet_unet(
            #     batch,
            #     noisy_latents, timesteps, camera_param, encoder_hidden_states,
            #     encoder_hidden_states_uncond,
            #     **batch['kwargs'],
            # )

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(
                    latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

            loss = F.mse_loss(
                model_pred.float(), target.float(), reduction='none')
            loss = loss.mean()

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                params_to_clip = self.controlnet_unet.parameters()
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.cfg.runner.max_grad_norm
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(
                set_to_none=self.cfg.runner.set_grads_to_none)

        return loss
