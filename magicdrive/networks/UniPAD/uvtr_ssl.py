from re import I
from collections import OrderedDict
import torch
import torch.nn as nn

from mmcv.cnn import Conv2d
from mmcv.runner import BaseModule, force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.core import multi_apply
# from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from pdb import set_trace
import pickle
import numpy as np
# from ..utils.uni3d_voxelpooldepth import DepthNet

from magicdrive.networks.UniPAD.render_head import RenderHead
from magicdrive.networks.UniPAD.backbones import MaskConvNeXt
from magicdrive.networks.UniPAD.necks import CustomFPN
from magicdrive.networks.UniPAD.utils.uni3d_voxelpooldepth import DepthNet



# @DETECTORS.register_module()
# class UVTRSSL(MVXTwoStageDetector):
class UVTRSSL(BaseModule):
# class UVTRSSL(nn.Module):
    """UVTR."""

    def __init__(
        self,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        depth_head=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        # super(UVTRSSL, self).__init__(
        #     pts_voxel_layer,
        #     pts_voxel_encoder,
        #     pts_middle_encoder,
        #     pts_fusion_layer,
        #     img_backbone,
        #     pts_backbone,
        #     img_neck,
        #     pts_neck,
        #     pts_bbox_head,
        #     img_roi_head,
        #     img_rpn_head,
        #     train_cfg,
        #     test_cfg,
        #     pretrained,
        # )
        super(UVTRSSL, self).__init__()

        # cfg = Config.fromfile('magicdrive/networks/UniPAD/configs/uvtr_cam_vs0.075_pretrain.py')
        # img_backbone = cfg.model['img_backbone']
        # img_neck = cfg.model['img_neck']
        # pts_bbox_head = cfg.model['pts_bbox_head']


        # # If point cloud range is changed, the models should also change their point
        # # cloud range accordingly
        # point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        # unified_voxel_size = [0.6, 0.6, 1.6]  # [180, 180, 5]
        # frustum_range = [0, 0, 1.0, 1600, 928, 60.0]
        # frustum_size = [32.0, 32.0, 0.5]

        # unified_voxel_shape = [
        #     int((point_cloud_range[3] - point_cloud_range[0]) / unified_voxel_size[0]),
        #     int((point_cloud_range[4] - point_cloud_range[1]) / unified_voxel_size[1]),
        #     int((point_cloud_range[5] - point_cloud_range[2]) / unified_voxel_size[2]),
        # ]


        # UVTRSSL_dict=dict(
        #     img_backbone=dict(
        #         # type="MaskConvNeXt",
        #         arch="small",
        #         drop_path_rate=0.2,
        #         out_indices=(3),
        #         norm_out=True,
        #         frozen_stages=1,
        #         init_cfg=dict(
        #             type="Pretrained",
        #             checkpoint="data/ckpts/convnextS_1kpretrained_official_style.pth",
        #         ),
        #         mae_cfg=dict(
        #             downsample_scale=32, downsample_dim=768, mask_ratio=0.3, learnable=False
        #         ),
        #     ),
        #     img_neck=dict(
        #         # type="CustomFPN",
        #         in_channels=[768],
        #         out_channels=256,
        #         num_outs=1,
        #         start_level=0,
        #         out_ids=[0],
        #     ),
        #     depth_head=dict(type="ComplexDepth", use_dcn=False, aspp_mid_channels=96),
        #     pts_bbox_head=dict(
        #         type="RenderHead",
        #         fp16_enabled=False,
        #         in_channels=256,
        #         unified_voxel_size=unified_voxel_size,
        #         unified_voxel_shape=unified_voxel_shape,
        #         pc_range=point_cloud_range,
        #         view_cfg=dict(
        #             type="Uni3DVoxelPoolDepth",
        #             frustum_range=frustum_range,
        #             frustum_size=frustum_size,
        #             num_convs=0,
        #             keep_sweep_dim=False,
        #             fp16_enabled=False,
        #             loss_cfg=dict(close_radius=3.0, depth_loss_weights=[0.0]),
        #         ),
        #         render_conv_cfg=dict(out_channels=32, kernel_size=3, padding=1),
        #         ray_sampler_cfg=dict(
        #             close_radius=3.0,
        #             only_img_mask=False,
        #             only_point_mask=False,
        #             replace_sample=False,
        #             point_nsample=512,
        #             point_ratio=0.5,
        #             pixel_interval=4,
        #             sky_region=0.4,
        #             merged_nsample=512,
        #         ),
        #         render_ssl_cfg=dict(
        #             type="NeuSModel",
        #             norm_scene=True,
        #             field_cfg=dict(
        #                 type="SDFField",
        #                 sdf_decoder_cfg=dict(
        #                     in_dim=32, out_dim=16 + 1, hidden_size=16, n_blocks=5
        #                 ),
        #                 rgb_decoder_cfg=dict(
        #                     in_dim=32 + 16 + 3 + 3, out_dim=3, hidden_size=16, n_blocks=3
        #                 ),
        #                 interpolate_cfg=dict(type="SmoothSampler", padding_mode="zeros"),
        #                 beta_init=0.3,
        #             ),
        #             collider_cfg=dict(type="AABBBoxCollider", near_plane=1.0),
        #             sampler_cfg=dict(
        #                 type="NeuSSampler",
        #                 initial_sampler="UniformSampler",
        #                 num_samples=72,
        #                 num_samples_importance=24,
        #                 num_upsample_steps=1,
        #                 train_stratified=True,
        #                 single_jitter=True,
        #             ),
        #             loss_cfg=dict(
        #                 sensor_depth_truncation=0.1,
        #                 sparse_points_sdf_supervised=False,
        #                 weights=dict(
        #                     depth_loss=10.0,
        #                     rgb_loss=10.0,
        #                 ),
        #             ),
        #         ),
        #     ),
        # )

        # img_backbone = UVTRSSL_dict['img_backbone']
        # img_neck = UVTRSSL_dict['img_neck']
        # pts_bbox_head = UVTRSSL_dict['pts_bbox_head']

        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)

            args_pts_bbox_head = pts_bbox_head.copy()
            obj_type_args_pts_bbox_head = args_pts_bbox_head.pop('type')

            self.pts_bbox_head = RenderHead(**args_pts_bbox_head)
        if img_backbone:

            args_img_backbone = img_backbone.copy()
            obj_type_args_img_backbone = args_img_backbone.pop('type')

            self.img_backbone = MaskConvNeXt(**args_img_backbone)
        if img_neck is not None:

            args_img_neck = img_neck.copy()
            obj_type_args_img_neck = args_img_neck.pop('type')

            self.img_neck = CustomFPN(**args_img_neck)

        if self.with_img_backbone:
            in_channels = self.img_neck.out_channels
            out_channels = self.pts_bbox_head.in_channels
            if isinstance(in_channels, list):
                in_channels = in_channels[0]
            self.input_proj = Conv2d(in_channels, out_channels, kernel_size=1)
            if depth_head is not None:
                depth_dim = self.pts_bbox_head.view_trans.depth_dim
                dhead_type = depth_head.pop("type", "SimpleDepth")
                if dhead_type == "SimpleDepth":
                    self.depth_net = Conv2d(out_channels, depth_dim, kernel_size=1)
                else:
                    self.depth_net = DepthNet(
                        out_channels, out_channels, depth_dim, **depth_head
                    )
            self.depth_head = depth_head

        if pts_middle_encoder:
            self.pts_fp16 = (
                True if hasattr(self.pts_middle_encoder, "fp16_enabled") else False
            )

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_depth_head(self):
        """bool: Whether the detector has a depth head."""
        return hasattr(self, "depth_head") and self.depth_head is not None

    @force_fp32()
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.pts_voxel_encoder or pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        if not self.pts_fp16:
            voxel_features = voxel_features.float()

        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        if self.with_pts_backbone:
            x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if img is not None:
            B = img.size(0)
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            img_feat = self.input_proj(img_feat)
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=("img"))
    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        if hasattr(self, "img_backbone"):
            img_feats = self.extract_img_feat(img, img_metas)
            img_depth = self.pred_depth(
                img=img, img_metas=img_metas, img_feats=img_feats
            )
        else:
            img_feats, img_depth = None, None

        if hasattr(self, "pts_voxel_encoder"):
            pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        else:
            pts_feats = None

        return pts_feats, img_feats, img_depth

    @auto_fp16(apply_to=("img"))
    def pred_depth(self, img, img_metas, img_feats=None):
        if img_feats is None or not self.with_depth_head:
            return None
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        depth = []
        for _feat in img_feats:
            _depth = self.depth_net(_feat.view(-1, *_feat.shape[-3:]))
            _depth = _depth.softmax(dim=1)
            depth.append(_depth)
        return depth

    # @force_fp32(apply_to=("pts_feats", "img_feats"))
    # def forward_pts_train(
    #     self, pts_feats, img_feats, points, img, img_metas, img_depth
    # ):
    #     """Forward function for point cloud branch.
    #     Args:
    #         pts_feats (list[torch.Tensor]): Features of point cloud branch
    #         gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
    #             boxes for each sample.
    #         gt_labels_3d (list[torch.Tensor]): Ground truth labels for
    #             boxes of each sampole
    #         img_metas (list[dict]): Meta information of samples.
    #         gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
    #             boxes to be ignored. Defaults to None.
    #     Returns:
    #         dict: Losses of each branch.
    #     """
    #     batch_rays = self.pts_bbox_head.sample_rays(points, img, img_metas)
    #     # batch_rays = 0
    #     out_dict = self.pts_bbox_head(
    #         pts_feats, img_feats, batch_rays, img_metas, img_depth
    #     )
    #     losses = self.pts_bbox_head.loss(out_dict, batch_rays)
    #     if self.with_depth_head and hasattr(self.pts_bbox_head.view_trans, "loss"):
    #         losses.update(
    #             self.pts_bbox_head.view_trans.loss(img_depth, points, img, img_metas)
    #         )
    #     return losses



    @force_fp32(apply_to=("pts_feats", "img_feats"))
    def forward_pts_train(
        self, pts_feats, img_feats, points, img, img_metas, img_depth
    ):
        # batch_rays = self.pts_bbox_head.sample_rays(points, img, img_metas)
        batch_rays = 0
        uni_feats = self.pts_bbox_head(
            pts_feats, img_feats, batch_rays, img_metas, img_depth
        )
        return uni_feats


    @force_fp32(apply_to=("img", "points"))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    # def forward_train(self, points=None, img_metas=None, img=None):
    #     """Forward training function.
    #     Args:
    #         points (list[torch.Tensor], optional): Points of each sample.
    #             Defaults to None.
    #         img_metas (list[dict], optional): Meta information of each sample.
    #             Defaults to None.
    #         gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
    #             Ground truth 3D boxes. Defaults to None.
    #         gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
    #             of 3D boxes. Defaults to None.
    #         gt_labels (list[torch.Tensor], optional): Ground truth labels
    #             of 2D boxes in images. Defaults to None.
    #         gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
    #             images. Defaults to None.
    #         img (torch.Tensor optional): Images of each sample with shape
    #             (N, C, H, W). Defaults to None.
    #         proposals ([list[torch.Tensor], optional): Predicted proposals
    #             used for training Fast RCNN. Defaults to None.
    #         gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
    #             2D boxes in images to be ignored. Defaults to None.
    #     Returns:
    #         dict: Losses of different branches.
    #     """
    #     pts_feats, img_feats, img_depth = self.extract_feat(
    #         points=points, img=img, img_metas=img_metas
    #     )
    #     losses = dict()
    #     losses_pts = self.forward_pts_train(
    #         pts_feats, img_feats, points, img, img_metas, img_depth
    #     )
    #     losses.update(losses_pts)
    #     return losses


    def forward_train(self, points=None, img_metas=None, img=None):
        pts_feats, img_feats, img_depth = self.extract_feat(
            points=points, img=img, img_metas=img_metas
        )
        uni_feats = self.forward_pts_train(
            pts_feats, img_feats, points, img, img_metas, img_depth
        )
        return uni_feats


    def forward_test(self, img_metas, points=None, img=None, **kwargs):
        num_augs = len(img_metas)
        if points is not None:
            if num_augs != len(points):
                raise ValueError(
                    "num of augmentations ({}) != num of image meta ({})".format(
                        len(points), len(img_metas)
                    )
                )

        assert num_augs == 1
        if not isinstance(img_metas[0], list):
            img_metas = [img_metas]
        if not isinstance(img, list):
            img = [img]
        results = self.simple_test(img_metas[0], points, img[0])

        return results

    def simple_test(self, img_metas, points=None, img=None):
        """Test function without augmentaiton."""
        pts_feats, img_feats, img_depth = self.extract_feat(
            points=points, img=img, img_metas=img_metas
        )

        batch_rays = []
        uni_feats = self.pts_bbox_head(
            pts_feats, img_feats, batch_rays, img_metas, img_depth
        )

        return uni_feats

        # batch_rays = self.pts_bbox_head.sample_rays_test(points, img, img_metas)
        # results = self.pts_bbox_head(
        #     pts_feats, img_feats, batch_rays, img_metas, img_depth
        # )
        # with open("outputs/{}.pkl".format(img_metas[0]["sample_idx"]), "wb") as f:
        #     H, W = img_metas[0]["img_shape"][0][0], img_metas[0]["img_shape"][0][1]
        #     num_cam = len(img_metas[0]["img_shape"])
        #     l = 2
        #     # init_weights = results[0]["vis_weights"]
        #     # init_weights = init_weights.reshape(num_cam, -1, *init_weights.shape[1:])
        #     # init_sampled_points = results[0]["vis_sampled_points"]
        #     # init_sampled_points = init_sampled_points.reshape(
        #     #     num_cam, -1, *init_sampled_points.shape[1:]
        #     # )
        #     # pts_idx = np.random.randint(
        #     #     0, high=init_sampled_points.shape[1], size=(256,), dtype=int
        #     # )
        #     # init_weights = init_weights[:, pts_idx]
        #     # init_sampled_points = init_sampled_points[:, pts_idx]
        #     pickle.dump(
        #         {
        #             "render_rgb": results[0]["rgb"]
        #             .reshape(num_cam, H // l, W // l, 3)
        #             .detach()
        #             .cpu()
        #             .numpy(),
        #             "render_depth": results[0]["depth"]
        #             .reshape(num_cam, H // l, W // l, 1)
        #             .detach()
        #             .cpu()
        #             .numpy(),
        #             "rgb": batch_rays[0]["rgb"].detach().cpu().numpy(),
        #             # "scaled_points": results[0]["scaled_points"].detach().cpu().numpy(),
        #             # "points": points[0].detach().cpu().numpy(),
        #             # "lidar2img": np.asarray(img_metas[0]["lidar2img"])[
        #             #     :, 0
        #             # ],  # (N, 4, 4)
        #             # 'weights': results[0]['vis_weights'].detach().cpu().numpy(),
        #             # 'sampled_points': results[0]['vis_sampled_points'].detach().cpu().numpy(),
        #             # "init_weights": init_weights.detach().cpu().numpy(),
        #             # "init_sampled_points": init_sampled_points.detach().cpu().numpy(),
        #         },
        #         f,
        #     )
        #     print("save to outputs/{}.pkl".format(img_metas[0]["sample_idx"]))
        # set_trace()
        # return results

    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        if points is None:
            points = [None] * len(img_metas)
        pts_feats, img_feats, img_depths = multi_apply(
            self.extract_feat, points, imgs, img_metas
        )
        return pts_feats, img_feats, img_depths
