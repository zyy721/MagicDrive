from typing import Any, Dict

import numpy as np
from pyquaternion import Quaternion

from mmdet.datasets import DATASETS

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset


@DATASETS.register_module()
class NuScenesDatasetM(NuScenesDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        dataset_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """

    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
    ) -> None:
        self.force_all_boxes = force_all_boxes
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            dataset_root=dataset_root,
            object_classes=object_classes,
            map_classes=map_classes,
            load_interval=load_interval,
            with_velocity=with_velocity,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            eval_version=eval_version,
            use_valid_flag=use_valid_flag,
        )

        # self.use_valid_flag = True
        # self.img_info_prototype = 'bevdet'

        self.return_gt_info = False


    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """

        # idx = 2000

        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag and not self.force_all_boxes:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]

        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"],
            location=info["location"],
        )
        add_key = [
            "description",
            "timeofday",
            "visibility",
            "flip_gt",
        ]
        for key in add_key:
            if key in info:
                data[key] = info[key]

        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = Quaternion(
            info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]
        data["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(
            info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego

        if self.modality["use_camera"]:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []

            for _, camera_info in info["cams"].items():
                data["image_paths"].append(camera_info["data_path"])

                # lidar to camera transform
                lidar2camera_r = np.linalg.inv(
                    camera_info["sensor2lidar_rotation"])
                lidar2camera_t = (
                    camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                data["lidar2camera"].append(lidar2camera_rt.T)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
                data["camera_intrinsics"].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt.T
                data["lidar2image"].append(lidar2image)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    camera_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                data["camera2ego"].append(camera2ego)

                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
                camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                data["camera2lidar"].append(camera2lidar)

        annos, mask = self.get_ann_info(index)
        if "visibility" in data:
            data["visibility"] = data["visibility"][mask]
        data["ann_info"] = annos


        # data["bevdet"] = self.bevdet_get_data_info(info["bevdet"])
        data["unipad"] = self.unipad_get_data_info(info["unipad"])

        return data

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.force_all_boxes:
            mask = np.ones_like(info["valid_flag"])
        elif self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results, mask




    # def unipad_get_data_info(self, index):
    def unipad_get_data_info(self, info):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        # info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
        )
        if self.return_gt_info:
            input_dict["info"] = info

        # convert file path to nori and process sweep number in loading function
        if self.modality["use_lidar"]:
            input_dict["sweeps"] = info["sweeps"]

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            # add lidar2img matrix
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)
                lidar2cam_rts.append(lidar2cam_rt.T)
                cam_intrinsics.append(viewpad)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    lidar2cam=lidar2cam_rts,
                    cam_intrinsic=cam_intrinsics,
                )
            )
            # use cam sweeps
            if "cam_sweep_num" in self.modality:
                cam_sweeps_paths = []
                cam_sweeps_id = []
                cam_sweeps_time = []
                lidar2img_sweeps_rts = []
                # add lidar2img matrix
                lidar2cam_sweeps_rts = []
                cam_sweeps_intrinsics = []
                cam_sweep_num = self.modality["cam_sweep_num"]
                for cam_idx, (cam_type, cam_infos) in enumerate(
                    info["cam_sweeps_info"].items()
                ):
                    # avoid none sweep
                    if len(cam_infos) == 0:
                        cam_sweeps = [
                            image_paths[cam_idx] for _ in range(cam_sweep_num)
                        ]
                        cam_ids = [0 for _ in range(cam_sweep_num)]
                        cam_time = [0.0 for _ in range(cam_sweep_num)]
                        lidar2img_sweeps = [
                            lidar2img_rts[cam_idx] for _ in range(cam_sweep_num)
                        ]
                        lidar2cam_sweeps = [
                            lidar2cam_rts[cam_idx] for _ in range(cam_sweep_num)
                        ]
                        intrinsics_sweeps = [
                            cam_intrinsics[cam_idx] for _ in range(cam_sweep_num)
                        ]
                    else:
                        cam_sweeps = []
                        cam_ids = []
                        cam_time = []
                        lidar2img_sweeps = []
                        lidar2cam_sweeps = []
                        intrinsics_sweeps = []
                        for sweep_id, sweep_info in enumerate(
                            cam_infos[:cam_sweep_num]
                        ):
                            cam_sweeps.append(sweep_info["data_path"])
                            cam_ids.append(sweep_id)
                            cam_time.append(
                                input_dict["timestamp"] - sweep_info["timestamp"] / 1e6
                            )
                            # obtain lidar to image transformation matrix
                            lidar2cam_r = np.linalg.inv(
                                sweep_info["sensor2lidar_rotation"]
                            )
                            lidar2cam_t = (
                                sweep_info["sensor2lidar_translation"] @ lidar2cam_r.T
                            )
                            lidar2cam_rt = np.eye(4)
                            lidar2cam_rt[:3, :3] = lidar2cam_r.T
                            lidar2cam_rt[3, :3] = -lidar2cam_t
                            intrinsic = sweep_info["cam_intrinsic"]
                            viewpad = np.eye(4)
                            viewpad[
                                : intrinsic.shape[0], : intrinsic.shape[1]
                            ] = intrinsic
                            lidar2img_rt = viewpad @ lidar2cam_rt.T
                            lidar2img_sweeps.append(lidar2img_rt)
                            lidar2cam_sweeps.append(lidar2cam_rt.T)
                            intrinsics_sweeps.append(viewpad)

                    # pad empty sweep with the last frame
                    if len(cam_sweeps) < cam_sweep_num:
                        cam_req = cam_sweep_num - len(cam_infos)
                        cam_ids = cam_ids + [cam_ids[-1] for _ in range(cam_req)]
                        cam_time = cam_time + [cam_time[-1] for _ in range(cam_req)]
                        cam_sweeps = cam_sweeps + [
                            cam_sweeps[-1] for _ in range(cam_req)
                        ]
                        lidar2img_sweeps = lidar2img_sweeps + [
                            lidar2img_sweeps[-1] for _ in range(cam_req)
                        ]
                        lidar2cam_sweeps = lidar2cam_sweeps + [
                            lidar2cam_sweeps[-1] for _ in range(cam_req)
                        ]
                        intrinsics_sweeps = intrinsics_sweeps + [
                            intrinsics_sweeps[-1] for _ in range(cam_req)
                        ]

                    # align to start time
                    cam_time = [_time - cam_time[0] for _time in cam_time]
                    # sweep id from 0->prev 1->prev 2
                    if cam_sweeps[0] == image_paths[cam_idx]:
                        cam_sweeps_paths.append(cam_sweeps[1:cam_sweep_num])
                        cam_sweeps_id.append(cam_ids[1:cam_sweep_num])
                        cam_sweeps_time.append(cam_time[1:cam_sweep_num])
                        lidar2img_sweeps_rts.append(lidar2img_sweeps[1:cam_sweep_num])
                        lidar2cam_sweeps_rts.append(lidar2cam_sweeps[1:cam_sweep_num])
                        cam_sweeps_intrinsics.append(intrinsics_sweeps[1:cam_sweep_num])
                    else:
                        raise ValueError

                if "cam_sweep_list" in self.modality:
                    sweep_list = self.modality["cam_sweep_list"]
                    for cam_idx in range(len(cam_sweeps_paths)):
                        cam_sweeps_paths[cam_idx] = [
                            cam_sweeps_paths[cam_idx][i] for i in sweep_list
                        ]
                        cam_sweeps_id[cam_idx] = [
                            cam_sweeps_id[cam_idx][i] for i in sweep_list
                        ]
                        cam_sweeps_time[cam_idx] = [
                            cam_sweeps_time[cam_idx][i] for i in sweep_list
                        ]
                        cam_sweeps_intrinsics[cam_idx] = [
                            cam_sweeps_intrinsics[cam_idx][i] for i in sweep_list
                        ]

                input_dict.update(
                    dict(
                        cam_sweeps_paths=cam_sweeps_paths,
                        cam_sweeps_id=cam_sweeps_id,
                        cam_sweeps_time=cam_sweeps_time,
                        lidar2img_sweeps=lidar2img_sweeps_rts,
                        lidar2cam_sweeps=lidar2cam_sweeps_rts,
                        cam_sweeps_intrinsics=cam_sweeps_intrinsics,
                    )
                )

        # self.test_mode = False
        if not self.test_mode:
            # annos = self.get_ann_info(index)
            annos = self.unipad_get_ann_info(info)
            input_dict["ann_info"] = annos

        return input_dict

    def unipad_get_ann_info(self, info):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # info = self.data_infos[index]
        # filter out bbox containing no points
        # if self.use_valid_flag:
        #     mask = info["valid_flag"]
        # else:
        #     mask = info["num_lidar_pts"] > 0
        mask = info["valid_flag"]
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, gt_names=gt_names_3d
        )
        return anns_results







    # def bevdet_get_data_info(self, index):
    def bevdet_get_data_info(self, info):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        # info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )
        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']
        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(
                        cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.
                            shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))

                if not self.test_mode:
                    # annos = self.get_ann_info(index)
                    annos = self.bevdet_get_ann_info(info)
                    input_dict['ann_info'] = annos
            else:
                assert 'bevdet' in self.img_info_prototype
                input_dict.update(dict(curr=info))
                # if '4d' in self.img_info_prototype:
                #     info_adj_list = self.get_adj_info(info, index)
                #     input_dict.update(dict(adjacent=info_adj_list))
        return input_dict


    def bevdet_get_adj_info(self, info, index):
        info_adj_list = []
        adj_id_list = list(range(*self.multi_adj_frame_id_cfg))
        if self.stereo:
            assert self.multi_adj_frame_id_cfg[0] == 1
            assert self.multi_adj_frame_id_cfg[2] == 1
            adj_id_list.append(self.multi_adj_frame_id_cfg[1])
        for select_id in adj_id_list:
            select_id = max(index - select_id, 0)
            if not self.data_infos[select_id]['scene_token'] == info[
                    'scene_token']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos[select_id])
        return info_adj_list

    # def bevdet_get_ann_info(self, index):
    def bevdet_get_ann_info(self, info):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # info = self.data_infos[index]
        # filter out bbox containing no points
        # if self.use_valid_flag:
        #     mask = info['valid_flag']
        # else:
        #     mask = info['num_lidar_pts'] > 0
        mask = info['valid_flag']
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results