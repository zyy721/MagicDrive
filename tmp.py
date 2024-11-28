import mmcv

# def main():
#     magicdrive = mmcv.load('data/nuscenes_mmdet3d_2/nuscenes_infos_train.pkl')
#     bevdet = mmcv.load('/home/yzhu/BEVDet/data/nuscenes/bevdetv3-nuscenes_infos_train.pkl')

#     magicdrive_w_bevdet = {}
#     magicdrive_w_bevdet['infos'] = magicdrive['infos']
#     magicdrive_w_bevdet['metadata'] = magicdrive['metadata']
#     for i, cur_magicdrive_w_bevdet in enumerate(magicdrive_w_bevdet['infos']):
#         cur_magicdrive_w_bevdet['bevdet'] = bevdet['infos'][i]
#         if cur_magicdrive_w_bevdet['token'] != bevdet['infos'][i]['token']:
#             print('different')

#     mmcv.dump(magicdrive_w_bevdet, 'data/magicdrive_w_bevdet/magicdrive_w_bevdet_train.pkl')

#     print()

# def main():
#     # magicdrive = mmcv.load('data/nuscenes_mmdet3d_2/nuscenes_infos_train.pkl')
#     # unipad = mmcv.load('/home/yzhu/UniPAD/data/nuscenes/nuscenes_unified_infos_train.pkl')
#     magicdrive = mmcv.load('data/nuscenes_mmdet3d_2/nuscenes_infos_val.pkl')
#     unipad = mmcv.load('/home/yzhu/UniPAD/data/nuscenes/nuscenes_unified_infos_val.pkl')

#     magicdrive_w_unipad = {}
#     magicdrive_w_unipad['infos'] = magicdrive['infos']
#     magicdrive_w_unipad['metadata'] = magicdrive['metadata']
#     for i, cur_magicdrive_w_unipad in enumerate(magicdrive_w_unipad['infos']):
#         cur_magicdrive_w_unipad['unipad'] = unipad['infos'][i]
#         if cur_magicdrive_w_unipad['token'] != unipad['infos'][i]['token']:
#             print('different')

#     # mmcv.dump(magicdrive_w_unipad, 'data/nuscenes_mmdet3d_2/magicdrive_w_unipad_train.pkl')
#     mmcv.dump(magicdrive_w_unipad, 'data/nuscenes_mmdet3d_2/magicdrive_w_unipad_val.pkl')

#     print()

def main():
    magicdrive_w_unipad_syntheocc = mmcv.load('data/nuscenes_mmdet3d_2/magicdrive_w_unipad_syntheocc_train.pkl')

    magicdrive = mmcv.load('data/nuscenes_mmdet3d_2/nuscenes_infos_train.pkl')
    unipad = mmcv.load('/home/yzhu/UniPAD/data/nuscenes/nuscenes_unified_infos_train.pkl')
    syntheocc = mmcv.load('/home/yzhu/SyntheOcc/data/nuscenes/nuscenes_occ_infos_train.pkl')
    # magicdrive = mmcv.load('data/nuscenes_mmdet3d_2/nuscenes_infos_val.pkl')
    # unipad = mmcv.load('/home/yzhu/UniPAD/data/nuscenes/nuscenes_unified_infos_val.pkl')
    # syntheocc = mmcv.load('/home/yzhu/SyntheOcc/data/nuscenes/nuscenes_occ_infos_val.pkl')

    magicdrive_w_unipad = {}
    magicdrive_w_unipad['infos'] = magicdrive['infos']
    magicdrive_w_unipad['metadata'] = magicdrive['metadata']
    for i, cur_magicdrive_w_unipad in enumerate(magicdrive_w_unipad['infos']):
        cur_magicdrive_w_unipad['unipad'] = unipad['infos'][i]
        if cur_magicdrive_w_unipad['token'] != unipad['infos'][i]['token']:
            print('different')
        cur_magicdrive_w_unipad['syntheocc'] = syntheocc['infos'][i]
        if cur_magicdrive_w_unipad['token'] != syntheocc['infos'][i]['token']:
            print('different')

    # mmcv.dump(magicdrive_w_unipad, 'data/nuscenes_mmdet3d_2/magicdrive_w_unipad_train.pkl')
    # mmcv.dump(magicdrive_w_unipad, 'data/nuscenes_mmdet3d_2/magicdrive_w_unipad_val.pkl')

    # mmcv.dump(magicdrive_w_unipad, 'data/nuscenes_mmdet3d_2/magicdrive_w_unipad_syntheocc_train.pkl')
    mmcv.dump(magicdrive_w_unipad, 'data/nuscenes_mmdet3d_2/magicdrive_w_unipad_syntheocc_val.pkl')

    print()


if __name__ == "__main__":
    main()