# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import math
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import trimesh
import json

import smpl
from lib.scanimate.utils.config import load_config
from lib.scanimate.utils.geo_util import compute_normal_v
from lib.scanimate.utils.mesh_util import reconstruction, save_obj_mesh, replace_hands_feet, replace_hands_feet_mesh
from lib.scanimate.utils.net_util import batch_rod2quat, homogenize, load_network, get_posemap, compute_knn_feat
from lib.scanimate.model.IGRSDFNet import IGRSDFNet
from lib.scanimate.model.LBSNet import LBSNet
from lib.scanimate.data.MVPDataset import MVPDataset_scan

import logging

logging.basicConfig(level=logging.DEBUG)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def gen_mesh1(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin,
              model, smpl_vitruvian, train_data_loader,
              cuda, name='', reference_body_v=None, every_n_frame=10):
    dataset = train_data_loader.dataset
    smpl_face = torch.LongTensor(model.faces[:, [0, 2, 1]].astype(np.int32))[None].to(cuda)

    def process(data, idx=0):
        frame_names = data['frame_name']
        betas = data['betas'][None].to(device=cuda)
        body_pose = data['body_pose'][None].to(device=cuda)
        scan_posed = data['scan_posed'][None].to(device=cuda)
        transl = data['transl'][None].to(device=cuda)
        f_ids = torch.LongTensor([data['f_id']]).to(device=cuda)
        faces = data['faces'].numpy()
        global_orient = body_pose[:, :3]
        body_pose = body_pose[:, 3:]

        if not reference_body_v == None:
            output = model(betas=betas, body_pose=body_pose, global_orient=0 * global_orient, transl=0 * transl,
                           return_verts=True, custom_out=True,
                           body_neutral_v=reference_body_v.expand(body_pose.shape[0], -1, -1))
        else:
            output = model(betas=betas, body_pose=body_pose, global_orient=0 * global_orient, transl=0 * transl,
                           return_verts=True, custom_out=True)
        smpl_posed_joints = output.joints
        rootT = model.get_root_T(global_orient, transl, smpl_posed_joints[:, 0:1, :])

        smpl_neutral = output.v_shaped
        smpl_cano = output.v_posed
        smpl_posed = output.vertices.contiguous()
        bmax = smpl_posed.max(1)[0]
        bmin = smpl_posed.min(1)[0]
        offset = 0.2 * (bmax - bmin)
        bmax += offset
        bmin -= offset
        jT = output.joint_transform[:, :24]
        smpl_n_posed = compute_normal_v(smpl_posed, smpl_face.expand(smpl_posed.shape[0], -1, -1))

        scan_posed = torch.einsum('bst,bvt->bsv', torch.inverse(rootT), homogenize(scan_posed))[:, :3,
                     :]  # remove root transform

        if name == '_pt2':
            # save_obj_mesh('%s/%ssmpl_posed%s%s.obj' % (result_dir, frame_names, str(idx).zfill(4), name), smpl_posed[0].cpu().numpy(), model.faces[:,[0,2,1]])
            # save_obj_mesh('%s/%ssmpl_cano%d%s.obj' % (result_dir, frame_names, idx, name), smpl_cano[0].cpu().numpy(), model.faces[:,[0,2,1]])
            save_obj_mesh('%s/%sscan_posed_gth%s%s.obj' % (result_dir, frame_names, str(idx).zfill(4), name),
                          scan_posed[0].t().cpu().numpy(), faces)

        if inv_skin_net.opt['g_dim'] > 0:
            lat = lat_vecs_inv_skin(f_ids)  # (B, Z)
            inv_skin_net.set_global_feat(lat)
        feat3d_posed = None
        res_scan_p = inv_skin_net(feat3d_posed, scan_posed, jT=jT, bmin=bmin[:, :, None], bmax=bmax[:, :, None])
        pred_scan_cano = res_scan_p['pred_smpl_cano'].permute(0, 2, 1)

        # for i in range(24):
        #     c_lbs = scalar_to_color(res_scan_p['pred_lbs_smpl_posed'][0,i,:].cpu().numpy(),min=0,max=1)
        # save_obj_mesh_with_color('%s/scan_lbs%d-%d%s.obj'%(result_dir, idx, i, name), scan_posed[0].t().cpu().numpy(), faces, c_lbs)

        res_smpl_p = inv_skin_net(feat3d_posed, smpl_posed.permute(0, 2, 1), jT=jT, bmin=bmin[:, :, None],
                                  bmax=bmax[:, :, None])
        if name == '_pt3':
            scan_faces, scan_mask = dataset.get_raw_scan_face_and_mask(frame_id=f_ids[0].cpu().numpy())
            valid_scan_faces = scan_faces[scan_mask, :]
            pred_scan_cano_mesh = trimesh.Trimesh(vertices=pred_scan_cano[0].cpu().numpy(),
                                                  faces=valid_scan_faces[:, [0, 2, 1]], process=False)
            pred_scan_cano_mesh.export('%s/canon.obj' % result_dir)
            # TODO
            # from lib.common.mesh_util import build_mesh_by_poisson
            #  new_vertices, new_faces = build_mesh_by_poisson(pred_scan_cano_mesh.vertices, pred_scan_cano_mesh.faces, 40000)
            # save_obj_mesh('%s/canon.obj' % result_dir, new_vertices, new_faces)

    if name == '_pt3':
        logging.info("Outputing samples of canonicalization results...")
        with torch.no_grad():
            for i in tqdm(range(len(dataset))):
                if not i % every_n_frame == 0:
                    continue
                data = dataset[i]
                process(data, i)


def pretrain_skinning_net(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin,
                          model, smpl_vitruvian, gt_lbs_smpl,
                          train_data_loader, test_data_loader,
                          cuda, reference_body_v=None):
    optimizer_lbs_c = torch.optim.Adam(fwd_skin_net.parameters(), lr=opt['training']['lr_pt1'])
    optimizer_lbs_p = torch.optim.Adam([
        {
            "params": inv_skin_net.parameters(),
            "lr": opt['training']['lr_pt1'],
        },
        {
            "params": lat_vecs_inv_skin.parameters(),
            "lr": opt['training']['lr_pt1'],
        },
    ])
    smpl_face = torch.LongTensor(model.faces[:, [0, 2, 1]].astype(np.int32))[None].to(cuda)

    n_iter = 0
    for epoch in range(opt['training']['num_epoch_pt1']):
        fwd_skin_net.train()
        inv_skin_net.train()

        if epoch % opt['training']['resample_every_n_epoch'] == 0:
            train_data_loader.dataset.resample_flag = True
        else:
            train_data_loader.dataset.resample_flag = False

        if epoch == opt['training']['num_epoch_pt1'] // 2 or epoch == 3 * (opt['training']['num_epoch_pt1'] // 4):
            for j, _ in enumerate(optimizer_lbs_c.param_groups):
                optimizer_lbs_c.param_groups[j]['lr'] *= 0.1
            for j, _ in enumerate(optimizer_lbs_p.param_groups):
                optimizer_lbs_p.param_groups[j]['lr'] *= 0.1
        for train_idx, train_data in enumerate(train_data_loader):
            betas = train_data['betas'].to(device=cuda)
            body_pose = train_data['body_pose'].to(device=cuda)
            scan_posed = train_data['scan_cano_uni'].to(device=cuda)
            scan_tri = train_data['scan_tri_posed'].to(device=cuda)
            transl = train_data['transl'].to(device=cuda)
            f_ids = train_data['f_id'].to(device=cuda)
            smpl_data = train_data['smpl_data']
            global_orient = body_pose[:, :3]
            body_pose = body_pose[:, 3:]

            smpl_neutral = smpl_data['smpl_neutral'].cuda()
            smpl_cano = smpl_data['smpl_cano'].cuda()
            smpl_posed = smpl_data['smpl_posed'].cuda()
            smpl_n_posed = smpl_data['smpl_n_posed'].cuda()
            bmax = smpl_data['bmax'].cuda()
            bmin = smpl_data['bmin'].cuda()
            jT = smpl_data['jT'].cuda()
            inv_rootT = smpl_data['inv_rootT'].cuda()

            # Get rid of global rotation from posed scans
            scan_posed = torch.einsum('bst,bvt->bvs', inv_rootT, homogenize(scan_posed))[:, :, :3]

            reference_lbs_scan = compute_knn_feat(scan_posed, smpl_posed,
                                                  gt_lbs_smpl.expand(scan_posed.shape[0], -1, -1).permute(0, 2, 1))[:,
                                 :, 0].permute(0, 2, 1)
            scan_posed = scan_posed.permute(0, 2, 1)

            if opt['model']['inv_skin_net']['g_dim'] > 0:
                lat = lat_vecs_inv_skin(f_ids)  # (B, Z)
                inv_skin_net.set_global_feat(lat)

            feat3d_posed = None
            res_lbs_p, err_lbs_p, err_dict = inv_skin_net(feat3d_posed, smpl_posed.permute(0, 2, 1), gt_lbs_smpl,
                                                          scan=scan_posed, reference_lbs_scan=None, jT=jT,
                                                          bmin=bmin[:, :, None],
                                                          bmax=bmax[:, :, None])  # jT=jT, v_tri=scan_tri,

            feat3d_cano = None
            res_lbs_c, err_lbs_c, err_dict_lbs_c = fwd_skin_net(feat3d_cano, smpl_cano, gt_lbs_smpl,
                                                                scan=res_lbs_p['pred_scan_cano'].detach(),
                                                                reference_lbs_scan=reference_lbs_scan)  # , jT=jT, res_posed=res_lbs_p)

            # Back propagation
            err_dict.update(err_dict_lbs_c)
            err_dict['All-inv'] = err_lbs_p.item()
            err_dict['All-lbs'] = err_lbs_c.item()

            optimizer_lbs_p.zero_grad()
            err_lbs_p.backward()
            optimizer_lbs_p.step()

            optimizer_lbs_c.zero_grad()
            err_lbs_c.backward()
            optimizer_lbs_c.step()

            if n_iter % opt['training']['freq_plot'] == 0:
                err_txt = ''.join(['{}: {:.3f} '.format(k, v) for k, v in err_dict.items()])
                print('[%03d/%03d]:[%04d/%04d] %s' % (
                    epoch, opt['training']['num_epoch_pt1'], train_idx, len(train_data_loader), err_txt))
            n_iter += 1

    train_data_loader.dataset.is_train = False
    gen_mesh1(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model, smpl_vitruvian, train_data_loader,
              cuda, '_pt1', reference_body_v=reference_body_v)
    train_data_loader.dataset.is_train = True

    return optimizer_lbs_c, optimizer_lbs_p


def train_skinning_net(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin,
                       model, smpl_vitruvian, gt_lbs_smpl,
                       train_data_loader, test_data_loader,
                       cuda, reference_body_v=None, optimizers=None):
    if not optimizers == None:
        optimizer_lbs_c = optimizers[0]
        optimizer_lbs_p = optimizers[1]
    else:
        optimizer = torch.optim.Adam(list(fwd_skin_net.parameters()) + list(inv_skin_net.parameters()),
                                     lr=opt['training']['lr_pt2'])

    smpl_face = torch.LongTensor(model.faces[:, [0, 2, 1]].astype(np.int32))[None].to(cuda)

    o_cyc_smpl = fwd_skin_net.opt['lambda_cyc_smpl']
    o_cyc_scan = fwd_skin_net.opt['lambda_cyc_scan']
    n_iter = 0
    for epoch in range(opt['training']['num_epoch_pt2']):
        fwd_skin_net.train()
        inv_skin_net.train()
        if epoch % opt['training']['resample_every_n_epoch'] == 0:
            train_data_loader.dataset.resample_flag = True
        else:
            train_data_loader.dataset.resample_flag = False
        if epoch == opt['training']['num_epoch_pt2'] // 2 or epoch == 3 * (opt['training']['num_epoch_pt2'] // 4):
            fwd_skin_net.opt['lambda_cyc_smpl'] *= 10.0
            fwd_skin_net.opt['lambda_cyc_scan'] *= 10.0
            if not optimizers == None:
                for j, _ in enumerate(optimizer_lbs_c.param_groups):
                    optimizer_lbs_c.param_groups[j]['lr'] *= 0.1
                for j, _ in enumerate(optimizer_lbs_p.param_groups):
                    optimizer_lbs_p.param_groups[j]['lr'] *= 0.1
            else:
                for j, _ in enumerate(optimizer.param_groups):
                    optimizer.param_groups[j]['lr'] *= 0.1
        for train_idx, train_data in enumerate(train_data_loader):
            betas = train_data['betas'].to(device=cuda)
            body_pose = train_data['body_pose'].to(device=cuda)
            # scan_cano = train_data['scan_cano'].to(device=cuda).permute(0,2,1)
            scan_posed = train_data['scan_cano_uni'].to(device=cuda)
            scan_tri = train_data['scan_tri_posed'].to(device=cuda)
            w_tri = train_data['w_tri'].to(device=cuda)
            transl = train_data['transl'].to(device=cuda)
            f_ids = train_data['f_id'].to(device=cuda)
            smpl_data = train_data['smpl_data']
            global_orient = body_pose[:, :3]
            body_pose = body_pose[:, 3:]

            smpl_neutral = smpl_data['smpl_neutral'].cuda()
            smpl_cano = smpl_data['smpl_cano'].cuda()
            smpl_posed = smpl_data['smpl_posed'].cuda()
            smpl_n_posed = smpl_data['smpl_n_posed'].cuda()
            bmax = smpl_data['bmax'].cuda()
            bmin = smpl_data['bmin'].cuda()
            jT = smpl_data['jT'].cuda()
            inv_rootT = smpl_data['inv_rootT'].cuda()

            scan_posed = torch.einsum('bst,bvt->bsv', inv_rootT, homogenize(scan_posed))[:, :3,
                         :]  # remove root transform
            scan_tri = torch.einsum('bst,btv->bsv', inv_rootT, homogenize(scan_tri, 1))[:, :3, :]

            reference_lbs_scan = compute_knn_feat(scan_posed.permute(0, 2, 1), smpl_posed,
                                                  gt_lbs_smpl.expand(scan_posed.shape[0], -1, -1).permute(0, 2, 1))[:,
                                 :, 0].permute(0, 2, 1)

            if opt['model']['inv_skin_net']['g_dim'] > 0:
                lat = lat_vecs_inv_skin(f_ids)  # (B, Z)
                inv_skin_net.set_global_feat(lat)

            feat3d_posed = None
            res_lbs_p, err_lbs_p, err_dict = inv_skin_net(feat3d_posed, smpl_posed.permute(0, 2, 1), gt_lbs_smpl,
                                                          scan_posed, reference_lbs_scan=reference_lbs_scan, jT=jT,
                                                          v_tri=scan_tri, w_tri=w_tri, bmin=bmin[:, :, None],
                                                          bmax=bmax[:, :, None])

            feat3d_cano = None
            res_lbs_c, err_lbs_c, err_dict_lbs_c = fwd_skin_net(feat3d_cano, smpl_cano, gt_lbs_smpl,
                                                                res_lbs_p['pred_scan_cano'], jT=jT, res_posed=res_lbs_p)

            # Back propagation
            err_dict.update(err_dict_lbs_c)
            err = err_lbs_p + err_lbs_c
            err_dict['All'] = err.item()

            if not optimizers == None:
                optimizer_lbs_c.zero_grad()
                optimizer_lbs_p.zero_grad()
            else:
                optimizer.zero_grad()
            err.backward()
            if not optimizers == None:
                optimizer_lbs_c.step()
                optimizer_lbs_p.step()
            else:
                optimizer.step()

            if n_iter % opt['training']['freq_plot'] == 0:
                err_txt = ''.join(['{}: {:.3f} '.format(k, v) for k, v in err_dict.items()])
                print('[%03d/%03d]:[%04d/%04d] %s' % (
                    epoch, opt['training']['num_epoch_pt2'], train_idx, len(train_data_loader), err_txt))
            n_iter += 1

    fwd_skin_net.opt['lambda_cyc_smpl'] = o_cyc_smpl
    fwd_skin_net.opt['lambda_cyc_scan'] = o_cyc_scan

    train_data_loader.dataset.is_train = False
    gen_mesh1(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model, smpl_vitruvian, train_data_loader,
              cuda, '_pt2', reference_body_v=reference_body_v)
    train_data_loader.dataset.is_train = True


def train(opt):
    cuda = torch.device('cuda:0')
    ckpt_dir = opt['experiment']['ckpt_dir']
    result_dir = opt['experiment']['result_dir']
    log_dir = opt['experiment']['log_dir']

    os.makedirs(result_dir, exist_ok=True)

    # Initialize vitruvian SMPL model
    if 'vitruvian_angle' not in opt['data']:
        opt['data']['vitruvian_angle'] = 25

    model = smpl.create(opt['data']['smpl_dir'], model_type='smpl_vitruvian',
                        gender=opt['data']['smpl_gender'], use_face_contour=False,
                        ext='npz').to(cuda)

    # Initialize dataset
    train_dataset = MVPDataset_scan(opt['data'], phase='train', smpl=model,
                                    device=cuda)
    test_dataset = MVPDataset_scan(opt['data'], phase='train', smpl=model,
                                   device=cuda)

    reference_body_vs_train = train_dataset.Tpose_minimal_v
    reference_body_vs_test = test_dataset.Tpose_minimal_v

    smpl_vitruvian = model.initiate_vitruvian(device=cuda,
                                              body_neutral_v=train_dataset.Tpose_minimal_v,
                                              vitruvian_angle=opt['data']['vitruvian_angle'])

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt['training']['batch_size'], shuffle=False,
                                   num_workers=opt['training']['num_threads'], pin_memory=opt['training']['pin_memory'])
    test_data_loader = DataLoader(test_dataset)

    # All the hand, face joints are glued to body joints for SMPL
    gt_lbs_smpl = model.lbs_weights[:, :24].clone()
    root_idx = model.parents.cpu().numpy()
    idx_list = list(range(root_idx.shape[0]))
    for i in range(root_idx.shape[0]):
        if i > 23:
            root = idx_list[root_idx[i]]
            gt_lbs_smpl[:, root] += model.lbs_weights[:, i]
            idx_list[i] = root
    gt_lbs_smpl = gt_lbs_smpl[None].permute(0, 2, 1)

    smpl_vitruvian = model.initiate_vitruvian(device=cuda,
                                              body_neutral_v=train_dataset.Tpose_minimal_v,
                                              vitruvian_angle=opt['data']['vitruvian_angle'])

    # define bounding box
    bbox_smpl = (smpl_vitruvian[0].cpu().numpy().min(0).astype(np.float32),
                 smpl_vitruvian[0].cpu().numpy().max(0).astype(np.float32))
    bbox_center, bbox_size = 0.5 * (bbox_smpl[0] + bbox_smpl[1]), (bbox_smpl[1] - bbox_smpl[0])
    bbox_min = np.stack([bbox_center[0] - 0.55 * bbox_size[0], bbox_center[1] - 0.6 * bbox_size[1],
                         bbox_center[2] - 1.5 * bbox_size[2]], 0).astype(np.float32)
    bbox_max = np.stack([bbox_center[0] + 0.55 * bbox_size[0], bbox_center[1] + 0.6 * bbox_size[1],
                         bbox_center[2] + 1.5 * bbox_size[2]], 0).astype(np.float32)

    # Initialize networks
    pose_map = get_posemap(opt['model']['posemap_type'], 24, model.parents, opt['model']['n_traverse'],
                           opt['model']['normalize_posemap'])

    igr_net = IGRSDFNet(opt['model']['igr_net'], bbox_min, bbox_max, pose_map).to(cuda)
    fwd_skin_net = LBSNet(opt['model']['fwd_skin_net'], bbox_min, bbox_max, posed=False).to(cuda)
    inv_skin_net = LBSNet(opt['model']['inv_skin_net'], bbox_min, bbox_max, posed=True).to(cuda)

    lat_vecs_igr = nn.Embedding(1, opt['model']['igr_net']['g_dim']).to(cuda)
    lat_vecs_inv_skin = nn.Embedding(len(train_dataset), opt['model']['inv_skin_net']['g_dim']).to(cuda)

    torch.nn.init.constant_(lat_vecs_igr.weight.data, 0.0)
    torch.nn.init.normal_(lat_vecs_inv_skin.weight.data, 0.0, 1.0 / math.sqrt(opt['model']['inv_skin_net']['g_dim']))

    print("igr_net:\n", igr_net)
    print("fwd_skin_net:\n", fwd_skin_net)
    print("inv_skin_net:\n", inv_skin_net)

    # Find checkpoints
    ckpt_dict = None
    if opt['experiment']['ckpt_file'] is not None:
        if os.path.isfile(opt['experiment']['ckpt_file']):
            logging.info('loading for ckpt...' + opt['experiment']['ckpt_file'])
            ckpt_dict = torch.load(opt['experiment']['ckpt_file'])
        else:
            logging.warning('ckpt does not exist [%s]' % opt['experiment']['ckpt_file'])
    elif opt['training']['continue_train']:
        model_path = '%s/ckpt_latest.pt' % ckpt_dir
        if os.path.isfile(model_path):
            logging.info('Resuming from ' + model_path)
            ckpt_dict = torch.load(model_path)
        else:
            logging.warning('ckpt does not exist [%s]' % model_path)
            opt['training']['use_trained_skin_nets'] = True
            model_path = '%s/ckpt_trained_skin_nets.pt' % ckpt_dir
            if os.path.isfile(model_path):
                logging.info('Resuming from ' + model_path)
                ckpt_dict = torch.load(model_path)
                logging.info('Pretrained model loaded.')
            else:
                logging.warning('ckpt does not exist [%s]' % model_path)
    elif opt['training']['use_trained_skin_nets']:
        model_path = '%s/ckpt_trained_skin_nets.pt' % ckpt_dir
        if os.path.isfile(model_path):
            logging.info('Resuming from ' + model_path)
            ckpt_dict = torch.load(model_path)
            logging.info('Pretrained model loaded.')
        else:
            logging.warning(
                'ckpt does not exist [%s] \n Failed to resume training, start training from beginning' % model_path)

    # Load checkpoints
    train_igr_start_epoch = 0
    if ckpt_dict is not None:
        if 'igr_net' in ckpt_dict:
            load_network(igr_net, ckpt_dict['igr_net'])
            if 'epoch' in ckpt_dict:
                train_igr_start_epoch = ckpt_dict['epoch']
        else:
            logging.warning("Couldn't find igr_net in checkpoints!")

        if 'fwd_skin_net' in ckpt_dict:
            load_network(fwd_skin_net, ckpt_dict['fwd_skin_net'])
        else:
            logging.warning("Couldn't find fwd_skin_net in checkpoints!")

        if 'inv_skin_net' in ckpt_dict:
            load_network(inv_skin_net, ckpt_dict['inv_skin_net'])
        else:
            logging.warning("Couldn't find inv_skin_net in checkpoints!")

        if 'lat_vecs_igr' in ckpt_dict:
            load_network(lat_vecs_igr, ckpt_dict['lat_vecs_igr'])
        else:
            logging.warning("Couldn't find lat_vecs_igr in checkpoints!")

        if 'lat_vecs_inv_skin' in ckpt_dict:
            load_network(lat_vecs_inv_skin, ckpt_dict['lat_vecs_inv_skin'])
        else:
            logging.warning("Couldn't find lat_vecs_inv_skin in checkpoints!")

    logging.info('train data size: %s' % str(len(train_dataset)))
    logging.info('test data size: %s' % str(len(test_dataset)))

    # Skip canonicalization
    if opt['training']['continue_train'] and os.path.isfile('%s/ckpt_trained_skin_nets.pt' % ckpt_dir):
        logging.info("Get fwd_skin_net, inv_skin_net and lat_vecs_inv_skin from trained skinning net!")
        trained_skin_nets_ckpt_dict = torch.load('%s/ckpt_trained_skin_nets.pt' % ckpt_dir)
        fwd_skin_net.load_state_dict(trained_skin_nets_ckpt_dict['fwd_skin_net'])
        inv_skin_net.load_state_dict(trained_skin_nets_ckpt_dict['inv_skin_net'])
        lat_vecs_inv_skin.load_state_dict(trained_skin_nets_ckpt_dict['lat_vecs_inv_skin'])

        opt['training']['skip_pt1'] = True
        opt['training']['skip_pt2'] = True

    # Pretrain fwd_skin_net and inv_skin_net net independently
    if not opt['training']['skip_pt1']:
        logging.info('start pretraining skinning nets (individual)')
        optimizers = pretrain_skinning_net(opt, log_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model,
                                           smpl_vitruvian, gt_lbs_smpl, train_data_loader, test_data_loader, cuda,
                                           reference_body_v=reference_body_vs_train)

    # Train fwd_skin_net and inv_skin_net jointly
    if not opt['training']['skip_pt2']:
        logging.info('start training skinning nets (joint)')
        train_skinning_net(opt, log_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model, smpl_vitruvian,
                           gt_lbs_smpl, train_data_loader, test_data_loader, cuda,
                           reference_body_v=reference_body_vs_train, optimizers=optimizers)

    if not opt['training']['skip_pt1'] and not opt['training']['skip_pt2']:
        ckpt_dict = {
            'opt': opt,
            'fwd_skin_net': fwd_skin_net.state_dict(),
            'inv_skin_net': inv_skin_net.state_dict(),
            'lat_vecs_inv_skin': lat_vecs_inv_skin.state_dict()
        }
        torch.save(ckpt_dict, '%s/ckpt_trained_skin_nets.pt' % ckpt_dir)

    # get only valid triangles
    train_data_loader.dataset.compute_valid_tri(inv_skin_net, model, lat_vecs_inv_skin, smpl_vitruvian)

    train_data_loader.dataset.is_train = False
    gen_mesh1(opt, result_dir, fwd_skin_net, inv_skin_net, lat_vecs_inv_skin, model, smpl_vitruvian, train_data_loader,
              cuda, '_pt3', reference_body_v=train_data_loader.dataset.Tpose_minimal_v, every_n_frame=1)
    train_data_loader.dataset.is_train = True


def trainWrapper(args=None):
    parser = argparse.ArgumentParser(
        description='Train SCANimate.'
    )
    parser.add_argument('--config', '-c', type=str, default='./lib/scanimate/config/example.yaml')
    parser.add_argument('--in_dir', '-i', type=str, default='./data_sample')
    parser.add_argument('--out_dir', '-o', type=str, default='./data_sample')
    args = parser.parse_args()

    opt = load_config(args.config, './lib/scanimate/config/default.yaml')

    for sid in os.listdir(args.in_dir):
        opt['data']['data_dir'] = os.path.join(args.in_dir, sid)
        opt['experiment']['ckpt_dir'] = os.path.join(args.out_dir, sid, 'cano')
        opt['experiment']['result_dir'] = os.path.join(args.out_dir, sid, 'cano')
        opt['experiment']['log_dir'] = os.path.join(args.out_dir, sid, 'cano')

        train(opt)


if __name__ == '__main__':
    trainWrapper()
