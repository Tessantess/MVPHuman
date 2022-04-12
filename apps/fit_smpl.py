import os
import sys
import pickle
import trimesh
import numpy as np
from tqdm import tqdm
from termcolor import colored
import torch
import torch.nn as nn
from pytorch3d.ops.knn import knn_points

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import smpl
from pathlib import Path
from torch.utils.data import Dataset

SMPL = smpl.create('../data/smpl_related/models')
smpl_faces = SMPL.faces.astype(np.int32)


class MaxMixturePosePrior(nn.Module):
    def __init__(self, n_gaussians=8, prefix=3, device=torch.device('cpu')):
        super(MaxMixturePosePrior, self).__init__()
        self.prefix = prefix
        self.n_gaussians = n_gaussians
        self.create_prior_from_cmu(device)

    def create_prior_from_cmu(self, device):
        """Load the gmm from the CMU motion database."""
        with open(os.path.join(os.path.dirname(__file__), '../data/smpl_related/smpl_data/gmm_08.pkl'), 'rb') as f:
            gmm = pickle.load(f, encoding='bytes')
        precs = np.asarray([np.linalg.cholesky(np.linalg.inv(cov)) for cov in gmm[b'covars']])
        means = np.asarray(gmm[b'means'])  # [8, 69]

        sqrdets = np.array([(np.sqrt(np.linalg.det(c))) for c in gmm[b'covars']])
        const = (2 * np.pi) ** (69 / 2.)
        weights = np.asarray(gmm[b'weights'] / (const * (sqrdets / sqrdets.min())))

        self.precs = torch.from_numpy(precs).to(device)  # [8, 69, 69]
        self.means = torch.from_numpy(means).to(device)  # [8, 69]
        self.weights = torch.from_numpy(weights).to(device)

    def forward(self, theta):
        theta = theta[:, self.prefix:]
        batch, dim = theta.shape
        theta = theta.expand(self.n_gaussians, batch, dim).permute(1, 0, 2)
        theta = (theta - self.means[None])[:, :, None, :]
        loglikelihoods = np.sqrt(0.5) * torch.matmul(theta, self.precs.expand(batch, *self.precs.shape)).squeeze(2)
        results = (loglikelihoods * loglikelihoods).sum(-1) - self.weights.log()
        return results.min()


def PoseAngleConstrain(theta):
    loss = torch.exp(theta[:, 55]) + torch.exp(-theta[:, 58]) + torch.exp(-theta[:, 12]) + torch.exp(-theta[:, 15])

    return torch.mean(loss ** 2)


def point_to_surface_dist(x, y):
    """
        x: Tensor of shape (N, P1, D) giving a batch of N point clouds, each
            containing up to P1 points of dimension D.
        y: Tensor of shape (N, P2, D) giving a batch of N point clouds, each
            containing up to P2 points of dimension D.
    """
    x_nn = knn_points(x, y)

    return x_nn.dists[..., 0].mean()


def normalize_vertices_batch(vertices, return_params=False):
    """ normalize vertices to [-1, 1]
    Args:
        vertices: FloatTensor [N, 3]
        return_params: return center and scale if True
    Return:
        normalized_v: FloatTensor [N, 3]
    """
    if not torch.is_tensor(vertices):
        vertices = torch.as_tensor(vertices)
    vmax = vertices.max(1)[0]
    vmin = vertices.min(1)[0]
    center = -0.5 * (vmax + vmin)
    scale = (1. / (vmax - vmin).max(1)[0])
    normalized_v = (vertices + center[:, None, :]) * scale[:, None, None] * 2.
    if return_params:
        return normalized_v, center, scale
    return normalized_v


def get_loss_weights():
    """Set loss weights"""

    loss_weight = {'s2m': lambda x, it: 10 ** -5 * x * (1 + it),
                   'm2s': lambda x, it: 10 ** -3 * x / (1 + it),
                   'betas': lambda x, it: 10. ** 0 * x / (1 + it),
                   'pose_pr': lambda x, it: 10. ** 0 * x / (1 + it),
                   'angle_pr': lambda x, it: 10. ** 0 * x / (1 + it),
                   }
    return loss_weight


def fit_smpl_batch(point_clouds,
                   initial_pose=None,
                   initial_beta=None,
                   learning_rate=0.01,
                   n_iter=600
                   ):
    batch = len(point_clouds)
    device = point_clouds.device

    # set initial parameters
    if initial_pose is None:
        initial_pose = torch.zeros(batch, 24, 3).to(device)
    if initial_beta is None:
        initial_beta = torch.zeros(batch, 10).to(device)
    output = SMPL(global_orient=initial_pose[:, :1], body_pose=initial_pose[:, 1:], betas=initial_beta,
                  custom_out=True, apply_trans=True)
    jointT = output.joint_transform[:, :24]

    # compute initial scale and trans
    _, scan_center, scan_scale = normalize_vertices_batch(point_clouds, return_params=True)
    _, smpl_center, smpl_scale = normalize_vertices_batch(output.vertices, return_params=True)
    initial_scale = smpl_scale / scan_scale
    initial_trans = -scan_center + smpl_center * initial_scale[:, None]

    if not torch.is_tensor(initial_pose):
        initial_pose = torch.as_tensor(initial_pose)
    if not torch.is_tensor(initial_beta):
        initial_beta = torch.as_tensor(initial_beta)

    # set parameters
    pose = torch.nn.Parameter(initial_pose.view(batch, 72).float().to(device), requires_grad=True)
    betas = torch.nn.Parameter(initial_beta.view(batch, 10).float().to(device), requires_grad=True)
    trans = torch.nn.Parameter(initial_trans.view(batch, 3).float().to(device), requires_grad=True)
    scale = torch.nn.Parameter(initial_scale.float().to(device), requires_grad=True)

    optimizer = torch.optim.Adam([pose, betas, scale, trans], lr=learning_rate)
    MMPP = MaxMixturePosePrior(device=device)

    print(colored('Star Fitting SMPL Body...'))
    pbar = tqdm(range(n_iter))
    loss_dict = dict()
    weight_dict = get_loss_weights()
    for it in pbar:
        output = SMPL(global_orient=pose[:, :1], body_pose=pose[:, 1:], betas=betas,
                      custom_out=True, apply_trans=True)
        jointT = output.joint_transform[:, :24]
        pred_v = output.vertices * scale[:, None, None] + trans[:, None, :]

        # penalize hands' points
        # hand_points = pred_v[:, hand_indices]
        # point_to_surface_dist()

        loss_dict['m2s'] = point_to_surface_dist(point_clouds, pred_v)
        loss_dict['s2m'] = point_to_surface_dist(pred_v, point_clouds)
        loss_dict['betas'] = torch.mean(betas ** 2)
        loss_dict['angle_pr'] = PoseAngleConstrain(pose)
        loss_dict['pose_pr'] = MMPP(pose)

        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = weight_dict[k](loss_dict[k], it)

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()

        pbar.set_description(
            "'Iter: %03d | loss: %.4f | pose prior: %.4f | s2m: %.4f | m2s: %.4f|  betas: %.4f " %
            (it, tot_loss.item(), w_loss['pose_pr'].item(), w_loss['s2m'].item(),
             w_loss['m2s'].item(), w_loss['betas'].item()))

        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()

    pose = pose.cpu().detach()
    betas = betas.cpu().detach()
    scale = scale.cpu().detach()
    transl = trans.cpu().detach()
    pred_v = pred_v.cpu().detach()
    v_shaped = output.v_shaped.cpu().detach()
    jointT = jointT.cpu().detach()

    return pose, betas, scale, transl, jointT, pred_v, v_shaped


def optimize_shape_only(point_clouds,
                        initial_pose,
                        initial_beta,
                        initial_trans,
                        initial_scale,
                        learning_rate=0.01,
                        n_iter=600
                        ):
    batch = len(point_clouds)
    device = point_clouds.device

    # set parameters
    pose = torch.nn.Parameter(initial_pose.view(batch, 72).float().to(device), requires_grad=True)
    betas = torch.nn.Parameter(initial_beta.view(batch, 10).float().to(device), requires_grad=True)
    trans = torch.nn.Parameter(initial_trans.view(batch, 3).float().to(device), requires_grad=True)
    scale = torch.nn.Parameter(initial_scale.float().to(device), requires_grad=True)

    optimizer = torch.optim.Adam([betas, scale, trans], lr=learning_rate)
    print('star optimizing shape only...')
    pbar = tqdm(range(n_iter))
    loss_dict = dict()
    weight_dict = get_loss_weights()
    for it in pbar:
        output = SMPL(global_orient=pose[:, :1], body_pose=pose[:, 1:], betas=betas,
                      custom_out=True, apply_trans=True)
        jointT = output.joint_transform[:, :24]
        pred_v = output.vertices * scale[:, None, None] + trans[:, None, :]

        loss_dict['m2s'] = point_to_surface_dist(point_clouds, pred_v)
        loss_dict['s2m'] = point_to_surface_dist(pred_v, point_clouds)
        loss_dict['betas'] = torch.mean(betas ** 2)

        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = weight_dict[k](loss_dict[k], it)

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()

        pbar.set_description(
            "'Iter: %03d | loss: %.4f | pose prior: %.4f | s2m: %.4f | m2s: %.4f|  betas: %.4f " %
            (it, tot_loss.item(), w_loss['pose_pr'].item(), w_loss['s2m'].item(),
             w_loss['m2s'].item(), w_loss['betas'].item()))

        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()

    pose = pose.cpu().detach()
    betas = betas.cpu().detach()
    scale = scale.cpu().detach()
    transl = trans.cpu().detach()
    pred_v = pred_v.cpu().detach()
    v_shaped = output.v_shaped.cpu().detach()
    jointT = jointT.cpu().detach()

    return pose, betas, scale, transl, jointT, pred_v, v_shaped


def load_front_mesh(obj_path, transform_path):
    mesh = trimesh.load(obj_path, process=False)
    transform_matrix = np.loadtxt(transform_path)
    mesh.vertices = np.matmul(mesh.vertices, transform_matrix[:3, :3].T)
    mesh.vertices -= 0.5 * (mesh.vertices.max(0) + mesh.vertices.min(0))[None]
    return mesh


def run_fit(point_clouds, device, save_path, initial_from_pymaf=False, initial_from_loopreg=False):
    if initial_from_pymaf:
        # todo
        # 1. render scan in front view
        # 2. predict initial smpl use pymaf
        # 3. optimize pose use silhouette [optional]
        # 4. initialize smpl parameters
        # 5. fit smpl
        pass
    elif initial_from_loopreg:
        # todo by Liu Xingcheng
        pass
    else:
        # fit use default parameters
        fit_smpl_batch(point_clouds.to(device), save_path=save_path)


class RawScanData(Dataset):
    def __init__(self, data_dir, device, num_samples=20000):
        self.root = data_dir
        self.subjects = sorted(os.listdir(data_dir))
        self.num_samples = num_samples
        self.render = Render(device=device)

    def __len__(self):
        return len(self.subjects)

    def load_origin_scan(self, sid):
        obj_path = f'{self.root}/{sid}/A03/Action03.obj'
        transform_path = f'{self.root}/{sid}/A03/R0.txt'
        mesh = load_front_mesh(obj_path, transform_path)
        return mesh

    def render_mesh(self, scan):
        color = torch.as_tensor(np.ones_like(scan.vertices)) * torch.FloatTensor([1, 0, 0])
        self.render.load_mesh(torch.from_numpy(scan.vertices),
                              torch.from_numpy(scan.faces),
                              color)
        image = self.render.get_image(cam_id=[2])
        return image

    def get_item(self, sid):
        scan = self.load_origin_scan(sid)
        image = self.render_mesh(scan)
        surface_points, _ = trimesh.sample.sample_surface(scan, self.num_samples)

        del scan
        return {
            'sid': sid,
            'samples': torch.from_numpy(surface_points).float(),
            'image': image
        }

    def __getitem__(self, index):
        sid = self.subjects[index]
        return self.get_item(sid)


if __name__ == '__main__':
    from tqdm import tqdm
    from lib.common.render import Render
    import argparse
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', type=str, default='./data_sample')
    parser.add_argument('-o', '--out_dir', type=str, default='./data_sample')
    parser.add_argument('-g', '--gpu_id', type=int, default=0)
    parser.add_argument('-v', '--vis', type=bool, default=True)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}')
    dataset = RawScanData(args.in_dir, device)
    SMPL.to(device)

    for sid in dataset.subjects:
        save_dir = f'{args.out_dir}/%s/Action03/smpl' % sid
        os.makedirs(save_dir, exist_ok=True)

        data = dataset.get_item(sid)

        samples = data['samples'].to(device)
        poses, betas, scales, trans, jointT, posed_v, v_shaped = fit_smpl_batch(samples[None])

        # save data
        minimal_mesh = trimesh.Trimesh(v_shaped[0], SMPL.faces[:, [0, 2, 1]])
        minimal_mesh.export(os.path.join(save_dir, 'minimally.obj'))
        np.savez(
            os.path.join(save_dir, 'Action03.npz'),
            pose=poses[0].numpy(),
            betas=betas[0].numpy(),
            scale=scales[0].numpy(),
            transl=trans[0].numpy(),
            jointT=jointT[0].numpy()
        )
        if args.vis:
            dataset.render.load_mesh(posed_v[0], torch.from_numpy(smpl_faces))
            smpl_image = dataset.render.get_image(cam_id=[2])

            overlap = (data['image'] + smpl_image) * 0.5
            cv2.imwrite(f'{save_dir}/overlap.png', overlap * 255)

