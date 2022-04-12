import cv2
import trimesh
import numpy as np
import torch
import os
import sys
import os.path as osp
import torchvision
from PIL import Image, ImageFont, ImageDraw
from scipy.spatial import cKDTree

from lib.common.sdf import create_grid, eval_grid_octree, eval_grid
from skimage import measure
from lib.common.pyrender import MeshRender

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.ops.knn import knn_points

from lib.common.render_utils import Pytorch3dRasterizer, face_vertices, batch_contains
import bvh_distance_queries
import smpl


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3, 3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3, 3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R


def rotate_mesh(mesh, R):
    """ normalize mesh
    Args:
        mesh: pytorch3d.structures.Meshes
        R: FloatTensor [3, 3]
    Return:
        rotated_mesh: pytorch3d.structures.Meshes
    """
    rotated_v = torch.matmul(mesh.verts_packed(), R.transpose(0, 1))
    rotated_mesh = mesh.offset_verts(rotated_v - mesh.verts_packed())
    return rotated_mesh


def normalize_v3(arr):
    """ Normalize a numpy array of 3 component vectors shape=(n,3) """
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles,
    # by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal.
    # Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


def build_mesh_by_poisson(vertices, faces, num_verts=30000):
    from pypoisson import poisson_reconstruction
    """ build a graph from mesh using https://github.com/mmolero/pypoisson """
    normals = compute_normal(vertices, faces)
    idx = np.random.randint(0, vertices.shape[0], size=(num_verts,))
    new_faces, new_vertices = poisson_reconstruction(vertices[idx], normals[idx], depth=10)
    return new_vertices, new_faces


def save_obj_mesh(mesh_path, verts, faces=None):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))

    if faces is not None:
        for f in faces:
            f_plus = f + 1
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def normalize_vertices(vertices, return_params=False):
    """ normalize vertices to [-1, 1]
    Args:
        vertices: FloatTensor [N, 3]
        return_params: return center and scale if True
    Return:
        normalized_v: FloatTensor [N, 3]
    """
    if not torch.is_tensor(vertices):
        vertices = torch.as_tensor(vertices)
    vmax = vertices.max(0)[0]
    vmin = vertices.min(0)[0]
    center = -0.5 * (vmax + vmin)
    scale = (1. / (vmax - vmin).max()).item()
    normalized_v = (vertices + center[None, :]) * scale * 2.
    if return_params:
        return normalized_v, center, scale
    return normalized_v


def normalize_vertices_batch(vertices, return_params=False):
    # TODO
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
    scale = (1. / (vmax - vmin).max()).item()
    normalized_v = (vertices + center[None, :]) * scale * 2.
    if return_params:
        return normalized_v, center, scale
    return normalized_v


def normalize_mesh(mesh, return_params=False):
    """ normalize mesh
    Args:
        mesh: pytorch3d.structures.Meshes
        return_params: return center and scale if True
    Return:
        normalized_mesh: pytorch3d.structures.Meshes
    """
    if return_params:
        normalized_v, center, scale = normalize_vertices(mesh.verts_packed(), return_params)
        normalized_mesh = mesh.offset_verts_(normalized_v - mesh.verts_packed())
        return normalized_mesh, center, scale
    normalized_v = normalize_vertices(mesh.verts_packed(), return_params)
    normalized_mesh = mesh.offset_verts_(normalized_v - mesh.verts_packed())
    return normalized_mesh


def get_optim_grid_image(per_loop_lst, loss=None, nrow=4, type='smpl'):
    font_path = os.path.join(os.path.dirname(__file__), "tbfo.ttf")
    font = ImageFont.truetype(font_path, 30)
    grid_img = torchvision.utils.make_grid(torch.cat(per_loop_lst, dim=0),
                                           nrow=nrow)
    grid_img = Image.fromarray(
        ((grid_img.permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 0.5 *
         255.0).astype(np.uint8))

    # add text
    draw = ImageDraw.Draw(grid_img)
    grid_size = 512
    if loss is not None:
        draw.text((10, 5), f"error: {loss:.3f}", (255, 0, 0), font=font)

    if type == 'smpl':
        for col_id, col_txt in enumerate(
                ['image', 'smpl-mask(render)', 'diff-mask']):
            draw.text((10 + (col_id * grid_size), 5), col_txt, (255, 0, 0), font=font)
    elif type == 'cloth':
        for col_id, col_txt in enumerate(
                ['image', 'cloth-norm(recon)', 'cloth-norm(pred)', 'diff-norm']):
            draw.text((10 + (col_id * grid_size), 5), col_txt, (255, 0, 0), font=font)
        for col_id, col_txt in enumerate(
                ['0', '90', '180', '270']):
            draw.text((10 + (col_id * grid_size), grid_size * 2 + 5), col_txt, (255, 0, 0), font=font)
    else:
        print(f"{type} should be 'smpl' or 'cloth'")

    grid_img = grid_img.resize((grid_img.size[0], grid_img.size[1]),
                               Image.ANTIALIAS)

    return grid_img


def detect_valid_triangle(canon_verts, posed_verts, faces):
    """
    detect valid faces within length
    :param posed_verts: Nx3
    :param faces: NFx3
    :return: triangle mask
    """
    e1 = torch.norm(posed_verts[faces[:, 0]] - posed_verts[faces[:, 1]], p=2, dim=1, keepdim=True)
    e2 = torch.norm(posed_verts[faces[:, 1]] - posed_verts[faces[:, 2]], p=2, dim=1, keepdim=True)
    e3 = torch.norm(posed_verts[faces[:, 2]] - posed_verts[faces[:, 0]], p=2, dim=1, keepdim=True)
    e = torch.cat([e1, e2, e3], 1)

    E1 = torch.norm(canon_verts[faces[:, 0]] - canon_verts[faces[:, 1]], p=2, dim=1, keepdim=True)
    E2 = torch.norm(canon_verts[faces[:, 1]] - canon_verts[faces[:, 2]], p=2, dim=1, keepdim=True)
    E3 = torch.norm(canon_verts[faces[:, 2]] - canon_verts[faces[:, 0]], p=2, dim=1, keepdim=True)
    E = torch.cat([E1, E2, E3], 1)

    max_edge = (E / e).max(1)[0]
    min_edge = (E / e).min(1)[0]

    # mask = 1.0 - (((max_edge > 2.0) & flag_tri) | (max_edge > 3.0) | (min_edge < 0.1)).cpu().float().numpy()
    mask = 1.0 - ((max_edge > 2.0) | (min_edge < 0.1)).cpu().float().numpy()
    tri_mask = mask > 0.5
    return tri_mask


