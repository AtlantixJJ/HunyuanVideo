import numpy as np
import json
import os
import glob
from tqdm import tqdm


def nersemble_load_camera_params(camera_params_path):
    """Load camera parameters from a json file (NeRSemble format)."""
    with open(camera_params_path, 'r') as f:
        camera_params = json.load(f)
    intrinsics = np.asarray(camera_params["intrinsics"])
    world2cam = {f'cam_{serial}': np.asarray(world2cam)
                 for serial, world2cam in camera_params["world_2_cam"].items()}
    return world2cam, intrinsics



def create_nersemble_ann(root_dir='data/NeRSemble', n_frame=49):
    """Create a dataframe with NeRSemble annotations.
    for each participant/scene, we store one file encoding:
    (1) All camera matrices and intrinsics
    (2) The camera indices and frame indices of the sampled sequence
    (3) The path to the images and masks of the sampled sequence
    (4) The encoded video latent of this sequence, including image sequence and condition sequence
    """
    camera_loop_order = open(f'{root_dir}/camera_order.txt').read().splitlines()
    n_loop = len(camera_loop_order)
    n_frame = 49

    seq_dir = 'extra_sequences'
    participants = [p for p in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, p))]
    participants.sort()
    data = []
    for p_id in tqdm(participants):
        p_dir = os.path.join(root_dir, p_id, seq_dir)
        scene_dirs = [s for s in os.listdir(p_dir)]
        scene_dirs.sort()

        world2cam, intrinsics = nersemble_load_camera_params(
            os.path.join(root_dir, p_id, 'camera_params.json'))
        world2cam = np.stack([world2cam[cam] for cam in camera_loop_order]).reshape(-1)

        for scene_dir in scene_dirs:
            # (1) All camera matrices and intrinsics
            scene_ann = {
                'world2cam': world2cam,
                'intrinsics': intrinsics
            }

            # sample the number of frames for each view
            coef = np.random.rand(n_loop)
            coef = np.clip(np.ceil(coef / coef.sum()), a_min=1)
            coef = (coef / coef.sum()).astype('int32')
            coef[-1] = n_frame


            frame_paths = sorted(glob.glob(os.path.join(p_dir, scene_dir, 'frame_*')))
            
            for frame_path in frame_paths:
                image_names = os.listdir(os.path.join(frame_path, 'images-2fps'))
                image_num = len(image_names)
                if image_num == 0:
                    print(f'Empty folder: {frame_path}')
                    continue # skip empty folder (possibly the last one)
                image_paths = [os.path.join('images-2fps', f'{cam}.jpg') for cam in camera_loop_order]
                mask_paths = [os.path.join('masks-2fps', f'{cam}.png') for cam in camera_loop_order]
                flame_cam = os.path.join('d3', 'cam_222200038_cam.pkl')
                flame_coef = os.path.join('d3', 'cam_222200038_flame.pkl')
                frame_name = os.path.basename(frame_path)
                cap_name = os.path.join(p_id, seq_dir, scene_dir, frame_name)
                data.append([cap_name, world2cam, intrinsics, image_paths, mask_paths, flame_cam, flame_coef])
    return data, DF_COLUMNS
