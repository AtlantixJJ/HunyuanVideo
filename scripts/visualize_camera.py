import sys
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
import json
from diffusers.utils import export_to_video, load_image
from python_tsp.exact import solve_tsp_dynamic_programming

sys.path.insert(0, '.')


def print_2darray(arr):
    for i in range(arr.shape[0]):
        print('\t'.join([f'{x:.3f}' for x in arr[i]]))


data_dir = 'data/NeRSemble/data/017/camera_params.json'


root_dir = 'data/NeRSemble'
instance_dirs = sorted([d for d in glob.glob(f'{root_dir}/data/*')
                        if os.path.isdir(d)])
camera_fpath = f'{instance_dirs[0]}/camera_params.json'
camera_dic = json.load(open(camera_fpath, 'r'))

cam2worlds = {k: np.asarray(v) for k, v in camera_dic['world_2_cam'].items()}
cam_degs = {k: R.from_matrix(v[:3, :3]).as_euler('yxz', True) for k, v in cam2worlds.items()}
degs = np.array([v for v in cam_degs.values()])
degs[:, 0][degs[:, 0] < 0] += 360 # convert to 0-360
degs[:, 0] -= 180 # move the center to 0
degs[:, 2] += 180
degs[:, 2][degs[:, 2] > 190] -= 360

dist_matrix = ((degs[:, None] - degs[None]) ** 2).sum(-1)
permutation, distance = solve_tsp_dynamic_programming(dist_matrix)
sorted_names = [k for k in camera_dic['world_2_cam'].keys()]
sorted_names = [sorted_names[i] for i in permutation]
with open(f'{root_dir}/camera_order.txt', 'w') as f:
    f.write('\n'.join(sorted_names))
# seems to have two cameras vertically
# '-46.753 -43.866 -26.852 -26.376 -15.171 -14.483 -5.214 -5.225 5.469 5.347 14.231 14.580 26.214 26.736 45.808 45.814'
#camera_names = [k for k in camera_dic['world_2_cam'].keys()]
#camera_names = camera_names[::2] + camera_names[1::2][::-1]
#sorted_degs = [cam_degs[c] for c in camera_names]

cname_fmt = 'data/NeRSemble/data/017/extra_sequences/EMO-1-shout+laugh/frame_00000/images-2fps/cam_{}.jpg'
images = [load_image(cname_fmt.format(cam_name)).resize((275, 401))
    for cam_name in camera_dic['world_2_cam'].keys()]
export_to_video([images[i] for i in permutation], 'camera_tsp.mp4', fps=2)

"""
ds = MultiViewDataset(
    root_dir='data/NeRSemble/data',
    ann_file='data/NeRSemble/ann.parquet',
    n_ref=16, n_tarfar=-1, n_tarnear=-1)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

for i, batch in enumerate(dl):
    break
    
world2cams = batch['ref_camera']['_camdata_world_view_transform'].permute(0, 1, 3, 2)
cam2worlds = torch.inverse(world2cams).numpy()
np.save('data/NeRSemble/cam2worlds.npy', cam2worlds)
"""

"""

def plot_cone(ax, t, R, color):
    # Define cone properties
    r = 0.1  # Base radius of the cone
    h = 0.2  # Height of the cone
    phi = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, h, 2)
    Z, Phi = np.meshgrid(z, phi)
    X = r * (1 - Z / h) * np.cos(Phi)
    Y = r * (1 - Z / h) * np.sin(Phi)

    # Rotate and translate the cone
    cone = np.array([X.flatten(), Y.flatten(), Z.flatten()])
    rotated_cone = np.dot(R, cone) + t[:, np.newaxis]

    # Plot the cone
    ax.plot_surface(rotated_cone[0, :].reshape(2, -1), rotated_cone[1, :].reshape(2, -1), rotated_cone[2, :].reshape(2, -1), color=color)


def plot_pyramid(ax, cam2world, color):
    # Define the vertices of the pyramid
    pyramid_vertices = np.array([
        [-0.2, -0.2, 0],
        [0.2, -0.2, 0],
        [0.2, 0.2, 0],
        [-0.2, 0.2, 0],
        [0, 0, -0.4]
    ])

    # Rotate and translate the pyramid
    rotated_pyramid = (cam2world[:3, :3] @ pyramid_vertices.T).T + cam2world[:3, 3]

    # Define the faces of the pyramid
    pyramid_faces = [
        [rotated_pyramid[0], rotated_pyramid[1], rotated_pyramid[4]],
        [rotated_pyramid[1], rotated_pyramid[2], rotated_pyramid[4]],
        [rotated_pyramid[2], rotated_pyramid[3], rotated_pyramid[4]],
        [rotated_pyramid[3], rotated_pyramid[0], rotated_pyramid[4]],
        [rotated_pyramid[0], rotated_pyramid[1], rotated_pyramid[2], rotated_pyramid[3]]
    ]

    ax.add_collection3d(Poly3DCollection(pyramid_faces, facecolors=color, linewidths=1, edgecolors='k'))


def euler_to_euclidean(yaw, pitch, roll, r):
    #Convert euler angles to euclidean coordinates
    x = np.cos(yaw) * np.cos(pitch)
    y = np.sin(yaw) * np.cos(pitch)
    z = np.sin(pitch)
    return np.array([x, y, z]) * r


# Create a 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 3, 0, 4, 8
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
# Plot cameras
for i, cam2world in enumerate(cam2worlds):
    plot_pyramid(ax, cam2world, colors[0])
plot_pyramid(ax, np.eye(4), (1, 1, 1))

# Set labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.legend()

plt.title('Camera Extrinsics Visualization')
#plt.show()
plt.savefig(f'{root_dir}/camera_extrinsics.png')
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
E = cam2worlds
RT = E[:1, :3, :3].transpose(0, 2, 1)
#E[:, :3, :3] = RT @ E[:, :3, :3]
E[:, :3] = RT @ E[:, :3]
for i, cam2world in enumerate(E):
    plot_pyramid(ax, cam2world, colors[i%3])
plot_pyramid(ax, np.eye(4), (1, 1, 1))

# Set labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.legend()
plt.savefig(f'camera_extrinsics_transform.png')
plt.close()
"""