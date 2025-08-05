import numpy as np
import pickle
import tensorflow as tf


class ConeGeometry(object):
    """
    Cone beam CT geometry. Note that we convert to meter from millimeter.
    """

    def __init__(self, data):

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data["DSD"] / 1000  # Distance Source Detector      (m)
        self.DSO = data["DSO"] / 1000  # Distance Source Origin        (m)
        # Detector parameters
        self.nDetector = np.array(
            data["nDetector"]
        )  # number of pixels              (px)
        self.dDetector = (
            np.array(data["dDetector"]) / 1000
        )  # size of each pixel            (m)
        self.sDetector = (
            self.nDetector * self.dDetector
        )  # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data["nVoxel"])  # number of voxels              (vx)
        self.dVoxel = (
            np.array(data["dVoxel"]) / 1000
        )  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = (
            np.array(data["offOrigin"]) / 1000
        )  # Offset of image from origin   (m)
        self.offDetector = (
            np.array(data["offDetector"]) / 1000
        )  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data[
            "accuracy"
        ]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data["mode"]  # parallel, cone                ...
        self.filter = data["filter"]


class TIGREDataset:
    def __init__(self, path, device, n_rays):

        with open(path, "rb") as reader:
            data = pickle.load(reader)

        self.geo = ConeGeometry(data)

        self.projections = tf.convert_to_tensor(
                data["train"]["projections"], dtype=tf.float32)
        angles = data["train"]["angles"]
        self.ground_truth = data["image"]
        self.rays = self.get_rays(angles)
        self.near, self.far = self.get_near_far(self.geo)
        self.n_rays = n_rays
        self.voxels = self.get_voxels(self.geo)

    def __getitem__(self, index):
        projection = tf.reshape(self.projections[index, :, :], -1)
        rays = tf.reshape(self.rays[:, :, :, index], (-1, 6))
        indices = tf.convert_to_tensor(np.random.choice(projection.shape[0], self.n_rays, replace=False))
        selected_projections = tf.gather(projection, indices)
        selected_ray = tf.gather(rays, indices)

        return selected_projections, selected_ray

    def get_voxels(self, geo: ConeGeometry):
        """
        Get the voxels.
        """
        n1, n2, n3 = geo.nVoxel
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2

        offOrigin = geo.offOrigin

        xyz = np.meshgrid(
            np.linspace(-s1, s1, n1),
            np.linspace(-s2, s2, n2),
            np.linspace(-s3, s3, n3),
            indexing="ij",
        )
        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
        voxel = voxel + offOrigin[None, None, None, :]
        return voxel

    def get_rays(self, angles):
        width, height = self.geo.nDetector
        DSD = self.geo.DSD
        rays = []

        for angle in angles:
            pose = tf.convert_to_tensor(self.angle2pose(angle))
            i, j = tf.meshgrid(
                tf.linspace(0, width - 1, width),
                 tf.linspace(0, height - 1, height),
                indexing="ij"
            )
            uu = (tf.transpose(i) + 0.5 - width / 2) * self.geo.dDetector[0] + self.geo.offDetector[0]
            vv = (tf.transpose(j) + 0.5 - height / 2) * self.geo.dDetector[1] + self.geo.offDetector[1]
            dirs = tf.stack([uu / DSD, vv / DSD, tf.ones_like(uu)], -1)
            rays_d = tf.reduce_sum(
                tf.matmul(pose[:3, :3], dirs[..., None]), -1
            )  # pose[:3, :3] *
            rays_o = tf.broadcast_to(pose[:3, -1], tf.shape(rays_d))
            rays.append(tf.concat([rays_o, rays_d], axis=-1))

        return tf.stack(rays, axis=-1)

    def angle2pose(self, angle):
        phi1 = -np.pi / 2
        R1 = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(phi1), -np.sin(phi1)],
                [0.0, np.sin(phi1), np.cos(phi1)],
            ]
        )
        phi2 = np.pi / 2
        R2 = np.array(
            [
                [np.cos(phi2), -np.sin(phi2), 0.0],
                [np.sin(phi2), np.cos(phi2), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        R3 = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rot = np.dot(np.dot(R3, R2), R1)
        trans = np.array([self.geo.DSO * np.cos(angle), self.geo.DSO * np.sin(angle), 0])
        T = np.eye(4)
        T[:-1, :-1] = rot
        T[:-1, -1] = trans
        return T

    def get_near_far(self, geo: ConeGeometry, tolerance=0.005):
        """
        Compute the near and far threshold.
        """
        dist1 = np.linalg.norm(
            [geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2]
        )
        dist2 = np.linalg.norm(
            [geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2]
        )
        dist3 = np.linalg.norm(
            [geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2]
        )
        dist4 = np.linalg.norm(
            [geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2]
        )
        dist_max = np.max([dist1, dist2, dist3, dist4])
        near = np.max([0, geo.DSO - dist_max - tolerance])
        far = np.min([geo.DSO * 2, geo.DSO + dist_max + tolerance])
        return near, far
