import numpy as np
import open3d as o3d


class SerializableMesh:
    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        self.vertices = np.asarray(mesh.vertices)
        self.triangles = np.asarray(mesh.triangles)

    def as_open3d(self) -> o3d.geometry.TriangleMesh:
        return o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.vertices),
            triangles=o3d.utility.Vector3iVector(self.triangles),
        )
