"""Utility functions to download data used in the test infrastructure."""
from abc import abstractclassmethod
import os
import shutil
import tarfile
import urllib.request
import zipfile

import open3d as o3d
import pyopenvdb as vdb


# All OpenVDB models are located here:
OPENVDB_BASE_URL = "https://artifacts.aswf.io/io/aswf/openvdb/models/"
STANDFORD_BASE_URL = "http://graphics.stanford.edu/pub/3Dscanrep/"
MESH_FUSION_BASE_URL = "https://raw.githubusercontent.com/davidstutz/mesh-fusion/master/"


def file_relative_path(path):
    """Returns /<path>/test/test_data/*.vdb."""
    file_path = os.path.realpath(__file__)
    file_dir = os.path.dirname(file_path)
    return os.path.join(file_dir, path)


class TestData:
    """Wrapper class around the .vdb File."""

    def __init__(self):
        # .vdb file
        self.vdb = None
        self.vdb_name = ""
        self.vdb_path = ""
        self.zip_path = ""

        # .ply file
        self.ply = None
        self.ply_path = ""
        self.tar_path = ""

    def _download_vdb(self):
        """Download .vdb from the official OpenVDB artifacts mirror."""
        if not os.path.exists(self.vdb_path):
            print("downloading", self.vdb_path)
            url = os.path.join(OPENVDB_BASE_URL, self.zip_orig)
            urllib.request.urlretrieve(url, self.zip_path)
            with zipfile.ZipFile(self.zip_path, "r") as zip_file:
                zip_file.extractall(os.path.dirname(self.vdb_path))
            os.remove(self.zip_path)
            self._post_download_vdb()

    def _read_vdb_grid(self):
        return vdb.read(self.vdb_path, self.vdb_name)

    def download_vdb(self):
        self._download_vdb()
        self.vdb = self._read_vdb_grid()

    def _download_ply(self):
        if not os.path.exists(self.ply_path):
            print("downloading", self.ply_path)
            url = os.path.join(STANDFORD_BASE_URL, self.tar_orig)
            urllib.request.urlretrieve(url, self.tar_path)
            with tarfile.open(self.tar_path) as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, path=os.path.dirname(self.ply_path))
            os.remove(self.tar_path)
            self._post_download_ply()

    def _read_triangle_mesh(self):
        mesh = o3d.io.read_triangle_mesh(self.ply_path)
        mesh.compute_vertex_normals()
        return mesh

    def download_ply(self):
        self._download_ply()
        self.ply = self._read_triangle_mesh()

    @abstractclassmethod
    def _post_download_ply(self):
        """To be implemented in derived classes."""
        pass

    @abstractclassmethod
    def _post_download_vdb(self):
        """To be implemented in derived classes."""
        pass


class Bunny(TestData):
    def __init__(self):
        super().__init__()
        # Get the vdb representation
        self.vdb_name = "ls_bunny"
        self.zip_orig = "bunny.vdb/1.0.0/bunny.vdb-1.0.0.zip"
        self.vdb_path = file_relative_path("test_data/bunny.vdb")
        self.zip_path = file_relative_path("test_data/bunny.zip")
        self.download_vdb()

        # Get the mesh representation
        self.tar_orig = "bunny.tar.gz"
        self.ply_path = file_relative_path("test_data/bunny.ply")
        self.tar_path = file_relative_path("test_data/bunny.tar.gz")
        self.download_ply()

    def _post_download_ply(self):
        shutil.move(
            os.path.join(
                os.path.dirname(self.ply_path),
                "bunny",
                "reconstruction",
                "bun_zipper.ply",
            ),
            self.ply_path,
        )
        shutil.rmtree(os.path.join(os.path.dirname(self.ply_path), "bunny"))


class Torus(TestData):
    def __init__(self):
        super().__init__()
        self.vdb_name = "ls_torus"
        self.zip_orig = "torus.vdb/1.0.0/torus.vdb-1.0.0.zip"
        self.vdb_path = file_relative_path("test_data/torus.vdb")
        self.zip_path = file_relative_path("test_data/torus.zip")
        self.download_vdb()


class Knot(TestData):
    def __init__(self):
        super().__init__()
        self.ply_path = file_relative_path("test_data/knot.ply")
        self.download_ply()


class TestDataOff(TestData):
    """Get some .off charis that are already online, so we don't populate our git repository."""

    def _download_off(self):
        if not os.path.exists(self.off_path):
            print("downloading", self.off_path)
            url = os.path.join(MESH_FUSION_BASE_URL, self.off_orig)
            urllib.request.urlretrieve(url, self.off_path)


class Chair1(TestDataOff):
    def __init__(self):
        super().__init__()
        self.off_orig = "examples/0_in/chair_0891.off"
        self.ply_path = self.off_path = file_relative_path("test_data/chair_0891.off")
        self._download_off()
        self.ply = self._read_triangle_mesh()


class Chair2(TestDataOff):
    def __init__(self):
        super().__init__()
        self.off_orig = "examples/0_in/chair_0894.off"
        self.ply_path = self.off_path = file_relative_path("test_data/chair_0894.off")
        self._download_off()
        self.ply = self._read_triangle_mesh()


if __name__ == "__main__":
    print("Downloading test data...")
    bunny = Bunny()
