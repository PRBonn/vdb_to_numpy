import copy
from functools import partial
import time

import numpy as np
import open3d as o3d

from ..utils import extract_mesh

AIS_BLUE = [0.2, 0.4, 0.6]
AIS_GREEN = [0.2, 0.604, 0]
AIS_RED = [0.604, 0, 0]
AIS_GREY = [0.8, 0.8, 0.8]


def get_leaf_node_voxel_grid(origin, shape, color, voxel_size=1.0):
    return o3d.geometry.VoxelGrid.create_dense(
        origin=origin,
        color=color,
        width=voxel_size * shape[0],
        height=voxel_size * shape[1],
        depth=voxel_size * shape[2],
        voxel_size=voxel_size,
    )


class LeafNodeGridVisualizer:
    def __init__(self, grid, model, sleep_time=200e-3):
        # Store the LeafNodeGrid and the original mesh model
        self.grid = grid
        self.model = model

        # Acess the leaf model from the class level
        self.camera_params = None
        self.geometries = None
        self.reconstructed_mesh = o3d.geometry.TriangleMesh()

        # Visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.render_options = self.vis.get_render_option()
        self.view_control = self.vis.get_view_control()
        self.render_mesh = False
        self.render_voxels = True
        self.render_reconstruction = False
        self.global_view = False

        # Initialize dataset item control variables
        self.idx = 0
        self.leaf_count = len(self.grid)

        # Continous time plot
        self.stop = False
        self.sleep_time = sleep_time
        self.sleep_step = self.sleep_time / 10.0

        # Initialize the default callbacks
        self.update_geometries() if not self.geometries else None
        self._register_key_callbacks()
        self._print_help()
        self._initialize_visualizer()

    def _initialize_visualizer(self):
        self.update_visualizer(reset_bounding_box=True)

    def update_geometries(self, inc_idx=1):
        """Tries to extract a mesh from a leaf node until it suceeds."""
        self.geometries = []
        while not self.geometries:
            try:
                # Obtain the new mesh patch
                origin_ijk, leaf_node = self.grid.leaf_nodes[self.idx]
                leaf_node_mesh = extract_mesh(leaf_node)
                leaf_node_mesh.scale(self.grid.voxel_size, center=np.zeros(3))
                leaf_node_mesh.paint_uniform_color(AIS_RED)

                # Get a coordinate_frame for visualization
                leaf_node_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=leaf_node.shape[0] * self.grid.voxel_size
                )

                # Compute the XYZ origin using the grid index and the voxel_size
                origin_xyz = self.grid.voxel_size * origin_ijk

                # Always update the reconstructed_mesh no matter what
                self._update_reconstruction(leaf_node_mesh, origin_xyz)

                if self.global_view:
                    model = self.model
                    leaf_node_mesh.translate(origin_xyz)
                    leaf_node_origin.translate(origin_xyz)
                    leaf_node_voxels = get_leaf_node_voxel_grid(
                        origin=origin_xyz,
                        shape=leaf_node.shape,
                        color=AIS_GREY,
                        voxel_size=self.grid.voxel_size,
                    )
                else:  # local_view
                    model = copy.deepcopy(self.model)
                    model.translate(-origin_xyz)
                    leaf_node_voxels = get_leaf_node_voxel_grid(
                        origin=np.zeros(3),
                        shape=leaf_node.shape,
                        color=AIS_GREY,
                        voxel_size=self.grid.voxel_size,
                    )
                # Update self.geometries cache
                self.geometries.append(leaf_node_mesh)
                if self.render_mesh:
                    self.geometries.append(model)
                if self.render_reconstruction and self.global_view:
                    self.geometries.append(self.reconstructed_mesh)
                if self.render_voxels:
                    self.geometries.append(leaf_node_voxels)
                    self.geometries.append(leaf_node_origin)
            except ValueError:
                self.idx = (self.idx + inc_idx) % self.leaf_count

        # When succeed, update the debug message
        print(
            "[{} view] visualizing #{}/{}".format(
                "Global" if self.global_view else "Local",
                self.idx,
                self.leaf_count,
            ),
            end="\r",
        )

    def _update_reconstruction(self, leaf_node_mesh, origin):
        leaf_node_mesh_t = copy.deepcopy(leaf_node_mesh)
        leaf_node_mesh_t.translate(origin)
        leaf_node_mesh_t.paint_uniform_color(AIS_RED)
        self.reconstructed_mesh += leaf_node_mesh_t

    def update_visualizer(self, reset_bounding_box=False):
        self.vis.clear_geometries()
        for geom in self.geometries:
            self.vis.add_geometry(geom, reset_bounding_box=reset_bounding_box)
        if reset_bounding_box and self.global_view and not self.render_mesh:
            self.vis.reset_view_point(True)
        self.vis.update_renderer()

    def reset_view(self, vis):
        """Reset the current view."""
        self.vis.reset_view_point(True)

    def toggle_view(self, vis):
        """Toggle between global/local view of the models."""
        self.global_view = not self.global_view
        # Store the current viewpoint for later
        current_camera = self.view_control.convert_to_pinhole_camera_parameters()
        self.update_geometries(inc_idx=0)
        self.update_visualizer(reset_bounding_box=False)
        self.reset_view(vis)
        if self.camera_params:
            self.view_control.convert_from_pinhole_camera_parameters(self.camera_params)
        # Cache the saved view for the next time we enter this toggle_view
        self.camera_params = current_camera

    def toggle_mesh(self, vis):
        """Add or remove the big mesh model."""
        self.render_mesh = not self.render_mesh
        self.update_geometries(inc_idx=0)
        self.update_visualizer(reset_bounding_box=False)

    def toggle_reconstructed_mesh(self, vis):
        """Add or remove the big mesh model."""
        self.render_reconstruction = not self.render_reconstruction
        self.update_geometries(inc_idx=0)
        self.update_visualizer(reset_bounding_box=False)

    def toggle_voxels(self, vis):
        """Add or remove the voxel grid used to visualize the leaf node."""
        self.render_voxels = not self.render_voxels
        self.update_geometries(inc_idx=0)
        self.update_visualizer(reset_bounding_box=False)

    def next_frame(self, vis):
        self.idx = (self.idx + 1) % self.leaf_count
        self.update_geometries(inc_idx=1)
        self.update_visualizer()
        return False

    def prev_frame(self, vis):
        self.idx = (self.idx - 1) % self.leaf_count
        self.update_geometries(inc_idx=-1)
        self.update_visualizer()
        return False

    def start_prev(self, vis):
        self.stop = False
        while not self.stop:
            self.next_frame(vis)
            time.sleep(self.sleep_time)

    def stop_prev(self, vis):
        self.stop = True

    def run_faster(self, vis):
        self.sleep_time = max(0.0, self.sleep_time - self.sleep_step)

    def run_slower(self, vis):
        self.sleep_time += self.sleep_step

    def set_render_options(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.render_options, key, value)

    def register_key_callback(self, key, callback):
        self.vis.register_key_callback(ord(str(key)), partial(callback))

    def _register_key_callbacks(self):
        self.register_key_callback("N", self.next_frame)
        self.register_key_callback("P", self.prev_frame)
        self.register_key_callback("S", self.start_prev)
        self.register_key_callback("X", self.stop_prev)
        self.register_key_callback("R", self.reset_view)
        self.register_key_callback("G", self.toggle_view)
        self.register_key_callback("M", self.toggle_mesh)
        self.register_key_callback("F", self.toggle_reconstructed_mesh)
        self.register_key_callback("V", self.toggle_voxels)
        self.register_key_callback("=", self.run_faster)  # means "+"
        self.register_key_callback("-", self.run_slower)

    @staticmethod
    def _print_help():
        print("N: next")
        print("P: previous")
        print("S: start")
        print("X: stop")
        print("V: toggle voxels")
        print("M: toggle mesh")
        print("F: toggle reconstructed mesh")
        print("G: toggle view")
        print("R: reset view")
        print("+: faster")
        print("-: slower")
        print("Q: quit")

    def run(self):
        self.vis.run()
        self.vis.destroy_window()
