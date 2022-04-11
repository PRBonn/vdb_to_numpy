#pragma once

#include <openvdb/openvdb.h>

#include <Eigen/Core>
#include <tuple>
#include <vector>

namespace vdb_to_numpy {
std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3i>>
ExtractTriangleMesh(openvdb::FloatGrid::Ptr grid, float voxel_size);
}
