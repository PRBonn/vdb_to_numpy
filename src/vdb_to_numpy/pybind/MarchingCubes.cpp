#include "MarchingCubes.h"

#include <openvdb/Types.h>
#include <openvdb/openvdb.h>

#include <Eigen/Core>
#include <memory>
#include <tuple>
#include <vector>

#include "MarchingCubesConst.h"

namespace openvdb {
static const openvdb::Coord shift[8] = {
    openvdb::Coord(0, 0, 0), openvdb::Coord(1, 0, 0), openvdb::Coord(1, 1, 0),
    openvdb::Coord(0, 1, 0), openvdb::Coord(0, 0, 1), openvdb::Coord(1, 0, 1),
    openvdb::Coord(1, 1, 1), openvdb::Coord(0, 1, 1),
};
}

// Taken from <open3d/utility/Eigen.h>
namespace {
template <typename T>
struct hash_eigen {
    std::size_t operator()(T const& matrix) const {
        size_t seed = 0;
        for (int i = 0; i < (int)matrix.size(); i++) {
            auto elem = *(matrix.data() + i);
            seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 +
                    (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};
}  // namespace

namespace vdb_to_numpy {
std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3i>>
ExtractTriangleMesh(openvdb::FloatGrid::Ptr grid, float voxel_size) {
    // implementation of marching cubes, based on Open3D
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> triangles;

    double half_voxel_length = voxel_size * 0.5;
    // Map of "edge_index = (x, y, z, 0) + edge_shift" to "global vertex index"
    std::unordered_map<
        Eigen::Vector4i, int, hash_eigen<Eigen::Vector4i>, std::equal_to<>,
        Eigen::aligned_allocator<std::pair<const Eigen::Vector4i, int>>>
        edgeindex_to_vertexindex;
    int edge_to_index[12];

    auto grid_acc = grid->getAccessor();
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        int cube_index = 0;
        float f[8];
        const openvdb::Coord& voxel = iter.getCoord();
        const int32_t x = voxel.x();
        const int32_t y = voxel.y();
        const int32_t z = voxel.z();
        for (int i = 0; i < 8; i++) {
            openvdb::Coord idx = voxel + openvdb::shift[i];
            f[i] = grid_acc.getValue(idx);
            if (f[i] < 0.0f) {
                cube_index |= (1 << i);
            }
        }
        if (cube_index == 0 || cube_index == 255) {
            continue;
        }
        for (int i = 0; i < 12; i++) {
            if ((edge_table[cube_index] & (1 << i)) != 0) {
                Eigen::Vector4i edge_index =
                    Eigen::Vector4i(x, y, z, 0) + edge_shift[i];
                if (edgeindex_to_vertexindex.find(edge_index) ==
                    edgeindex_to_vertexindex.end()) {
                    edge_to_index[i] = (int)vertices.size();
                    edgeindex_to_vertexindex[edge_index] = (int)vertices.size();
                    Eigen::Vector3d pt(
                        half_voxel_length + voxel_size * edge_index(0),
                        half_voxel_length + voxel_size * edge_index(1),
                        half_voxel_length + voxel_size * edge_index(2));
                    double f0 = std::abs((double)f[edge_to_vert[i][0]]);
                    double f1 = std::abs((double)f[edge_to_vert[i][1]]);
                    pt(edge_index(3)) += f0 * voxel_size / (f0 + f1);
                    vertices.push_back(pt /* + origin_*/);
                } else {
                    edge_to_index[i] =
                        edgeindex_to_vertexindex.find(edge_index)->second;
                }
            }
        }
        for (int i = 0; tri_table[cube_index][i] != -1; i += 3) {
            triangles.emplace_back(edge_to_index[tri_table[cube_index][i]],
                                   edge_to_index[tri_table[cube_index][i + 2]],
                                   edge_to_index[tri_table[cube_index][i + 1]]);
        }
    }
    return std::make_tuple(vertices, triangles);
}

}  // namespace vdb_to_numpy
