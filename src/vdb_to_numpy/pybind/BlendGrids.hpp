#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/tools/ChangeBackground.h>

#include <Eigen/Core>
#include <tuple>
#include <vector>

namespace vdb_to_numpy {
void inline BlendGrids(openvdb::FloatGrid::Ptr grid_a,
                       openvdb::FloatGrid::Ptr grid_b,
                       float eta) {
    grid_a->tree().combine(grid_b->tree(),
                           [=](const float& a, const float& b, float& result) {
                               result = eta * a + (1.0 - eta) * b;
                           });
}

openvdb::FloatGrid::Ptr inline NormalizeGrid(openvdb::FloatGrid::Ptr grid) {
    // First normalize all active values
    auto grid_copy = grid->deepCopy();
    for (auto iter = grid_copy->beginValueOn(); iter.test(); ++iter) {
        iter.setValue(iter.getValue() / grid->background());
    }

    // Now is time to change the background, since it's meaningless
    openvdb::tools::changeBackground(grid_copy->tree(),
                                     openvdb::FloatGrid::ValueType(1.0));

    // Spit the normalized grid
    return grid_copy;
}

}  // namespace vdb_to_numpy
