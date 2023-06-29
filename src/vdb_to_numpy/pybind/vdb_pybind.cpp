#include "vdb_pybind.hpp"

#include <openvdb/openvdb.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <Eigen/Core>
#include <boost/python.hpp>
#include <memory>
#include <vector>

#include "BlendGrids.hpp"

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>)
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3i>)

namespace py = pybind11;
using namespace py::literals;

namespace pybind11::detail {
template <>
struct type_caster<openvdb::FloatGrid::Ptr> {
public:
    /// Converts Python to C++
    bool load(handle src, bool) {
        PyObject* source = src.ptr();
        boost::python::extract<typename openvdb::FloatGrid::Ptr> grid_ptr(
            source);
        if (!grid_ptr.check()) return false;
        value = grid_ptr();
        return (value && !PyErr_Occurred());
    }

    /// Converts from C++ to Python
    static handle cast(openvdb::FloatGrid::Ptr src,
                       return_value_policy,
                       handle) {
        if (!src) return none().inc_ref();
        py::module::import("pyopenvdb").attr("FloatGrid");
        boost::python::object obj = boost::python::object(src);
        py::object out = reinterpret_borrow<py::object>(obj.ptr());
        return out.release();
    }

    PYBIND11_TYPE_CASTER(openvdb::FloatGrid::Ptr, _("pyopenvdb.FloatGrid"));
};
}  // namespace pybind11::detail

namespace vdb_to_numpy {

PYBIND11_MODULE(vdb_pybind, m) {
    py::bind_vector<std::vector<Eigen::Vector3d>>(m, "_VectorEigen3d");
    py::bind_vector<std::vector<Eigen::Vector3i>>(m, "_VectorEigen3i");
    m.def("extract_leaf_nodes", &ExtractLeafNodes<openvdb::FloatGrid>,
          "Extract all the leaf nodes from a openvdb::FloatGrid into a list "
          "containing the numpy arrarys. Each element corresponds to a dense "
          "(8, 8, 8) grid containing the floating point values of the leaf "
          "node.",
          "grid"_a);
    m.def("_blend_grids", &BlendGrids, "grid_a"_a, "grid_b"_a, "eta"_a);
    m.def("_normalize_grid", &NormalizeGrid, "grid"_a);
}
}  // namespace vdb_to_numpy
