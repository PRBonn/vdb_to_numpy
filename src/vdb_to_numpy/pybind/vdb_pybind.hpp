// pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// Boost Python
#include <boost/python.hpp>

namespace vdb_to_numpy {

namespace py = pybind11;

/// Integer Coordinates = [10, 22, 33]
template <typename LeafNodeType, typename ValueType = int32_t>
py::array_t<ValueType> LeafNodeOriginToNumpy(LeafNodeType& leaf) {
    const std::size_t itemsize = sizeof(ValueType);
    const std::string format = py::format_descriptor<ValueType>::value;
    const std::size_t ndim = 1;
    const std::vector<std::size_t> shape = {3};
    const std::vector<std::size_t> strides = {itemsize};

    auto origin = leaf.origin();
    return py::array_t<ValueType>(py::buffer_info(
        origin.data(),  // Pointer to the underlying storage
        itemsize,       // Size of individual items in bytes
        format,         // set to format_descriptor<T>::format()
        ndim,           // Number of dimensions
        shape,          // Shape of the tensor (1 entry per dimension)
        strides         // Number of bytes between adjacent entries
        ));
}

template <typename LeafNodeType,
          typename ValueType = typename LeafNodeType::ValueType>
py::array_t<ValueType> LeafNodeBufferToNumpy(LeafNodeType& leaf) {
    const std::size_t itemsize = sizeof(ValueType);
    const std::string format = py::format_descriptor<ValueType>::value;
    const std::size_t ndim = LeafNodeType::LOG2DIM;
    const std::vector<std::size_t> shape = {
        LeafNodeType::DIM,  // 8
        LeafNodeType::DIM,  // 8
        LeafNodeType::DIM   // 8
    };
    const std::vector<std::size_t> strides = {
        LeafNodeType::DIM * LeafNodeType::DIM * itemsize,  // 256
        LeafNodeType::DIM * itemsize,                      // 32
        itemsize,                                          // 4
    };

    auto buffer = leaf.buffer();
    return py::array_t<ValueType>(py::buffer_info(
        buffer.data(),  // Pointer to the underlying storage
        itemsize,       // Size of individual items in bytes
        format,         // set to format_descriptor<T>::format()
        ndim,           // Number of dimensions
        shape,          // Shape of the tensor (1 entry per dimension)
        strides         // Number of bytes between adjacent entries
        ));
}

template <typename LeafNodeType>
py::tuple ExtractLeafNode(LeafNodeType& leaf) {
    return py::make_tuple(LeafNodeOriginToNumpy(leaf),
                          LeafNodeBufferToNumpy(leaf));
}

template <typename GridType>
typename GridType::Ptr getGridFromPyObject(py::object py_obj) {
    boost::python::extract<typename GridType::Ptr> x(py_obj.ptr());
    if (x.check()) {
        return x();
    } else {
        std::string error_string("GridType: ");
        error_string += py::str(py_obj.get_type());
        error_string += " not supported";
        throw std::invalid_argument(error_string);
    }
}

template <typename GridType>
py::list ExtractLeafNodes(py::object py_obj) {
    auto grid = getGridFromPyObject<GridType>(py_obj);
    py::list leaf_nodes;
    for (auto iter = grid->tree().cbeginLeaf(); iter; ++iter) {
        auto leaf = *iter;
        auto leaf_node = ExtractLeafNode(leaf);
        leaf_nodes.append(leaf_node);
    }
    return leaf_nodes;
}

}  // namespace vdb_to_numpy
