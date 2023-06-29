pybind11_add_module(vdb_pybind vdb_pybind.cpp)
target_link_libraries(vdb_pybind PRIVATE OpenVDB::openvdb ${Boost_LIBRARIES})
