#include "PyVolume.h"

#include "Volume/initializeVolume.h"
void declare_volume_initializer(py::module& m) {
	using Class = dyno::VolumeInitializer;
	using Parent = dyno::PluginEntry;
	std::string pyclass_name = std::string("VolumeInitializer");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("instance", &Class::instance);
}

//#include "Volume/VolumeHelper.h"
//void declare_position_node(py::module& m) {
//	typedef unsigned long long int OcKey;
//	using Class = dyno::PositionNode;
//	std::string pyclass_name = std::string("PositionNode");
//	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
//		.def(py::init<>())
//		.def(py::init<int, OcKey>())
//		.def("__gt__", &Class::operator>)
//		.def_readwrite("surface_index", &Class::surface_index)
//		.def_readwrite("position_index", &Class::position_index);
//}

void pybind_volume(py::module& m)
{
	declare_volume_initializer(m);
	declare_sparse_marching_cubes<dyno::DataType3f>(m, "3f");
	declare_sparse_volume_clipper<dyno::DataType3f>(m, "3f");
	declare_volume_to_grid_cell<dyno::DataType3f>(m, "3f");
	declare_volume_to_triangle_set<dyno::DataType3f>(m, "3f");
	declare_volume<dyno::DataType3f>(m, "3f");
	declare_volume_bool<dyno::DataType3f>(m, "3f");
	declare_volume_generator<dyno::DataType3f>(m, "3f");
	declare_volume_octree<dyno::DataType3f>(m, "3f");
	//declare_volume_octree_boolean<dyno::DataType3f>(m, "3f");
	//declare_position_node(m);
	declare_volume_octree_generator<dyno::DataType3f>(m, "3f");
	//declare_volume_uniform_generator<dyno::DataType3f>(m, "3f");
}