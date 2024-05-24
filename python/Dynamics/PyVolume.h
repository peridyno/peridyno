#include "../PyCommon.h"

#include "Volume/Module/VolumeToGridCell.h"
template <typename TDataType>
void declare_volume_to_grid_cell(py::module& m, std::string typestr) {
	using Class = dyno::VolumeToGridCell<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("VolumeToGridCell") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_iso_value", &Class::varIsoValue, py::return_value_policy::reference)
		.def("in_volume", &Class::inVolume, py::return_value_policy::reference)
		.def("out_grid_cell", &Class::outGridCell, py::return_value_policy::reference);
}

#include "Volume/Module/VolumeToTriangleSet.h"
template <typename TDataType>
void declare_volume_to_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::VolumeToTriangleSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("VolumeToTriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_iso_value", &Class::varIsoValue, py::return_value_policy::reference)
		.def("io_volume", &Class::ioVolume, py::return_value_policy::reference)
		.def("out_triangle_set", &Class::outTriangleSet, py::return_value_policy::reference);
}

#include "Volume/SparseMarchingCubes.h"
template <typename TDataType>
void declare_sparse_marching_cubes(py::module& m, std::string typestr) {
	using Class = dyno::SparseMarchingCubes<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("SparseMarchingCubes") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_iso_value", &Class::varIsoValue, py::return_value_policy::reference)
		.def("get_sparse_volume", &Class::getSparseVolume)
		.def("import_sparse_volume", &Class::importSparseVolume, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference);
}

#include "Volume/SparseVolumeClipper.h"
template <typename TDataType>
void declare_sparse_volume_clipper(py::module& m, std::string typestr) {
	using Class = dyno::SparseVolumeClipper<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("SparseVolumeClipper") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_translation", &Class::varTranslation, py::return_value_policy::reference)
		.def("var_rotation", &Class::varRotation, py::return_value_policy::reference)
		.def("state_field", &Class::stateField, py::return_value_policy::reference)
		.def("state_vertices", &Class::stateVertices, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("get_sparse_volume", &Class::getSparseVolume, py::return_value_policy::reference)
		.def("import_sparse_volume", &Class::importSparseVolume, py::return_value_policy::reference);
}

#include "Volume/Volume.h"
template <typename TDataType>
void declare_volume(py::module& m, std::string typestr) {
	using Class = dyno::Volume<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("Volume") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("state_sdf", &Class::stateSDF, py::return_value_policy::reference)
		.def("state_topology", &Class::stateTopology, py::return_value_policy::reference);
}

#include "Volume/VolumeBool.h"
template <typename TDataType>
void declare_volume_bool(py::module& m, std::string typestr) {
	using Class = dyno::VolumeBool<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("VolumeBool") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>VB(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	VB.def(py::init<>())
		.def("in_a", &Class::inA, py::return_value_policy::reference)
		.def("in_b", &Class::inB, py::return_value_policy::reference)
		.def("out_sdf", &Class::outSDF, py::return_value_policy::reference)
		.def("in_spacing", &Class::inSpacing, py::return_value_policy::reference)
		.def("in_padding", &Class::inPadding, py::return_value_policy::reference)
		.def("var_bool_type", &Class::varBoolType, py::return_value_policy::reference);

	py::enum_<typename Class::BoolType>(VB, "BoolType")
		.value("Intersect", Class::BoolType::Intersect)
		.value("Union", Class::BoolType::Union)
		.value("Minus", Class::BoolType::Minus);
}

#include "Volume/VolumeGenerator.h"
template <typename TDataType>
void declare_volume_generator(py::module& m, std::string typestr) {
	using Class = dyno::VolumeGenerator<TDataType>;
	using Parent = dyno::Volume<TDataType>;
	std::string pyclass_name = std::string("VolumeGenerator") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("load_closed_surface", &Class::loadClosedSurface)
		.def("load", &Class::load)
		.def("in_triangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("out_gen_sdf", &Class::outGenSDF, py::return_value_policy::reference)
		.def("in_spacing", &Class::inSpacing, py::return_value_policy::reference)
		.def("in_padding", &Class::inPadding, py::return_value_policy::reference)
		.def("make_level_set", &Class::makeLevelSet);
}

#include "Volume/VolumeIntersect.h"
template <typename TDataType>
void declare_volume_intersect(py::module& m, std::string typestr) {
	using Class = dyno::VolumeIntersect<TDataType>;
	using Parent = dyno::Volume<TDataType>;
	std::string pyclass_name = std::string("VolumeIntersect") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_a", &Class::getA)
		.def("import_a", &Class::importA, py::return_value_policy::reference)
		.def("get_b", &Class::getB)
		.def("import_b", &Class::importB, py::return_value_policy::reference);
}

#include "Volume/VolumeMinus.h"
template <typename TDataType>
void declare_volume_minus(py::module& m, std::string typestr) {
	using Class = dyno::VolumeMinus<TDataType>;
	using Parent = dyno::Volume<TDataType>;
	std::string pyclass_name = std::string("VolumeMinus") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_a", &Class::getA)
		.def("import_a", &Class::importA, py::return_value_policy::reference)
		.def("get_b", &Class::getB)
		.def("import_b", &Class::importB, py::return_value_policy::reference);
}

#include "Volume/VolumeOctree.h"
template <typename TDataType>
void declare_volume_octree(py::module& m, std::string typestr) {
	using Class = dyno::VolumeOctree<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("VolumeOctree") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_inverted", &Class::varInverted, py::return_value_policy::reference)
		.def("var_level_number", &Class::varLevelNumber, py::return_value_policy::reference)
		.def("state_sdf_topolopy", &Class::stateSDFTopology, py::return_value_policy::reference);
}

//#include "Volume/VolumeOctreeBoolean.h"
//template <typename TDataType>
//void declare_volume_octree_boolean(py::module& m, std::string typestr) {
//	using Class = dyno::VolumeOctreeBoolean<TDataType>;
//	using Parent = dyno::VolumeOctree<TDataType>;
//	std::string pyclass_name = std::string("VolumeOctreeBoolean") + typestr;
//	py::class_<Class, Parent, std::shared_ptr<Class>>VOB(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
//	VOB.def(py::init<>())
//		.def("get_octree_a", &Class::getOctreeA)
//		.def("import_octree_a", &Class::importOctreeA, py::return_value_policy::reference)
//		.def("get_octree_b", &Class::getOctreeB)
//		.def("import_octree_b", &Class::importOctreeB, py::return_value_policy::reference)
//		.def("var_min_dx", &Class::varMinDx, py::return_value_policy::reference)
//		.def("var_boolean_type", &Class::varBooleanType, py::return_value_policy::reference);
//
//	py::enum_<typename Class::BooleanOperation>(VOB, "BooleanOperation")
//		.value("UNION_SET", Class::BooleanOperation::UNION_SET)
//		.value("INTERSECTION_SET", Class::BooleanOperation::INTERSECTION_SET)
//		.value("SUBTRACTION_SET", Class::BooleanOperation::SUBTRACTION_SET);
//}

#include "Volume/VolumeOctreeGenerator.h"
template <typename TDataType>
void declare_volume_octree_generator(py::module& m, std::string typestr) {
	using Class = dyno::VolumeOctreeGenerator<TDataType>;
	using Parent = dyno::VolumeOctree<TDataType>;
	std::string pyclass_name = std::string("VolumeOctreeGenerator") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//.def("load", &Class::load)
		.def("lower_bound", &Class::lowerBound)
		.def("upper_bound", &Class::upperBound)
		.def("in_triangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("var_spacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("var_padding", &Class::varPadding, py::return_value_policy::reference)
		.def("var_aabb_padding", &Class::varAABBPadding, py::return_value_policy::reference)
		.def("var_forward_vector", &Class::varForwardVector, py::return_value_policy::reference);
}

//#include "Volume/VolumeUniformGenerator.h"
//template <typename TDataType>
//void declare_volume_uniform_generator(py::module& m, std::string typestr) {
//	using Class = dyno::VolumeUniformGenerator<TDataType>;
//	using Parent = dyno::VolumeOctree<TDataType>;
//	std::string pyclass_name = std::string("VolumeUniformGenerator") + typestr;
//	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
//		.def(py::init<>())
//		.def("load", &Class::load)
//		.def("reset_states", &Class::resetStates)
//		.def("update_states", &Class::updateStates)
//		.def("init_parameter", &Class::initParameter)
//		.def("dx", &Class::Dx)
//		.def("nx", &Class::nx)
//		.def("origin", &Class::Origin)
//		.def("ny", &Class::ny)
//		.def("nz", &Class::nz)
//		.def("get_sign_distance", &Class::getSignDistance);
//}

#include "Volume/VolumeUnion.h"
template <typename TDataType>
void declare_volume_union(py::module& m, std::string typestr) {
	using Class = dyno::VolumeUnion<TDataType>;
	using Parent = dyno::Volume<TDataType>;
	std::string pyclass_name = std::string("VolumeUnion") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_a", &Class::getA)
		.def("import_a", &Class::importA, py::return_value_policy::reference)
		.def("get_b", &Class::getB)
		.def("import_b", &Class::importB, py::return_value_policy::reference);
}


void declare_volume_initializer(py::module& m);

void pybind_volume(py::module& m);