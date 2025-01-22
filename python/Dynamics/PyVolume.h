#include "../PyCommon.h"

#include "Volume/Module/AdaptiveVolumeToTriangleSet.h"
template <typename TDataType>
void declare_adaptive_volume_to_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::AdaptiveVolumeToTriangleSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("AdaptiveVolumeToTriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_iso_value", &Class::varIsoValue, py::return_value_policy::reference)
		.def("io_volume", &Class::ioVolume, py::return_value_policy::reference)
		.def("out_triangle_set", &Class::outTriangleSet, py::return_value_policy::reference);
}

#include "Volume/Module/FastMarchingMethodGPU.h"
template <typename TDataType>
void declare_fast_marching_method_GPU(py::module& m, std::string typestr) {
	using Class = dyno::FastMarchingMethodGPU<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("FastMarchingMethodGPU") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_spacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("var_bool_type", &Class::varBoolType, py::return_value_policy::reference)
		.def("var_marching_number", &Class::varMarchingNumber, py::return_value_policy::reference)
		.def("in_level_set_a", &Class::inLevelSetA, py::return_value_policy::reference)
		.def("in_level_set_b", &Class::inLevelSetB, py::return_value_policy::reference)
		.def("out_level_set", &Class::outLevelSet, py::return_value_policy::reference);
}

#include "Volume/Module/FastSweepingMethod.h"
template <typename TDataType>
void declare_fast_sweeping_method(py::module& m, std::string typestr) {
	using Class = dyno::FastSweepingMethod<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("FastSweepingMethod") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_spacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("var_padding", &Class::varPadding, py::return_value_policy::reference)

		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("out_level_set", &Class::outLevelSet, py::return_value_policy::reference);
}

#include "Volume/Module/FastSweepingMethodGPU.h"
template <typename TDataType>
void declare_fast_sweeping_method_GPU(py::module& m, std::string typestr) {
	using Class = dyno::FastSweepingMethodGPU<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("FastSweepingMethodGPU") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_spacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("var_padding", &Class::varPadding, py::return_value_policy::reference)
		.def("var_pass_number", &Class::varPassNumber, py::return_value_policy::reference)

		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("out_level_set", &Class::outLevelSet, py::return_value_policy::reference);
}

#include "Volume/Module/MarchingCubesHelper.h"
template <typename TDataType>
void declare_marching_cubes_helper(py::module& m, std::string typestr) {
	using Class = dyno::MarchingCubesHelper<TDataType>;
	std::string pyclass_name = std::string("MarchingCubesHelper") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("reconstruct_sdf", &Class::reconstructSDF)
		.def("count_vertice_number", &Class::countVerticeNumber)
		.def("construct_triangles", &Class::constructTriangles)
		.def("count_vertice_number_for_clipper", &Class::countVerticeNumberForClipper)
		.def("construct_tiangles_for_clipper", &Class::constructTrianglesForClipper)
		.def("count_vertice_number_for_octree", &Class::countVerticeNumberForOctree)
		.def("construct_triangles_for_octree", &Class::constructTrianglesForOctree)
		.def("count_vertice_number_for_octree_clipper", &Class::countVerticeNumberForOctreeClipper)
		.def("construct_triangles_for_octree_clipper", &Class::constructTrianglesForOctreeClipper);
}

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

#include "Volume/Volume.h"
template <typename TDataType>
void declare_volume(py::module& m, std::string typestr) {
	using Class = dyno::Volume<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("Volume") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)
		.def("state_level_set", &Class::stateLevelSet, py::return_value_policy::reference);
}

#include "Volume/BasicShapeToVolume.h"
template <typename TDataType>
void declare_basic_shape_to_volume(py::module& m, std::string typestr) {
	using Class = dyno::BasicShapeToVolume<TDataType>;
	using Parent = dyno::Volume<TDataType>;
	std::string pyclass_name = std::string("BasicShapeToVolume") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_inerted", &Class::varInerted, py::return_value_policy::reference)
		.def("var_grid_spacing", &Class::varGridSpacing, py::return_value_policy::reference)
		.def("get_shape", &Class::getShape)
		.def("import_shape", &Class::importShape, py::return_value_policy::reference);
}

#include "Volume/MarchingCubes.h"
template <typename TDataType>
void declare_marching_cubes(py::module& m, std::string typestr) {
	using Class = dyno::MarchingCubes<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("MarchingCubes") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)
		.def("var_iso_value", &Class::varIsoValue, py::return_value_policy::reference)
		.def("var_grid_spacing", &Class::varGridSpacing, py::return_value_policy::reference)
		.def("in_level_set", &Class::inLevelSet, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference);
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
		.def("get_sparse_volume", &Class::getSparseVolume)
		.def("import_sparse_volume", &Class::importSparseVolume, py::return_value_policy::reference);
}

#include "Volume/VolumeBoolean.h"
template <typename TDataType>
void declare_volume_bool(py::module& m, std::string typestr) {
	using Class = dyno::VolumeBoolean<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("VolumeBool") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>VB(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	VB.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)

		.def("var_spacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("var_padding", &Class::varPadding, py::return_value_policy::reference)
		.def("var_bool_type", &Class::varBoolType, py::return_value_policy::reference)

		.def("in_a", &Class::inA, py::return_value_policy::reference)
		.def("in_b", &Class::inB, py::return_value_policy::reference)
		.def("out_level_set", &Class::outLevelSet, py::return_value_policy::reference);
}

#include "Volume/VolumeClipper.h"
template <typename TDataType>
void declare_volume_clipper(py::module& m, std::string typestr) {
	using Class = dyno::VolumeClipper<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("VolumeClipper") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)

		.def("var_translation", &Class::varTranslation, py::return_value_policy::reference)
		.def("var_rotation", &Class::varRotation, py::return_value_policy::reference)
		.def("state_field", &Class::stateField, py::return_value_policy::reference)
		.def("state_plane", &Class::statePlane, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("in_level_set", &Class::inLevelSet, py::return_value_policy::reference);
}

#include "Volume/VolumeGenerator.h"
template <typename TDataType>
void declare_volume_generator(py::module& m, std::string typestr) {
	using Class = dyno::VolumeGenerator<TDataType>;
	using Parent = dyno::Volume<TDataType>;
	std::string pyclass_name = std::string("VolumeGenerator") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_spacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("var_padding", &Class::varPadding, py::return_value_policy::reference)

		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("out_level_set", &Class::outLevelSet, py::return_value_policy::reference);
}

#include "Volume/VolumeLoader.h"
template <typename TDataType>
void declare_volume_loader(py::module& m, std::string typestr) {
	using Class = dyno::VolumeLoader<TDataType>;
	using Parent = dyno::Volume<TDataType>;
	std::string pyclass_name = std::string("VolumeLoader") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_file_name", &Class::varFileName, py::return_value_policy::reference);
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
		.def("state_sdf_topolopy", &Class::stateSDFTopology, py::return_value_policy::reference)
		.def_readwrite("m_object", &Class::m_object)
		.def_readwrite("m_normal", &Class::m_normal);
}

#include "Volume/VolumeOctreeBoolean.h"
template <typename TDataType>
void declare_volume_octree_boolean(py::module& m, std::string typestr) {
	using Class = dyno::VolumeOctreeBoolean<TDataType>;
	using Parent = dyno::VolumeOctree<TDataType>;
	std::string pyclass_name = std::string("VolumeOctreeBoolean") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>VOB(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	VOB.def(py::init<>())
		.def("get_octree_a", &Class::getOctreeA)
		.def("import_octree_a", &Class::importOctreeA, py::return_value_policy::reference)
		.def("get_octree_b", &Class::getOctreeB)
		.def("import_octree_b", &Class::importOctreeB, py::return_value_policy::reference)
		.def("var_min_dx", &Class::varMinDx, py::return_value_policy::reference)
		.def("var_boolean_type", &Class::varBooleanType, py::return_value_policy::reference);

	py::enum_<typename Class::BooleanOperation>(VOB, "BooleanOperation")
		.value("UNION_SET", Class::BooleanOperation::UNION_SET)
		.value("INTERSECTION_SET", Class::BooleanOperation::INTERSECTION_SET)
		.value("SUBTRACTION_SET", Class::BooleanOperation::SUBTRACTION_SET);
}

#include "Volume/VolumeOctreeGenerator.h"
template <typename TDataType>
void declare_volume_octree_generator(py::module& m, std::string typestr) {
	using Class = dyno::VolumeOctreeGenerator<TDataType>;
	using Parent = dyno::VolumeOctree<TDataType>;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	std::string pyclass_name = std::string("VolumeOctreeGenerator") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("load", py::overload_cast<std::string>(&Class::load))
		.def("load", py::overload_cast<std::string, Coord, Real, Coord>(&Class::load))
		.def("lower_bound", &Class::lowerBound)
		.def("upper_bound", &Class::upperBound)
		.def("in_triangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("var_spacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("var_padding", &Class::varPadding, py::return_value_policy::reference)
		.def("var_aabb_padding", &Class::varAABBPadding, py::return_value_policy::reference)
		.def("var_forward_vector", &Class::varForwardVector, py::return_value_policy::reference);
}

void pybind_volume(py::module& m);