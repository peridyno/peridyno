#include "../PyCommon.h"

#include "Volume/Module/FastMarchingMethodGPU.h"
template <typename TDataType>
void declare_fast_marching_method_GPU(py::module& m, std::string typestr) {
	using Class = dyno::FastMarchingMethodGPU<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("FastMarchingMethodGPU") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("varBoolType", &Class::varBoolType, py::return_value_policy::reference)
		.def("varMarchingNumber", &Class::varMarchingNumber, py::return_value_policy::reference)

		.def("inLevelSetA", &Class::inLevelSetA, py::return_value_policy::reference)
		.def("inLevelSetB", &Class::inLevelSetB, py::return_value_policy::reference)
		.def("outLevelSet", &Class::outLevelSet, py::return_value_policy::reference);
}

#include "Volume/Module/FastSweepingMethod.h"
template <typename TDataType>
void declare_fast_sweeping_method(py::module& m, std::string typestr) {
	using Class = dyno::FastSweepingMethod<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("FastSweepingMethod") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("varPadding", &Class::varPadding, py::return_value_policy::reference)

		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("outLevelSet", &Class::outLevelSet, py::return_value_policy::reference);
}

#include "Volume/Module/FastSweepingMethodGPU.h"
template <typename TDataType>
void declare_fast_sweeping_method_GPU(py::module& m, std::string typestr) {
	using Class = dyno::FastSweepingMethodGPU<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("FastSweepingMethodGPU") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("varPadding", &Class::varPadding, py::return_value_policy::reference)
		.def("varPassNumber", &Class::varPassNumber, py::return_value_policy::reference)

		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("outLevelSet", &Class::outLevelSet, py::return_value_policy::reference);
}

#include "Volume/Module/MarchingCubesHelper.h"
template <typename TDataType>
void declare_marching_cubes_helper(py::module& m, std::string typestr) {
	using Class = dyno::MarchingCubesHelper<TDataType>;
	std::string pyclass_name = std::string("MarchingCubesHelper") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("reconstructSDF", &Class::reconstructSDF)
		.def("countVerticeNumber", &Class::countVerticeNumber)
		.def("constructTriangles", &Class::constructTriangles)
		.def("countVerticeNumberForClipper", &Class::countVerticeNumberForClipper)
		.def("constructTrianglesForClipper", &Class::constructTrianglesForClipper)
		.def("countVerticeNumberForOctree", &Class::countVerticeNumberForOctree)
		.def("constructTrianglesForOctree", &Class::constructTrianglesForOctree)
		.def("countVerticeNumberForOctreeClipper", &Class::countVerticeNumberForOctreeClipper)
		.def("constructTrianglesForOctreeClipper", &Class::constructTrianglesForOctreeClipper);
}

#include "Volume/Module/VolumeToTriangleSet.h"
template <typename TDataType>
void declare_volume_to_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::VolumeToTriangleSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("VolumeToTriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varIsoValue", &Class::varIsoValue, py::return_value_policy::reference)
		.def("inVolume", &Class::inVolume, py::return_value_policy::reference)
		.def("outTriangleSet", &Class::outTriangleSet, py::return_value_policy::reference);
}

#include "Volume/Volume.h"
template <typename TDataType>
void declare_volume(py::module& m, std::string typestr) {
	using Class = dyno::Volume<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("Volume") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		.def("stateLevelSet", &Class::stateLevelSet, py::return_value_policy::reference);
}

#include "Volume/BasicShapeToVolume.h"
template <typename TDataType>
void declare_basic_shape_to_volume(py::module& m, std::string typestr) {
	using Class = dyno::BasicShapeToVolume<TDataType>;
	using Parent = dyno::Volume<TDataType>;
	std::string pyclass_name = std::string("BasicShapeToVolume") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varInerted", &Class::varInerted, py::return_value_policy::reference)
		.def("varGridSpacing", &Class::varGridSpacing, py::return_value_policy::reference)

		.def("getShape", &Class::getShape)
		.def("importShape", &Class::importShape, py::return_value_policy::reference);
}

#include "Volume/MarchingCubes.h"
template <typename TDataType>
void declare_marching_cubes(py::module& m, std::string typestr) {
	using Class = dyno::MarchingCubes<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("MarchingCubes") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)

		.def("varIsoValue", &Class::varIsoValue, py::return_value_policy::reference)
		.def("varGridSpacing", &Class::varGridSpacing, py::return_value_policy::reference)
		.def("inLevelSet", &Class::inLevelSet, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference);
}



#include "Volume/VolumeBoolean.h"
template <typename TDataType>
void declare_volume_bool(py::module& m, std::string typestr) {
	using Class = dyno::VolumeBoolean<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("VolumeBool") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>VB(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	VB.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)

		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("varPadding", &Class::varPadding, py::return_value_policy::reference)
		.def("varBoolType", &Class::varBoolType, py::return_value_policy::reference)

		.def("inA", &Class::inA, py::return_value_policy::reference)
		.def("inB", &Class::inB, py::return_value_policy::reference)
		.def("outLevelSet", &Class::outLevelSet, py::return_value_policy::reference);
}

#include "Volume/VolumeClipper.h"
template <typename TDataType>
void declare_volume_clipper(py::module& m, std::string typestr) {
	using Class = dyno::VolumeClipper<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("VolumeClipper") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)

		.def("stateField", &Class::stateField, py::return_value_policy::reference)
		.def("statePlane", &Class::statePlane, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("inLevelSet", &Class::inLevelSet, py::return_value_policy::reference);
}

#include "Volume/VolumeGenerator.h"
template <typename TDataType>
void declare_volume_generator(py::module& m, std::string typestr) {
	using Class = dyno::VolumeGenerator<TDataType>;
	using Parent = dyno::Volume<TDataType>;
	std::string pyclass_name = std::string("VolumeGenerator") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("varPadding", &Class::varPadding, py::return_value_policy::reference)

		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("outLevelSet", &Class::outLevelSet, py::return_value_policy::reference);
}

#include "Volume/VolumeLoader.h"
template <typename TDataType>
void declare_volume_loader(py::module& m, std::string typestr) {
	using Class = dyno::VolumeLoader<TDataType>;
	using Parent = dyno::Volume<TDataType>;
	std::string pyclass_name = std::string("VolumeLoader") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varFileName", &Class::varFileName, py::return_value_policy::reference);
}

void pybind_volume(py::module& m);