#include "../PyCommon.h"

#include "Volume/Module/FastMarchingMethodGPU.h"
template <typename TDataType>
void declare_fast_marching_method_GPU(py::module& m, std::string typestr) {
	using Class = dyno::FastMarchingMethodGPU<TDataType>;
	using Parent = dyno::ComputeModule;

	class FastMarchingMethodGPUTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::FastMarchingMethodGPU<TDataType>,
				compute,
				);
		}

	};

	class FastMarchingMethodGPUPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("FastMarchingMethodGPU") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("varBoolType", &Class::varBoolType, py::return_value_policy::reference)
		.def("varMarchingNumber", &Class::varMarchingNumber, py::return_value_policy::reference)

		.def("inLevelSetA", &Class::inLevelSetA, py::return_value_policy::reference)
		.def("inLevelSetB", &Class::inLevelSetB, py::return_value_policy::reference)
		.def("outLevelSet", &Class::outLevelSet, py::return_value_policy::reference)
		// protected
		.def("compute", &FastMarchingMethodGPUPublicist::compute, py::return_value_policy::reference);
}

#include "Volume/Module/FastSweepingMethod.h"
template <typename TDataType>
void declare_fast_sweeping_method(py::module& m, std::string typestr) {
	using Class = dyno::FastSweepingMethod<TDataType>;
	using Parent = dyno::ComputeModule;

	class FastSweepingMethodTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::FastSweepingMethod<TDataType>,
				compute,
				);
		}

	};

	class FastSweepingMethodPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("FastSweepingMethod") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("varPadding", &Class::varPadding, py::return_value_policy::reference)

		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("outLevelSet", &Class::outLevelSet, py::return_value_policy::reference)
		// protected
		.def("compute", &FastSweepingMethodPublicist::compute, py::return_value_policy::reference);
}

#include "Volume/Module/FastSweepingMethodGPU.h"
template <typename TDataType>
void declare_fast_sweeping_method_GPU(py::module& m, std::string typestr) {
	using Class = dyno::FastSweepingMethodGPU<TDataType>;
	using Parent = dyno::ComputeModule;

	class FastSweepingMethodGPUTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::FastSweepingMethodGPU<TDataType>,
				compute,
				);
		}

	};

	class FastSweepingMethodGPUPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("FastSweepingMethodGPU") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("varPadding", &Class::varPadding, py::return_value_policy::reference)
		.def("varPassNumber", &Class::varPassNumber, py::return_value_policy::reference)

		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("outLevelSet", &Class::outLevelSet, py::return_value_policy::reference)
		// protected
		.def("compute", &FastSweepingMethodGPUPublicist::compute, py::return_value_policy::reference);
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

	class VolumeToTriangleSetTrampoline : public Class
	{
	public:
		using Class::Class;

		bool apply() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::VolumeToTriangleSet<TDataType>,
				apply,
				);
		}

	};

	class VolumeToTriangleSetPublicist : public Class
	{
	public:
		using Class::apply;
	};

	std::string pyclass_name = std::string("VolumeToTriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varIsoValue", &Class::varIsoValue, py::return_value_policy::reference)
		.def("inVolume", &Class::inVolume, py::return_value_policy::reference)
		.def("outTriangleSet", &Class::outTriangleSet, py::return_value_policy::reference)
		// protected
		.def("apply", &VolumeToTriangleSetPublicist::apply, py::return_value_policy::reference);
}

#include "Volume/Module/MultiscaleFastIterativeMethod.h"
template <typename TDataType>
void declare_multiscale_fast_iterative_method(py::module& m, std::string typestr) {
	using Class = dyno::MultiscaleFastIterativeMethod<TDataType>;
	using Parent = dyno::ComputeModule;

	class MultiscaleFastIterativeMethodTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::MultiscaleFastIterativeMethod<TDataType>,
				compute,
				);
		}

	};

	class MultiscaleFastIterativeMethodPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("MultiscaleFastIterativeMethod") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varSpacing", &Class::varSpacing,py::return_value_policy::reference)
		.def("varPadding", &Class::varPadding, py::return_value_policy::reference)
		.def("varVCircle", &Class::varVCircle, py::return_value_policy::reference)

		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("outLevelSet", &Class::outLevelSet, py::return_value_policy::reference)
		// protected
		.def("compute", &MultiscaleFastIterativeMethodPublicist::compute, py::return_value_policy::reference);
}

#include "Volume/Module/MultiscaleFastIterativeMethodForBoolean.h"
template <typename TDataType>
void declare_multiscale_fast_iterative_for_boolean_method(py::module& m, std::string typestr) {
	using Class = dyno::MultiscaleFastIterativeMethodForBoolean<TDataType>;
	using Parent = dyno::ComputeModule;

	class MultiscaleFastIterativeMethodForBooleanTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::MultiscaleFastIterativeMethodForBoolean<TDataType>,
				compute,
				);
		}

	};

	class MultiscaleFastIterativeMethodForBooleanPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("MultiscaleFastIterativeMethodForBoolean") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varBoolType", &Class::varBoolType, py::return_value_policy::reference)
		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("varPadding", &Class::varPadding, py::return_value_policy::reference)
		.def("varVCircle", &Class::varVCircle, py::return_value_policy::reference)

		.def("inLevelSetA", &Class::inLevelSetA, py::return_value_policy::reference)
		.def("inLevelSetB", &Class::inLevelSetB, py::return_value_policy::reference)
		.def("outLevelSet", &Class::outLevelSet, py::return_value_policy::reference)
		// protected
		.def("compute", &MultiscaleFastIterativeMethodForBooleanPublicist::compute, py::return_value_policy::reference);
}


#include "Volume/Module/LevelSetConstructionAndBooleanHelper.h"
template <typename TDataType>
void declare_level_set_construction_and_boolean_helper(py::module& m, std::string typestr) {
	using Class = dyno::LevelSetConstructionAndBooleanHelper<TDataType>;
	std::string pyclass_name = std::string("LevelSetConstructionAndBooleanHelper") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("initialFromTriangle", &Class::initialFromTriangle)
		.def("fastIterative", &Class::fastIterative)
		.def("initialForBoolean", &Class::initialForBoolean);
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

	class BasicShapeToVolumeTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::BasicShapeToVolume<TDataType>,
				resetStates,
				);
		}

		bool validateInputs() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::BasicShapeToVolume<TDataType>,
				validateInputs,
				);
		}

	};

	class BasicShapeToVolumePublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::validateInputs;
	};

	std::string pyclass_name = std::string("BasicShapeToVolume") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varInerted", &Class::varInerted, py::return_value_policy::reference)
		.def("varGridSpacing", &Class::varGridSpacing, py::return_value_policy::reference)

		.def("getShape", &Class::getShape)
		.def("importShape", &Class::importShape, py::return_value_policy::reference)
		// protected
		.def("resetStates", &BasicShapeToVolumePublicist::resetStates, py::return_value_policy::reference)
		.def("validateInputs", &BasicShapeToVolumePublicist::validateInputs, py::return_value_policy::reference);
}

#include "Volume/MarchingCubes.h"
template <typename TDataType>
void declare_marching_cubes(py::module& m, std::string typestr) {
	using Class = dyno::MarchingCubes<TDataType>;
	using Parent = dyno::Node;

	class MarchingCubesTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::MarchingCubes<TDataType>,
				resetStates,
				);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::MarchingCubes<TDataType>,
				updateStates,
				);
		}

	};

	class MarchingCubesPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("MarchingCubes") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)

		.def("varIsoValue", &Class::varIsoValue, py::return_value_policy::reference)
		.def("varGridSpacing", &Class::varGridSpacing, py::return_value_policy::reference)
		.def("inLevelSet", &Class::inLevelSet, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &MarchingCubesPublicist::resetStates, py::return_value_policy::reference)
		.def("updateStates", &MarchingCubesPublicist::updateStates, py::return_value_policy::reference);
}

#include "Volume/VolumeBoolean.h"
template <typename TDataType>
void declare_volume_bool(py::module& m, std::string typestr) {
	using Class = dyno::VolumeBoolean<TDataType>;
	using Parent = dyno::Node;

	class VolumeBooleanTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::VolumeBoolean<TDataType>,
				resetStates,
				);
		}

	};

	class VolumeBooleanPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("VolumeBool") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>VB(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	VB.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)

		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("varPadding", &Class::varPadding, py::return_value_policy::reference)
		.def("varBoolType", &Class::varBoolType, py::return_value_policy::reference)

		.def("inA", &Class::inA, py::return_value_policy::reference)
		.def("inB", &Class::inB, py::return_value_policy::reference)
		.def("outLevelSet", &Class::outLevelSet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &VolumeBooleanPublicist::resetStates, py::return_value_policy::reference);
}

#include "Volume/VolumeClipper.h"
template <typename TDataType>
void declare_volume_clipper(py::module& m, std::string typestr) {
	using Class = dyno::VolumeClipper<TDataType>;
	using Parent = dyno::Node;

	class VolumeClipperTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::VolumeClipper<TDataType>,
				resetStates,
				);
		}

	};

	class VolumeClipperPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("VolumeClipper") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)

		.def("stateField", &Class::stateField, py::return_value_policy::reference)
		.def("statePlane", &Class::statePlane, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("inLevelSet", &Class::inLevelSet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &VolumeClipperPublicist::resetStates, py::return_value_policy::reference);
}

#include "Volume/VolumeGenerator.h"
template <typename TDataType>
void declare_volume_generator(py::module& m, std::string typestr) {
	using Class = dyno::VolumeGenerator<TDataType>;
	using Parent = dyno::Volume<TDataType>;

	class VolumeGeneratorTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::VolumeGenerator<TDataType>,
				resetStates,
				);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::VolumeGenerator<TDataType>,
				updateStates,
				);
		}

	};

	class VolumeGeneratorPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("VolumeGenerator") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("varPadding", &Class::varPadding, py::return_value_policy::reference)

		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("outLevelSet", &Class::outLevelSet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &VolumeGeneratorPublicist::resetStates, py::return_value_policy::reference)
		.def("updateStates", &VolumeGeneratorPublicist::updateStates, py::return_value_policy::reference);
}

#include "Volume/VolumeLoader.h"
template <typename TDataType>
void declare_volume_loader(py::module& m, std::string typestr) {
	using Class = dyno::VolumeLoader<TDataType>;
	using Parent = dyno::Volume<TDataType>;

	class VolumeLoaderTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::VolumeLoader<TDataType>,
				resetStates,
				);
		}

	};

	class VolumeLoaderPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("VolumeLoader") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varFileName", &Class::varFileName, py::return_value_policy::reference)
		// protected
		.def("resetStates", &VolumeLoaderPublicist::resetStates, py::return_value_policy::reference);
}

void pybind_volume(py::module& m);