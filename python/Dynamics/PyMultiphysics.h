#pragma once
#include "../PyCommon.h"

#include "Multiphysics/Module/PoissionDiskPositionShifting.h"
template <typename TDataType>
void declare_poission_disk_position_shifting(py::module& m, std::string typestr) {
	using Class = dyno::PoissionDiskPositionShifting<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;

	class PoissionDiskPositionShiftingTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PoissionDiskPositionShifting<TDataType>,
				compute
			);
		}
	};

	class PoissionDiskPositionShiftingPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("PoissionDiskPositionShifting") + typestr;
	py::class_<Class, Parent, PoissionDiskPositionShiftingTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inDelta", &Class::inDelta, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("outDensity", &Class::outDensity, py::return_value_policy::reference)

		.def("varIterationNumber", &Class::varIterationNumber, py::return_value_policy::reference)
		.def("varRestDensity", &Class::varRestDensity, py::return_value_policy::reference)
		.def("varKappa", &Class::varKappa, py::return_value_policy::reference)

		.def("updatePosition", &Class::updatePosition)
		// protected
		.def("compute", &PoissionDiskPositionShiftingPublicist::compute, py::return_value_policy::reference);
}

#include "Multiphysics/ComputeSurfaceLevelSet.h"
template <typename TDataType>
void declare_compute_surface_level_set(py::module& m, std::string typestr) {
	using Class = dyno::ComputeSurfaceLevelset<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("ComputeSurfaceLevelset") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("inPoints", &Class::inPoints, py::return_value_policy::reference)
		.def("inLevelSet", &Class::inLevelSet, py::return_value_policy::reference)
		.def("inGridSpacing", &Class::inGridSpacing, py::return_value_policy::reference);
}

#include "Multiphysics/ParticleSkinning.h"
template <typename TDataType>
void declare_particle_skinning(py::module& m, std::string typestr) {
	using Class = dyno::ParticleSkinning<TDataType>;
	using Parent = dyno::Node;

	class ParticleSkinningTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ParticleSkinning<TDataType>,
				resetStates
			);
		}

		void preUpdateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ParticleSkinning<TDataType>,
				preUpdateStates
			);
		}
	};

	class ParticleSkinningPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::preUpdateStates;
	};

	std::string pyclass_name = std::string("ParticleSkinning") + typestr;
	py::class_<Class, Parent, ParticleSkinningTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())

		.def("importParticleSystem", &Class::importParticleSystem, py::return_value_policy::reference)
		.def("getParticleSystem", &Class::getParticleSystem)

		.def("statePoints", &Class::statePoints, py::return_value_policy::reference)
		.def("stateLevelSet", &Class::stateLevelSet, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("stateGridPoistion", &Class::stateGridPoistion, py::return_value_policy::reference)
		.def("stateGridSpacing", &Class::stateGridSpacing, py::return_value_policy::reference)
		// protected
		.def("resetStates", &ParticleSkinningPublicist::resetStates, py::return_value_policy::reference)
		.def("preUpdateStates", &ParticleSkinningPublicist::preUpdateStates, py::return_value_policy::reference);
}

#include "Multiphysics/VolumeBoundary.h"
template <typename TDataType>
void declare_volume_boundary(py::module& m, std::string typestr) {
	using Class = dyno::VolumeBoundary<TDataType>;
	using Parent = dyno::Node;

	class VolumeBoundaryTrampoline : public Class
	{
	public:
		using Class::Class;

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::VolumeBoundary<TDataType>,
				updateStates
			);
		}

	};

	class VolumeBoundaryPublicist : public Class
	{
	public:
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("VolumeBoundary") + typestr;
	py::class_<Class, Parent, VolumeBoundaryTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		//DEF_VAR
		.def("varNormalFriction", &Class::varNormalFriction, py::return_value_policy::reference)
		.def("varTangentialFriction", &Class::varTangentialFriction, py::return_value_policy::reference)
		//DEF_NODE_PORTS
		.def("importVolumes", &Class::importVolumes, py::return_value_policy::reference)
		.def("getVolumes", &Class::getVolumes)
		.def("addVolume", &Class::addVolume)
		.def("removeVolume", &Class::removeVolume)

		.def("importParticleSystems", &Class::importParticleSystems, py::return_value_policy::reference)
		.def("getParticleSystems", &Class::getParticleSystems)
		.def("addParticleSystem", &Class::addParticleSystem)
		.def("removeParticleSystem", &Class::removeParticleSystem)

		.def("importTriangularSystems", &Class::importTriangularSystems, py::return_value_policy::reference)
		.def("getTriangularSystems", &Class::getTriangularSystems)
		.def("addTriangularSystem", &Class::addTriangularSystem)
		.def("removeTriangularSystem", &Class::removeTriangularSystem)

		.def("importTetrahedralSystems", &Class::importTetrahedralSystems, py::return_value_policy::reference)
		.def("getTetrahedralSystems", &Class::getTetrahedralSystems)
		.def("addTetrahedralSystem", &Class::addTetrahedralSystem)
		.def("removeTetrahedralSystem", &Class::removeTetrahedralSystem)
		//DEF_INSTANCE_STATE
		.def("stateTopology", &Class::stateTopology, py::return_value_policy::reference)
		// protected
		.def("updateStates", &VolumeBoundaryPublicist::updateStates, py::return_value_policy::reference);
}

#include "Multiphysics/SdfSampler.h"
template <typename TDataType>
void declare_sdf_sampler(py::module& m, std::string typestr) {
	using Class = dyno::SdfSampler<TDataType>;
	using Parent = dyno::Sampler<TDataType>;

	class SdfSamplerTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SdfSampler<TDataType>,
				resetStates
			);
		}

		bool validateInputs() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::SdfSampler<TDataType>,
				validateInputs
			);
		}

	};

	class SdfSamplerPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::validateInputs;
	};

	std::string pyclass_name = std::string("SdfSampler") + typestr;
	py::class_<Class, Parent, SdfSamplerTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)

		.def("importVolume", &Class::importVolume, py::return_value_policy::reference)
		.def("getVolume", &Class::getVolume)
		// protected
		.def("resetStates", &SdfSamplerPublicist::resetStates, py::return_value_policy::reference)
		.def("validateInputs", &SdfSamplerPublicist::validateInputs, py::return_value_policy::reference);
}

#include "Multiphysics/DevicePoissonDiskSampler.h"
template <typename TDataType>
void declare_device_poisson_disk_sampler(py::module& m, std::string typestr) {
	using Class = dyno::DevicePoissonDiskSampler<TDataType>;
	using Parent = dyno::SdfSampler<TDataType>;
	std::string pyclass_name = std::string("DevicePoissonDiskSampler") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("resetStates", &Class::resetStates)

		.def("statePosition", &Class::statePosition, py::return_value_policy::reference)

		.def("varDelta", &Class::varDelta, py::return_value_policy::reference)
		.def("stateNeighborLength", &Class::stateNeighborLength, py::return_value_policy::reference)
		.def("stateNeighborIds", &Class::stateNeighborIds, py::return_value_policy::reference)
		.def("outDensity", &Class::outDensity, py::return_value_policy::reference)
		.def("varConstraintDisable", &Class::varConstraintDisable, py::return_value_policy::reference)
		.def("varMaxIteration", &Class::varMaxIteration, py::return_value_policy::reference);
}


#include "Multiphysics/PoissonDiskSampler.h"
template <typename TDataType>
void declare_poisson_disk_sampling(py::module& m, std::string typestr) {
	using Class = dyno::PoissonDiskSampler<TDataType>;
	using Parent = dyno::SdfSampler<TDataType>;
	std::string pyclass_name = std::string("PoissonDiskSampler") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("resetStates", &Class::resetStates, py::return_value_policy::reference);
}

void pybind_multiphysics(py::module& m);