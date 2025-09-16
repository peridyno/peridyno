#pragma once
#include "../PyCommon.h"


#include "Multiphysics/ComputeSurfaceLevelset.h"
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
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())

		.def("importParticleSystem", &Class::importParticleSystem, py::return_value_policy::reference)
		.def("getParticleSystem", &Class::getParticleSystem)

		.def("statePoints", &Class::statePoints, py::return_value_policy::reference)
		.def("stateLevelSet", &Class::stateLevelSet, py::return_value_policy::reference)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("state_grid_position", &Class::stateGridPoistion, py::return_value_policy::reference)
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
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		//DEF_VAR
		.def("varTangentialFriction", &Class::varTangentialFriction, py::return_value_policy::reference)
		.def("varNormalFriction", &Class::varNormalFriction, py::return_value_policy::reference)
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
	std::string pyclass_name = std::string("SdfSampler") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("importVolume", &Class::importVolume, py::return_value_policy::reference)
		.def("getVolume", &Class::getVolume);
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