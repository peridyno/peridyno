#pragma once
#include "../PyCommon.h"

#include "DualParticleSystem/DualParticleFluidSystem.h"
template <typename TDataType>
void declare_dual_particle_fluid_system(py::module& m, std::string typestr) {
	using Class = dyno::DualParticleFluidSystem<TDataType>;
	using Parent = dyno::ParticleFluid<TDataType>;
	std::string pyclass_name = std::string("DualParticleFluidSystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>DPFS(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	DPFS.def(py::init<>())
		.def(py::init<int>())
		.def("state_particle_attribute", &Class::stateParticleAttribute, py::return_value_policy::reference)
		.def("state_boundary_norm", &Class::stateBoundaryNorm, py::return_value_policy::reference)
		.def("state_virtual_position", &Class::stateVirtualPosition, py::return_value_policy::reference)
		.def("state_virtual_pointSet", &Class::stateVirtualPointSet, py::return_value_policy::reference)
		.def("var_virtual_particle_sampling_strategy", &Class::varVirtualParticleSamplingStrategy, py::return_value_policy::reference);

	py::enum_<typename Class::EVirtualParticleSamplingStrategy>(m, "EVirtualParticleSamplingStrategy")
		.value("ColocationStrategy", Class::EVirtualParticleSamplingStrategy::ColocationStrategy)
		.value("ParticleShiftingStrategy", Class::EVirtualParticleSamplingStrategy::ParticleShiftingStrategy)
		.value("SpatiallyAdaptiveStrategy", Class::EVirtualParticleSamplingStrategy::SpatiallyAdaptiveStrategy)
		.export_values();
}

void pybind_dual_particle_system(py::module& m);