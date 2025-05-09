#pragma once
#include "../PyCommon.h"

#include "DualParticleSystem/DualParticleFluid.h"
template <typename TDataType>
void declare_dual_particle_fluid_system(py::module& m, std::string typestr) {
	using Class = dyno::DualParticleFluid<TDataType>;
	using Parent = dyno::ParticleFluid<TDataType>;
	std::string pyclass_name = std::string("DualParticleFluid") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>DPFS(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	DPFS.def(py::init<>())
		.def(py::init<int>())
		.def("state_particle_attribute", &Class::stateParticleAttribute, py::return_value_policy::reference)
		.def("state_boundary_norm", &Class::stateBoundaryNorm, py::return_value_policy::reference)
		.def("state_virtual_position", &Class::stateVirtualPosition, py::return_value_policy::reference)
		.def("state_virtual_pointSet", &Class::stateVirtualPointSet, py::return_value_policy::reference)
		.def("var_virtual_particle_sampling_strategy", &Class::varVirtualParticleSamplingStrategy, py::return_value_policy::reference);

	py::enum_<typename Class::EVirtualParticleSamplingStrategy>(DPFS, "EVirtualParticleSamplingStrategy")
		.value("ColocationStrategy", Class::EVirtualParticleSamplingStrategy::ColocationStrategy)
		.value("ParticleShiftingStrategy", Class::EVirtualParticleSamplingStrategy::ParticleShiftingStrategy)
		.value("SpatiallyAdaptiveStrategy", Class::EVirtualParticleSamplingStrategy::SpatiallyAdaptiveStrategy);
}

#include "DualParticleSystem/Module/DualParticleIsphModule.h"
template <typename TDataType>
void declare_dual_particle_isph_module(py::module& m, std::string typestr) {
	using Class = dyno::DualParticleIsphModule<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("DualParticleIsphModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("var_rest_density", &Class::varRestDensity, py::return_value_policy::reference)
		.def("var_sampling_distance", &Class::varSamplingDistance, py::return_value_policy::reference)
		.def("var_smoothing_length", &Class::varSmoothingLength, py::return_value_policy::reference)
		.def("var_ppe_smoothing_length", &Class::varPpeSmoothingLength, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("in_r_position", &Class::inRPosition, py::return_value_policy::reference)
		.def("in_v_position", &Class::inVPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_particle_attribute", &Class::inParticleAttribute, py::return_value_policy::reference)
		.def("in_boundary_norm", &Class::inBoundaryNorm, py::return_value_policy::reference)
		.def("in_neighbor_ids", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("in_vr_neighbor_ids", &Class::inVRNeighborIds, py::return_value_policy::reference)
		.def("in_rv_neighbor_ids", &Class::inRVNeighborIds, py::return_value_policy::reference)
		.def("in_vv_neighbor_ids", &Class::inVVNeighborIds, py::return_value_policy::reference)
		.def("out_virtual_bool", &Class::outVirtualBool, py::return_value_policy::reference)
		.def("out_virtual_weight", &Class::outVirtualWeight, py::return_value_policy::reference)
		.def("var_residual_threshold", &Class::varResidualThreshold, py::return_value_policy::reference);
}

#include "DualParticleSystem/Module/EnergyAnalysis.h"
template <typename TDataType>
void declare_energy_analyish(py::module& m, std::string typestr)
{
	using Class = dyno::EnergyAnalysis<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("EnergyAnalysis") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_neighbor_ids", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("initialize_impl", &Class::initializeImpl)
		.def("set_name_prefix", &Class::setNamePrefix)
		.def("set_output_path", &Class::setOutputPath);
}

#include "DualParticleSystem/Module/PaticleUniformAnalysis.h"
template <typename TDataType>
void declare_paticle_uniform_analysis(py::module& m, std::string typestr)
{
	using Class = dyno::PaticleUniformAnalysis<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("PaticleUniformAnalysis") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_neighbor_ids", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("constrain", &Class::constrain)
		.def("initialize_impl", &Class::initializeImpl)
		.def("set_name_prefix", &Class::setNamePrefix)
		.def("set_output_path", &Class::setOutputPath)
		.def("in_smoothing_length", &Class::inSmoothingLength, py::return_value_policy::reference)
		.def("in_sampling_distance", &Class::inSamplingDistance, py::return_value_policy::reference);
}

#include "DualParticleSystem/Module/VirtualParticleGenerator.h"
template <typename TDataType>
void declare_virtual_particle_generator(py::module& m, std::string typestr) {
	using Class = dyno::VirtualParticleGenerator<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("VirtualParticleGenerator") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("out_virtual_particles", &Class::outVirtualParticles, py::return_value_policy::reference);
}

#include "DualParticleSystem/Module/VirtualColocationStrategy.h"
template <typename TDataType>
void declare_virtual_colocation_strategy(py::module& m, std::string typestr) {
	using Class = dyno::VirtualColocationStrategy<TDataType>;
	using Parent = dyno::VirtualParticleGenerator<TDataType>;
	std::string pyclass_name = std::string("VirtualColocationStrategy") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("in_r_position", &Class::inRPosition, py::return_value_policy::reference);
}

#include "DualParticleSystem/Module/VirtualParticleShiftingStrategy.h"
template <typename TDataType>
void declare_virtual_particle_shifting_strategy(py::module& m, std::string typestr) {
	using Class = dyno::VirtualParticleShiftingStrategy<TDataType>;
	using Parent = dyno::VirtualParticleGenerator<TDataType>;
	std::string pyclass_name = std::string("VirtualParticleShiftingStrategy") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("take_one_iteration", &Class::takeOneIteration)
		.def("vector_resize", &Class::VectorResize)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("var_iteration_number", &Class::varIterationNumber, py::return_value_policy::reference)
		.def("var_rest_density", &Class::varRestDensity, py::return_value_policy::reference)
		.def("var_sampling_distance", &Class::varSamplingDistance, py::return_value_policy::reference)
		.def("var_smoothing_length", &Class::varSmoothingLength, py::return_value_policy::reference)
		.def("in_frame_number", &Class::inFrameNumber, py::return_value_policy::reference)
		.def("in_r_position", &Class::inRPosition, py::return_value_policy::reference)
		.def("out_vv_neighbor_ids", &Class::outVVNeighborIds, py::return_value_policy::reference)
		.def("out_v_density", &Class::outVDensity, py::return_value_policy::reference);
}

#include "DualParticleSystem/Module/VirtualSpatiallyAdaptiveStrategy.h"
template <typename TDataType>
void declare_virtual_spatially_adaptive_strategy(py::module& m, std::string typestr) {
	using Class = dyno::VirtualSpatiallyAdaptiveStrategy<TDataType>;
	using Parent = dyno::VirtualParticleGenerator<TDataType>;
	std::string pyclass_name = std::string("VirtualSpatiallyAdaptiveStrategy") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>VSAS(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());

	py::enum_<typename Class::CandidatePointCount>(VSAS, "CandidatePointCount")
		.value("neighbors_8", Class::CandidatePointCount::neighbors_8)
		.value("neighbors_27", Class::CandidatePointCount::neighbors_27)
		.value("neighbors_33", Class::CandidatePointCount::neighbors_33)
		.value("neighbors_125", Class::CandidatePointCount::neighbors_125)
		.export_values();

	VSAS.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("set_hash_grid_size", &Class::setHashGridSize)
		.def("var_candidate_point_count", &Class::varCandidatePointCount, py::return_value_policy::reference)
		.def("var_rest_density", &Class::varRestDensity, py::return_value_policy::reference)
		.def("var_sampling_distance", &Class::varSamplingDistance, py::return_value_policy::reference)
		.def("in_r_position", &Class::inRPosition, py::return_value_policy::reference);
}

void pybind_dual_particle_system(py::module& m);