#pragma once
#include "../PyCommon.h"

#include "ParticleSystem/ParticleFluid.h"
#include "ParticleSystem/GhostParticles.h"

#include "ParticleSystem/Emitters/SquareEmitter.h"
#include "ParticleSystem/Emitters/CircularEmitter.h"

#include "Peridynamics/ElasticBody.h"
#include "Peridynamics/Module/LinearElasticitySolver.h"

#include "Peridynamics/TriangularSystem.h"

#include "RigidBody/RigidBody.h"

using Node = dyno::Node;
using NodePort = dyno::NodePort;

#include "ParticleSystem/Module/ApproximateImplicitViscosity.h"
template <typename TDataType>
void declare_approximate_implicit_viscosity(py::module& m, std::string typestr) {
	using Class = dyno::ApproximateImplicitViscosity<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("ApproximateImplicitViscosity") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>AIV(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	AIV.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("set_cross", &Class::SetCross)
		.def("var_viscosity", &Class::varViscosity, py::return_value_policy::reference)
		.def("in_neighbor_ids", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_attribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("var_lower_bound_viscosity", &Class::varLowerBoundViscosity, py::return_value_policy::reference)
		.def("var_cross_k", &Class::varCrossK, py::return_value_policy::reference)
		.def("var_cross_N", &Class::varCrossN, py::return_value_policy::reference)
		.def("var_fluid_type", &Class::varFluidType, py::return_value_policy::reference);

	py::enum_<typename Class::FluidType>(AIV, "FluidType")
		.value("NewtonianFluid", Class::FluidType::NewtonianFluid)
		.value("NonNewtonianFluid", Class::FluidType::NonNewtonianFluid)
		.export_values();
}

#include "ParticleSystem/Module/BoundaryConstraint.h"
template <typename TDataType>
void declare_boundary_constraint(py::module& m, std::string typestr) {
	using Class = dyno::BoundaryConstraint<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("BoundaryConstraint") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//.def("constrain", &Class::constrain)
		.def("load", &Class::load)
		.def("set_cube", &Class::setCube)
		.def("set_sphere", &Class::setSphere)
		.def("var_tangential_friction", &Class::varTangentialFriction, py::return_value_policy::reference)
		.def("var_normal_friction", &Class::varNormalFriction, py::return_value_policy::reference);
}

#include "ParticleSystem/Module/ImplicitViscosity.h"
template <typename TDataType>
void declare_implicit_viscosity(py::module& m, std::string typestr) {
	using Class = dyno::ImplicitViscosity<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;
	std::string pyclass_name = std::string("ImplicitViscosity") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_viscosity", &Class::varViscosity, py::return_value_policy::reference)
		.def("var_interation_number", &Class::varInterationNumber, py::return_value_policy::reference)
		.def("in_smoothing_length", &Class::inSmoothingLength, py::return_value_policy::reference)
		.def("in_sampling_distance", &Class::inSamplingDistance, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("in_neighbor_ids", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference);
}

#include "ParticleSystem/Module/ParticleApproximation.h"
template <typename TDataType>
void declare_particle_approximation(py::module& m, std::string typestr) {
	using Class = dyno::ParticleApproximation<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("ParticleApproximation") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>PA(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	PA.def(py::init<>())
		.def("compute", &Class::compute)
		.def("in_smoothing_length", &Class::inSmoothingLength, py::return_value_policy::reference)
		.def("in_sampling_distance", &Class::inSamplingDistance, py::return_value_policy::reference)
		.def("var_kernel_type", &Class::varKernelType, py::return_value_policy::reference);

	py::enum_<typename Class::EKernelType>(PA, "EKernelType")
		.value("KT_Smooth", Class::EKernelType::KT_Smooth)
		.value("KT_Spiky", Class::EKernelType::KT_Spiky)
		.export_values();
}

#include "ParticleSystem/Module/IterativeDensitySolver.h"
template <typename TDataType>
void declare_iterative_densitySolver(py::module& m, std::string typestr) {
	using Class = dyno::IterativeDensitySolver<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;
	std::string pyclass_name = std::string("IterativeDensitySolver") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("in_neighbor_ids", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("out_density", &Class::outDensity, py::return_value_policy::reference)
		.def("var_iteration_number", &Class::varIterationNumber, py::return_value_policy::reference)
		.def("var_rest_density", &Class::varRestDensity, py::return_value_policy::reference)
		.def("var_kappa", &Class::varKappa, py::return_value_policy::reference)
		.def("take_one_iteration", &Class::takeOneIteration)
		.def("update_velocity", &Class::updateVelocity);
}

#include "ParticleSystem/Module/LinearDamping.h"
template <typename TDataType>
void declare_linear_damping(py::module& m, std::string typestr) {
	using Class = dyno::LinearDamping<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("LinearDamping") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_damping_coefficient", &Class::varDampingCoefficient, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference);
}

#include "ParticleSystem/Module/ParticleIntegrator.h"
template <typename TDataType>
void declare_particle_integrator(py::module& m, std::string typestr) {
	using Class = dyno::ParticleIntegrator<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("ParticleIntegrator") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_attribute", &Class::inAttribute, py::return_value_policy::reference);
}

#include "Samplers/PoissonPlane.h"
template <typename TDataType>
void declare_poisson_plane(py::module& m, std::string typestr) {
	using Class = dyno::PoissonPlane<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("PoissonPlane") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("construct_grid", &Class::ConstructGrid)
		.def("collision_judge", &Class::collisionJudge)
		.def("var_sampling_distance", &Class::varSamplingDistance, py::return_value_policy::reference)
		.def("var_upper", &Class::varUpper, py::return_value_policy::reference)
		.def("var_lower", &Class::varLower, py::return_value_policy::reference)
		.def("compute", &Class::compute)
		.def("get_points", &Class::getPoints);
}

#include "ParticleSystem/Module/PositionBasedFluidModel.h"
template <typename TDataType>
void declare_position_based_fluid_model(py::module& m, std::string typestr) {
	using Class = dyno::PositionBasedFluidModel<TDataType>;
	using Parent = dyno::GroupModule;
	std::string pyclass_name = std::string("PositionBasedFluidModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_sampling_distance", &Class::inSamplingDistance, py::return_value_policy::reference)
		.def("in_smoothing_length", &Class::inSmoothingLength, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference);
}

#include "ParticleSystem/Module/ProjectionBasedFluidModel.h"
template <typename TDataType>
void declare_projection_based_fluid_model(py::module& m, std::string typestr) {
	using Class = dyno::ProjectionBasedFluidModel<TDataType>;
	using Parent = dyno::GroupModule;
	std::string pyclass_name = std::string("ProjectionBasedFluidModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_sampling_distance", &Class::inSamplingDistance, py::return_value_policy::reference)
		.def("in_smoothing_length", &Class::inSmoothingLength, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_normal", &Class::inNormal, py::return_value_policy::reference)
		.def("in_attribute", &Class::inAttribute, py::return_value_policy::reference);
}

#include "ParticleSystem/Module/SimpleVelocityConstraint.h"
template <typename TDataType>
void declare_simple_velocity_constraint(py::module& m, std::string typestr) {
	using Class = dyno::SimpleVelocityConstraint<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("SimpleVelocityConstraint") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("initialize", &Class::initialize)
		.def("resize_vector", &Class::resizeVector)
		/*.def("vis_value_set", &Class::visValueSet)*/
		//
		.def("vis_vector_set", &Class::visVectorSet)
		.def("simple_iter_num_set", &Class::SIMPLE_IterNumSet)
		//.def("set_cross", &Class:SetCross)
		.def("in_neighbor_ids", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_normal", &Class::inNormal, py::return_value_policy::reference)
		.def("in_attribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("in_smoothing_length", &Class::inSmoothingLength, py::return_value_policy::reference)
		.def("var_rest_density", &Class::varRestDensity, py::return_value_policy::reference)
		.def("in_sampling_distance", &Class::inSamplingDistance, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("var_viscosity", &Class::varViscosity, py::return_value_policy::reference)
		.def("var_simple_iteration_enable", &Class::varSimpleIterationEnable, py::return_value_policy::reference);
}

#include "ParticleSystem/Module/SummationDensity.h"
template <typename TDataType>
void declare_summation_density(py::module& m, std::string typestr) {
	using Class = dyno::SummationDensity<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;
	std::string pyclass_name = std::string("SummationDensity") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//.def("compute", &Class::compute)
		.def("var_rest_density", &Class::varRestDensity, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_other", &Class::inOther, py::return_value_policy::reference)
		.def("in_neighbor_ids", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("out_density", &Class::outDensity, py::return_value_policy::reference);
}

#include "ParticleSystem/Module/SurfaceEnergyForce.h"
template <typename TDataType>
void declare_surface_tension(py::module& m, std::string typestr) {
	using Class = dyno::SurfaceEnergyForce<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;
	std::string pyclass_name = std::string("SurfaceTension") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_position", &Class::inPosition)
		.def("in_velocity", &Class::inVelocity);
}

#include "ParticleSystem/Module/VariationalApproximateProjection.h"
template <typename TDataType>
void declare_variational_approximate_projection(py::module& m, std::string typestr) {
	using Class = dyno::VariationalApproximateProjection<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;
	std::string pyclass_name = std::string("VariationalApproximateProjection") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_rest_density", &Class::varRestDensity, py::return_value_policy::reference)
		.def("in_sampling_distance", &Class::inSamplingDistance, py::return_value_policy::reference)
		.def("in_smoothing_length", &Class::inSmoothingLength, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("in_neighbor_ids", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_normal", &Class::inNormal, py::return_value_policy::reference)
		.def("in_attribute", &Class::inAttribute, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_particle_emitter(py::module& m, std::string typestr) {
	using Class = dyno::ParticleEmitter<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("ParticleEmitter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("size_of_particles", &Class::sizeOfParticles)
		.def("get_positions", &Class::getPositions)
		.def("get_velocities", &Class::getVelocities)
		.def("get_node_type", &Class::getNodeType)
		.def("var_velocity_magnitude", &Class::varVelocityMagnitude, py::return_value_policy::reference)
		.def("var_sampling_distance", &Class::varSamplingDistance, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_particle_system(py::module& m, std::string typestr) {
	using Class = dyno::ParticleSystem<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("ParticleSystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)
		.def("get_dt", &Class::getDt)
		//DEF_ARRAY_STATE
		.def("state_position", &Class::statePosition, py::return_value_policy::reference)
		.def("state_velocity", &Class::stateVelocity, py::return_value_policy::reference)
		.def("state_point_set", &Class::statePointSet, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_circular_emitter(py::module& m, std::string typestr) {
	using Class = dyno::CircularEmitter<TDataType>;
	using Parent = dyno::ParticleEmitter<TDataType>;
	std::string pyclass_name = std::string("CircularEmitter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_radius", &Class::varRadius, py::return_value_policy::reference)
		.def("state_outline", &Class::stateOutline, py::return_value_policy::reference);
}

#include "Samplers/Sampler.h"
template <typename TDataType>
void declare_sampler(py::module& m, std::string typestr) {
	using Class = dyno::Sampler<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("Sampler") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//DEF_INSTANCE_STATE
		.def("state_point_set", &Class::statePointSet, py::return_value_policy::reference);
}

#include "Samplers/ShapeSampler.h"
template <typename TDataType>
void declare_cube_sampler(py::module& m, std::string typestr) {
	using Class = dyno::ShapeSampler<TDataType>;
	using Parent = dyno::Sampler<TDataType>;
	std::string pyclass_name = std::string("CubeSampler") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//DEF_VAR
		.def("var_sampling_distance", &Class::varSamplingDistance, py::return_value_policy::reference)
		//DEF_VAR_IN
		.def("import_shape", &Class::importShape, py::return_value_policy::reference);
}

#include "ParticleSystem/GhostFluid.h"
template <typename TDataType>
void declare_ghost_fluid(py::module& m, std::string typestr) {
	using Class = dyno::GhostFluid<TDataType>;
	using Parent = dyno::ParticleSystem<TDataType>;
	std::string pyclass_name = std::string("GhostFluid") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("state_attribute_merged", &Class::stateAttributeMerged, py::return_value_policy::reference)
		.def("state_normal_merged", &Class::stateNormalMerged, py::return_value_policy::reference)
		.def("import_initial_states", &Class::importInitialStates, py::return_value_policy::reference)
		.def("import_boundary_particles", &Class::importBoundaryParticles, py::return_value_policy::reference)
		.def("get_boundary_particles", &Class::getBoundaryParticles);
}

template <typename TDataType>
void declare_ghost_particlesm(py::module& m, std::string typestr) {
	using Class = dyno::GhostParticles<TDataType>;
	using Parent = dyno::ParticleSystem<TDataType>;
	std::string pyclass_name = std::string("GhostParticles") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("state_normal", &Class::stateNormal, py::return_value_policy::reference)
		.def("state_attribute", &Class::stateAttribute, py::return_value_policy::reference);
}

#include "ParticleSystem/MakeParticleSystem.h"
template <typename TDataType>
void declare_make_particle_system(py::module& m, std::string typestr) {
	using Class = dyno::MakeParticleSystem<TDataType>;
	using Parent = dyno::ParticleSystem<TDataType>;
	std::string pyclass_name = std::string("MakeParticleSystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//DEF_VAR
		.def("var_initial_velocity", &Class::varInitialVelocity, py::return_value_policy::reference)
		//DEF_INSTANCE_IN
		.def("in_points", &Class::inPoints, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_particle_fluid(py::module& m, std::string typestr) {
	using Class = dyno::ParticleFluid<TDataType>;
	using Parent = dyno::ParticleSystem<TDataType>;
	std::string pyclass_name = std::string("ParticleFluid") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//DEF_VAR
		.def("var_reshuffle_particles", &Class::varReshuffleParticles, py::return_value_policy::reference)
		//DEF_NODE_PORTS
		.def("import_particle_emitters", &Class::importParticleEmitters, py::return_value_policy::reference)
		.def("get_particle_emitters", &Class::getParticleEmitters)
		.def("add_particle_emitter", &Class::addParticleEmitter)
		.def("remove_particle_emitter", &Class::removeParticleEmitter)
		.def("import_initial_states", &Class::importInitialStates, py::return_value_policy::reference)
		.def("get_initial_states", &Class::getInitialStates)
		.def("add_initial_state", &Class::addInitialState)
		.def("remove_initial_state", &Class::removeInitialState);
}

#include "ParticleSystem/ParticleSystemHelper.h"
template <typename TDataType>
void declare_particle_system_helper(py::module& m, std::string typestr) {
	using Class = dyno::ParticleSystemHelper<TDataType>;
	std::string pyclass_name = std::string("ParticleSystemHelper") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("calculate_morton_code", &Class::calculateMortonCode)
		.def("reorder_particles", &Class::reorderParticles);
}

#include "Multiphysics/PoissonDiskSampling.h"
template <typename TDataType>
void declare_poisson_disk_sampling(py::module& m, std::string typestr) {
	using Class = dyno::PoissonDiskSampling<TDataType>;
	using Parent = dyno::Sampler<TDataType>;
	std::string pyclass_name = std::string("PoissonDiskSampling") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("construct_grid", &Class::ConstructGrid)
		.def("collision_judge_2d", &Class::collisionJudge2D)
		.def("collision_judge", &Class::collisionJudge)
		//.def("var_sampling_distance", &Class::varSamplingDistance, py::return_value_policy::reference)
		//.def("var_dimension", &Class::varDimension, py::return_value_policy::reference)
		.def("var_box_a", &Class::varBox_a, py::return_value_policy::reference)
		.def("var_box_b", &Class::varBox_b, py::return_value_policy::reference)
		.def("var_sdf_file_name", &Class::varSdfFileName, py::return_value_policy::reference)
		.def("lerp", &Class::lerp)
		.def("get_distance_from_sdf", &Class::getDistanceFromSDF)
		.def("get_sdf", &Class::getSDF)
		.def("get_one_point_inside_sdf", &Class::getOnePointInsideSDF);
}

#include "ParticleSystem/Emitters/PoissonEmitter.h"
template <typename TDataType>
void declare_poisson_emitter(py::module& m, std::string typestr) {
	using Class = dyno::PoissonEmitter<TDataType>;
	using Parent = dyno::ParticleEmitter<TDataType>;
	std::string pyclass_name = std::string("PoissonEmitter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>PE(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	PE.def(py::init<>())
		.def("var_width", &Class::varWidth, py::return_value_policy::reference)
		.def("var_height", &Class::varHeight, py::return_value_policy::reference)
		.def("var_delay_start", &Class::varDelayStart, py::return_value_policy::reference)
		.def("state_outline", &Class::stateOutline, py::return_value_policy::reference)
		.def("var_emitter_shape", &Class::varEmitterShape, py::return_value_policy::reference);

	py::enum_<typename Class::EmitterShape>(PE, "EmitterShape")
		.value("Square", Class::EmitterShape::Square)
		.value("Round", Class::EmitterShape::Round)
		.export_values();
}

template <typename TDataType>
void declare_square_emitter(py::module& m, std::string typestr) {
	using Class = dyno::SquareEmitter<TDataType>;
	using Parent = dyno::ParticleEmitter<TDataType>;
	std::string pyclass_name = std::string("SquareEmitter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_width", &Class::varWidth, py::return_value_policy::reference)
		.def("var_height", &Class::varHeight, py::return_value_policy::reference)
		.def("state_outline", &Class::stateOutline, py::return_value_policy::reference);
}

void declare_particle_system_initializer(py::module& m);

void declare_particle_type(py::module& m);

void pybind_particle_system(py::module& m);
