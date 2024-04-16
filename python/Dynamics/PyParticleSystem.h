#pragma once
#include "../PyCommon.h"

#include "ParticleSystem/SquareEmitter.h"
#include "ParticleSystem/ParticleFluid.h"
#include "ParticleSystem/CircularEmitter.h"
#include "ParticleSystem/GhostParticles.h"
#include "Peridynamics/ElasticBody.h"
#include "Peridynamics/Module/LinearElasticitySolver.h"

#include "RigidBody/RigidBody.h"

using Node = dyno::Node;
using NodePort = dyno::NodePort;

// class: CircularEmitter   - for example_2: Qt_WaterPouring
template <typename TDataType>
void declare_circular_emitter(py::module& m, std::string typestr) {
	using Class = dyno::CircularEmitter<TDataType>;
	using Parent = dyno::ParticleEmitter<TDataType>;
	std::string pyclass_name = std::string("CircularEmitter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_location", &Class::varLocation, py::return_value_policy::reference);
}

#include "ParticleSystem/CubeSampler.h"
template <typename TDataType>
void declare_cube_sampler(py::module& m, std::string typestr) {
	using Class = dyno::CubeSampler<TDataType>;
	using Parent = dyno::Sampler<TDataType>;
	std::string pyclass_name = std::string("CubeSampler") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//DEF_VAR
		.def("var_sampling_distance", &Class::varSamplingDistance, py::return_value_policy::reference)
		//DEF_VAR_IN
		.def("in_cube", &Class::inCube, py::return_value_policy::reference);
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
void declare_particle_emitter(py::module& m, std::string typestr) {
	using Class = dyno::ParticleEmitter<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("ParticleEmitter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_location", &Class::varLocation, py::return_value_policy::reference);
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
		.def("import_particles_emitters", &Class::importParticleEmitters, py::return_value_policy::reference)
		.def("get_particle_emitters", &Class::getParticleEmitters)
		.def("add_particle_emitter", &Class::addParticleEmitter)
		.def("remove_particle_emitter", &Class::removeParticleEmitter)
		.def("import_initial_states", &Class::importInitialStates, py::return_value_policy::reference)
		.def("get_initial_states", &Class::getInitialStates)
		.def("add_initial_state", &Class::addInitialState)
		.def("remove_initial_state", &Class::removeInitialState);
}

template <typename TDataType>
void declare_particle_system(py::module& m, std::string typestr) {
	using Class = dyno::ParticleSystem<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("ParticleSystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("load_particles", (void (Class::*)(Class::Coord lo, Class::Coord hi, Class::Real distance)) & Class::loadParticles)
		.def("get_node_type", &Class::getNodeType)
		.def("get_dt", &Class::getDt)
		//DEF_ARRAY_STATE
		.def("state_position", &Class::statePosition, py::return_value_policy::reference)
		.def("state_velocity", &Class::stateVelocity, py::return_value_policy::reference)
		.def("state_force", &Class::stateForce, py::return_value_policy::reference)
		.def("state_point_set", &Class::statePointSet, py::return_value_policy::reference);
}

#include "ParticleSystem/Sampler.h"
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

template <typename TDataType>
void declare_particle_emitter_square(py::module& m, std::string typestr) {
	using Class = dyno::SquareEmitter<TDataType>;
	using Parent = dyno::ParticleEmitter<TDataType>;
	std::string pyclass_name = std::string("SquareEmitter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

#include "ParticleSystem/StaticBoundary.h"
template <typename TDataType>
void declare_static_boundary(py::module& m, std::string typestr) {
	using Class = dyno::StaticBoundary<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("StaticBoundary") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//public
		//.def("load_sdf", &Class::loadSDF)
		//.def("load_cube", &Class::loadCube)
		.def("load_sphere", &Class::loadShpere)
		.def("translate", &Class::translate)
		.def("scale", &Class::scale)
		.def("reset_states", &Class::resetStates)
		//DEF_VAR
		.def("var_tangential_friction", &Class::varTangentialFriction, py::return_value_policy::reference)
		.def("var_normal_friction", &Class::varNormalFriction, py::return_value_policy::reference)
		.def("var_cube_vertex_lo", &Class::varCubeVertex_lo, py::return_value_policy::reference)
		.def("var_cube_vertex_hi", &Class::varCubeVertex_hi, py::return_value_policy::reference)
		.def("var_file_name", &Class::varFileName, py::return_value_policy::reference)
		//DEF_NODE_PORTS
		.def("get_particle_system", &Class::getParticleSystems)
		.def("add_particle_system", &Class::addParticleSystem)
		.def("remove_particle_system", &Class::removeParticleSystem)
		.def("import_particle_systems", &Class::importParticleSystems, py::return_value_policy::reference)

		.def("get_rigid_body", &Class::getRigidBodys)
		.def("add_rigid_body", &Class::addRigidBody)
		.def("remove_rigid_body", &Class::removeRigidBody)
		.def("import_rigid_bodys", &Class::importRigidBodys, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_multi_node_port(py::module& m, std::string typestr) {
	using Class = dyno::MultipleNodePort<TDataType>;
	using Parent = dyno::NodePort;
	std::string pyclass_name = std::string("MultipleNodePort_") + typestr;
	py::class_<Class, Parent>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
}

template <typename TDataType>
void declare_particle_elastic_body(py::module& m, std::string typestr) {
	using Class = dyno::ElasticBody<TDataType>;
	using Parent = dyno::ParticleSystem<TDataType>;
	std::string pyclass_name = std::string("ParticleElasticBody") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//		.def("load_particles", (void (Class::*)(Class::Coord lo, Class::Coord hi, Class::Real distance)) &Class::loadParticles)
		.def("load_particles", (void (Class::*)(std::string)) & Class::loadParticles)
		.def("translate", &Class::translate);
}

void declare_func(py::module& m, std::string typestr);

void declare_attribute(py::module& m, std::string typestr);

void pybind_particle_system(py::module& m);