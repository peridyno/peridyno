#include "PyParticleSystem.h"

#include "ParticleSystem/SquareEmitter.h"
#include "ParticleSystem/ParticleFluid.h"
#include "ParticleSystem/StaticBoundary.h"
#include "ParticleSystem/GhostParticles.h"

#include "Peridynamics/ElasticBody.h"
#include "Peridynamics/Module/ElasticityModule.h"



#include "RigidBody/RigidBody.h"

using Node = dyno::Node;
using NodePort = dyno::NodePort;

template <typename TDataType>
void declare_multi_node_port(py::module& m, std::string typestr) {
	using Class = dyno::MultipleNodePort<TDataType>;
	using Parent = dyno::NodePort;
	std::string pyclass_name = std::string("MultipleNodePort_") + typestr;
	py::class_<Class, Parent>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
}

template <typename TDataType>
void declare_static_boundary(py::module &m, std::string typestr) {
	using Class = dyno::StaticBoundary<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("StaticBoundary") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("add_rigid_body", &Class::addRigidBody)
		.def("add_particle_system", &Class::addParticleSystem)
		.def("load_sdf", &Class::loadSDF)
		.def("load_cube", &Class::loadCube)
		.def("load_sphere", &Class::loadShpere)
		.def("translate", &Class::translate)
		.def("scale", &Class::scale)
		.def("import_particle_systems", &Class::importParticleSystems, py::return_value_policy::reference);
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
void declare_particle_emitter_square(py::module& m, std::string typestr) {
	using Class = dyno::SquareEmitter<TDataType>;
	using Parent = dyno::ParticleEmitter<TDataType>;
	std::string pyclass_name = std::string("SquareEmitter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

template <typename TDataType>
void declare_particle_fluid(py::module& m, std::string typestr) {
	using Class = dyno::ParticleFluid<TDataType>;
	using Parent = dyno::ParticleSystem<TDataType>;
	std::string pyclass_name = std::string("ParticleFluid") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("import_particles_emitters", &Class::importParticleEmitters, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_particle_system(py::module &m, std::string typestr) {
	using Class = dyno::ParticleSystem<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("ParticleSystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("load_particles", (void (Class::*)(Class::Coord lo, Class::Coord hi, Class::Real distance)) & Class::loadParticles)
		.def("state_velocity", &Class::stateVelocity, py::return_value_policy::reference)
		.def("state_position", &Class::statePosition, py::return_value_policy::reference)
		.def("state_force", &Class::stateForce, py::return_value_policy::reference)
		.def("state_point_set", &Class::statePointSet, py::return_value_policy::reference);
}


template <typename TDataType>
void declare_ghost_particlesm(py::module& m, std::string typestr) {
	using Class = dyno::GhostParticles<TDataType>;
	using Parent = dyno::ParticleSystem<TDataType>;
	std::string pyclass_name = std::string("GhostParticles") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("state_normal", &Class::stateNormal, py::return_value_policy::reference)
		.def("state_attribute", &Class::stateAttribute, py::return_value_policy::reference)
		;
}



void declare_attribute(py::module& m, std::string typestr) {
	using Class = dyno::Attribute;

	std::string pyclass_name = std::string("Attribute") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("set_fluid", &Class::setFluid)
		.def("set_dynamic", &Class::setDynamic);
}

template <typename TDataType>
void declare_particle_elastic_body(py::module &m, std::string typestr) {
	using Class = dyno::ElasticBody<TDataType>;
	using Parent = dyno::ParticleSystem<TDataType>;
	std::string pyclass_name = std::string("ParticleElasticBody") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
//		.def("load_particles", (void (Class::*)(Class::Coord lo, Class::Coord hi, Class::Real distance)) &Class::loadParticles)
		.def("load_particles", (void (Class::*)(std::string)) &Class::loadParticles)
		.def("translate", &Class::translate);
}

void declare_func(py::module& m, std::string typestr) {
	using Class = dyno::Attribute;
}

void pybind_particle_system(py::module& m)
{
	declare_func(m, "");

	declare_attribute(m, "");

	declare_multi_node_port<dyno::ParticleEmitter<dyno::DataType3f>>(m, "ParticleEmitter3f");
	declare_multi_node_port<dyno::ParticleSystem<dyno::DataType3f>>(m, "ParticleSystem3f");

	declare_static_boundary<dyno::DataType3f>(m, "3f");

	declare_particle_emitter<dyno::DataType3f>(m, "3f");
	declare_particle_emitter_square<dyno::DataType3f>(m, "3f");
	declare_particle_system<dyno::DataType3f>(m, "3f");
	declare_particle_fluid<dyno::DataType3f>(m, "3f");
	declare_particle_elastic_body<dyno::DataType3f>(m, "3f");


	
	declare_ghost_particlesm<dyno::DataType3f>(m, "3f");
}

