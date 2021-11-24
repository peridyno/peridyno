#include "PyParticleSystem.h"

#include "ParticleSystem/StaticBoundary.h"

#include "Peridynamics/ElasticBody.h"
#include "Peridynamics/ElasticityModule.h"

#include "RigidBody/RigidBody.h"

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
		.def("scale", &Class::scale);
}


template <typename TDataType>
void declare_particle_system(py::module &m, std::string typestr) {
	using Class = dyno::ParticleSystem<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("ParticleSystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

template <typename TDataType>
void declare_particle_elastic_body(py::module &m, std::string typestr) {
	using Class = dyno::ElasticBody<TDataType>;
	using Parent = dyno::ParticleSystem<TDataType>;
	std::string pyclass_name = std::string("ParticleElasticBody") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("load_surface", &Class::loadSurface)
//		.def("load_particles", (void (Class::*)(Class::Coord lo, Class::Coord hi, Class::Real distance)) &Class::loadParticles)
		.def("load_particles", (void (Class::*)(std::string)) &Class::loadParticles)
		.def("translate", &Class::translate)
		.def("get_surface_node", &Class::getSurfaceNode);
}

void pybind_particle_system(py::module& m)
{
	declare_static_boundary<dyno::DataType3f>(m, "3f");
	declare_particle_system<dyno::DataType3f>(m, "3f");
	declare_particle_elastic_body<dyno::DataType3f>(m, "3f");
}

