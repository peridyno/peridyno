#pragma once
#include "../PyCommon.h"

#include "Multiphysics/VolumeBoundary.h"
template <typename TDataType>
void declare_volume_boundary(py::module& m, std::string typestr) {
	using Class = dyno::VolumeBoundary<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("VolumeBoundary") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("translate", &Class::translate)
		.def("load_sdf", &Class::loadSDF)
		.def("load_cube", &Class::loadCube)
		.def("load_shpere", &Class::loadShpere)
		//DEF_VAR
		.def("var_tangential_friction", &Class::varTangentialFriction, py::return_value_policy::reference)
		.def("var_normal_friction", &Class::varNormalFriction, py::return_value_policy::reference)
		//DEF_NODE_PORTS
		.def("import_particle_system", &Class::importParticleSystems, py::return_value_policy::reference)
		.def("get_particle_system", &Class::getParticleSystems)
		.def("add_particle_system", &Class::addParticleSystem)
		.def("remove_particle_system", &Class::removeParticleSystem)

		.def("import_triangular_system", &Class::importTriangularSystems, py::return_value_policy::reference)
		.def("get_triangular_system", &Class::getTriangularSystems)
		.def("add_triangular_system", &Class::addTriangularSystem)
		.def("remove_triangular_system", &Class::removeTriangularSystem)

		.def("import_tetrahedral_system", &Class::importTetrahedralSystems, py::return_value_policy::reference)
		.def("get_tetrahedral_system", &Class::getTetrahedralSystems)
		.def("add_tetrahedral_system", &Class::addTetrahedralSystem)
		.def("remove_tetrahedral_system", &Class::removeTetrahedralSystem)
		//DEF_INSTANCE_STATE
		.def("state_topology", &Class::stateTopology, py::return_value_policy::reference);
}

void pybind_multiphysics(py::module& m);