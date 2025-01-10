#pragma once
#include "../PyCommon.h"

#include "Multiphysics/AdaptiveBoundary.h"
template <typename TDataType>
void declare_adaptive_boundary(py::module& m, std::string typestr) {
	using Class = dyno::AdaptiveBoundary<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("AdaptiveBoundary") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)
		.def("var_tangential_friction", &Class::varTangentialFriction, py::return_value_policy::reference)
		.def("var_normal_friction", &Class::varNormalFriction, py::return_value_policy::reference)

		.def("import_rigid_bodys", &Class::importRigidBodys, py::return_value_policy::reference)
		.def("get_rigid_bodys", &Class::getRigidBodys)
		.def("add_rigid_body", &Class::addRigidBody)
		.def("remove_rigid_body", &Class::removeRigidBody)

		.def("import_particle_systems", &Class::importParticleSystems, py::return_value_policy::reference)
		.def("get_particle_systems", &Class::getParticleSystems)
		.def("add_particle_system", &Class::addParticleSystem)
		.def("remove_particle_system", &Class::removeParticleSystem)

		.def("import_triangular_systems", &Class::importTriangularSystems, py::return_value_policy::reference)
		.def("get_triangular_systems", &Class::getTriangularSystems)
		.def("add_triangular_system", &Class::addTriangularSystem)
		.def("remove_triangular_system", &Class::removeTriangularSystem)

		.def("import_boundarys", &Class::importBoundarys, py::return_value_policy::reference)
		.def("get_boundarys", &Class::getBoundarys)
		.def("add_boundary", &Class::addBoundary)
		.def("remove_boundary", &Class::removeBoundary);
}

#include "Multiphysics/ComputeSurfaceLevelset.h"
template <typename TDataType>
void declare_compute_surface_level_set(py::module& m, std::string typestr) {
	using Class = dyno::ComputeSurfaceLevelset<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("ComputeSurfaceLevelset") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("in_points", &Class::inPoints, py::return_value_policy::reference)
		.def("in_level_set", &Class::inLevelSet, py::return_value_policy::reference)
		.def("in_grid_spacing", &Class::inGridSpacing, py::return_value_policy::reference);
}

#include "Multiphysics/ParticleSkinning.h"
template <typename TDataType>
void declare_particle_skinning(py::module& m, std::string typestr) {
	using Class = dyno::ParticleSkinning<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("ParticleSkinning") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())

		.def("import_particle_systems", &Class::importParticleSystem, py::return_value_policy::reference)
		.def("get_particle_systems", &Class::getParticleSystem)

		.def("state_points", &Class::statePoints, py::return_value_policy::reference)
		.def("state_level_set", &Class::stateLevelSet, py::return_value_policy::reference)
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("state_grid_position", &Class::stateGridPoistion, py::return_value_policy::reference)
		.def("state_grid_spacing", &Class::stateGridSpacing, py::return_value_policy::reference);
}

#include "Multiphysics/VolumeBoundary.h"
template <typename TDataType>
void declare_volume_boundary(py::module& m, std::string typestr) {
	using Class = dyno::VolumeBoundary<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("VolumeBoundary") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_node_type", &Class::getNodeType)
		//DEF_VAR
		.def("var_tangential_friction", &Class::varTangentialFriction, py::return_value_policy::reference)
		.def("var_normal_friction", &Class::varNormalFriction, py::return_value_policy::reference)
		//DEF_NODE_PORTS
		.def("import_volumes", &Class::importVolumes, py::return_value_policy::reference)
		.def("get_volumes", &Class::getVolumes)
		.def("add_volume", &Class::addVolume)
		.def("remove_volume", &Class::removeVolume)

		.def("import_particle_systems", &Class::importParticleSystems, py::return_value_policy::reference)
		.def("get_particle_systems", &Class::getParticleSystems)
		.def("add_particle_system", &Class::addParticleSystem)
		.def("remove_particle_system", &Class::removeParticleSystem)

		.def("import_triangular_systems", &Class::importTriangularSystems, py::return_value_policy::reference)
		.def("get_triangular_systems", &Class::getTriangularSystems)
		.def("add_triangular_system", &Class::addTriangularSystem)
		.def("remove_triangular_system", &Class::removeTriangularSystem)

		.def("import_tetrahedral_systems", &Class::importTetrahedralSystems, py::return_value_policy::reference)
		.def("get_tetrahedral_systems", &Class::getTetrahedralSystems)
		.def("add_tetrahedral_system", &Class::addTetrahedralSystem)
		.def("remove_tetrahedral_system", &Class::removeTetrahedralSystem)
		//DEF_INSTANCE_STATE
		.def("state_topology", &Class::stateTopology, py::return_value_policy::reference);
}

void declare_multiphysics_initializer(py::module& m);

void pybind_multiphysics(py::module& m);