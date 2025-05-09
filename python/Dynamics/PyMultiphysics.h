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

#include "Multiphysics/SdfSampler.h"
template <typename TDataType>
void declare_sdf_sampler(py::module& m, std::string typestr) {
	using Class = dyno::SdfSampler<TDataType>;
	using Parent = dyno::Sampler<TDataType>;
	std::string pyclass_name = std::string("SdfSampler") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("reset_states", &Class::resetStates)
		.def("validate_inputs", &Class::validateInputs)
		.def("convert_2_uniform", &Class::convert2Uniform)
		.def("import_volume", &Class::importVolume, py::return_value_policy::reference)
		.def("get_volume", &Class::getVolume)
		.def("var_spacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("var_cube_tilt", &Class::varCubeTilt, py::return_value_policy::reference)
		.def("var_x", &Class::varX, py::return_value_policy::reference)
		.def("var_y", &Class::varY, py::return_value_policy::reference)
		.def("var_z", &Class::varZ, py::return_value_policy::reference)
		.def("var_alpha", &Class::varAlpha, py::return_value_policy::reference)
		.def("var_beta", &Class::varBeta, py::return_value_policy::reference)
		.def("var_gamma", &Class::varGamma, py::return_value_policy::reference);
}

#include "Multiphysics/PoissonDiskSampling.h"
template <typename TDataType>
void declare_poisson_disk_sampling(py::module& m, std::string typestr) {
	using Class = dyno::PoissonDiskSampling<TDataType>;
	using Parent = dyno::SdfSampler<TDataType>;
	std::string pyclass_name = std::string("PoissonDiskSampling") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("construct_grid", &Class::ConstructGrid)
		.def("collision_judge_2d", &Class::collisionJudge2D)
		.def("collision_judge", &Class::collisionJudge)
		.def("load_sdf", &Class::loadSdf)
		.def("var_box_a", &Class::varBox_a, py::return_value_policy::reference)
		.def("var_box_b", &Class::varBox_b, py::return_value_policy::reference)
		.def("var_sdf_file_name", &Class::varSdfFileName, py::return_value_policy::reference)
		.def("lerp", &Class::lerp)
		.def("get_distance_from_sdf", &Class::getDistanceFromSDF)
		.def("get_sdf", &Class::getSDF)
		.def("get_one_point_inside_sdf", &Class::getOnePointInsideSDF);
}

void pybind_multiphysics(py::module& m);