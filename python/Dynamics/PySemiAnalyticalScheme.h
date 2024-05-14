#include "../PyCommon.h"

//#include "SemiAnalyticalScheme/SemiAnalyticalSFINode.h"
//template <typename TDataType>
//void declare_semi_analytical_sfi_node(py::module& m, std::string typestr) {
//	using Class = dyno::SemiAnalyticalSFINode<TDataType>;
//	using Parent = dyno::Node;
//	std::string pyclass_name = std::string("SemiAnalyticalSFINode") + typestr;
//	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
//		.def(py::init<>())
//		.def("import_particle_systems", &Class::importParticleSystems, py::return_value_policy::reference)
//		.def("in_triangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
//		.def("get_particle_systems", &Class::getParticleSystems)
//		.def("add_particle_system", &Class::addParticleSystem)
//		.def("remove_particle_system", &Class::removeParticleSystem)
//		.def("var_fast", &Class::varFast, py::return_value_policy::reference)
//		.def("var_sync_boundary", &Class::varSyncBoundary, py::return_value_policy::reference)
//		.def("state_position", &Class::statePosition, py::return_value_policy::reference)
//		.def("state_velocity", &Class::stateVelocity, py::return_value_policy::reference)
//		.def("state_force_density". & Class::stateForceDensity, py::return_value_policy::reference)
//		.def("state_attribute", &Class::stateAttribute, py::return_value_policy::reference);
//}

#include "SemiAnalyticalScheme/ComputeParticleAnisotropy.h"
template <typename TDataType>
void declare_compute_particle_anisotropy(py::module& m, std::string typestr) {
	using Class = dyno::ComputeParticleAnisotropy<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("ComputeParticleAnisotropy") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("compute", &Class::compute)
		.def("var_smoothing_length", &Class::varSmoothingLength, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_neighbor_ids", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("out_transform", &Class::outTransform, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/SemiAnalyticalIncompressibilityModule.h"
template <typename TDataType>
void declare_semi_analytical_incompressibility_module(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalIncompressibilityModule<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("SemiAnalyticalIncompressibilityModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("in_neighbor_particle_ids", &Class::inNeighborParticleIds, py::return_value_policy::reference)
		.def("in_neighbor_triang_ids", &Class::inNeighborTriangleIds, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/SemiAnalyticalIncompressibleFluidModel.h"
template <typename TDataType>
void declare_semi_analytical_incompressible_fluid_model(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalIncompressibleFluidModel<TDataType>;
	using Parent = dyno::GroupModule;
	std::string pyclass_name = std::string("SemiAnalyticalIncompressibleFluidModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("update_impl", &Class::updateImpl)
		.def("set_smoothing_length", &Class::setSmoothingLength)
		.def("set_rest_density", &Class::setRestDensity);
}

#include "SemiAnalyticalScheme/SemiAnalyticalParticleShifting.h"
template <typename TDataType>
void declare_semi_analytical_particle_shifting(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalParticleShifting<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;
	std::string pyclass_name = std::string("SemiAnalyticalParticleShifting") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_interation_number", &Class::varInterationNumber, py::return_value_policy::reference)
		.def("var_inertia", &Class::varInertia, py::return_value_policy::reference)
		.def("var_bulk", &Class::varBulk, py::return_value_policy::reference)
		.def("var_surface_tension", &Class::varSurfaceTension, py::return_value_policy::reference)
		.def("var_adhesion_intensity", &Class::varAdhesionIntensity, py::return_value_policy::reference)
		.def("var_rest_density", &Class::varRestDensity, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_attribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("in_neighbor_ids", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("in_neighbor_tri_ids", &Class::inNeighborTriIds, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/SemiAnalyticalPBD.h"
template <typename TDataType>
void declare_semi_analytical_pbd(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalPBD<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("SemiAnalyticalPBD") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("var_interation_number", &Class::varInterationNumber, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("in_smoothing_length", &Class::inSmoothingLength, py::return_value_policy::reference)
		.def("in_sampling_distance", &Class::inSamplingDistance, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("in_neighbor_particle_ids", &Class::inNeighborParticleIds, py::return_value_policy::reference)
		.def("in_neighbor_triang_ids", &Class::inNeighborTriangleIds, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/SemiAnalyticalPositionBasedFluidModel.h"
template <typename TDataType>
void declare_semi_analytical_position_based_fluid_model(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalPositionBasedFluidModel<TDataType>;
	using Parent = dyno::GroupModule;
	std::string pyclass_name = std::string("SemiAnalyticalPositionBasedFluidModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_smoothing_length", &Class::varSmoothingLength, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_force", &Class::inForce, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/SemiAnalyticalSFINode.h"
template <typename TDataType>
void declare_semi_analytical_sfi_node(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalSFINode<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("SemiAnalyticalSFINode") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("get_particle_system", &Class::getParticleSystems)
		.def("add_particle_system", &Class::addParticleSystem)
		.def("remove_particle_system", &Class::removeParticleSystem)
		.def("import_particle_systems", &Class::importParticleSystems, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("var_fast", &Class::varFast, py::return_value_policy::reference)
		.def("var_sync_boundary", &Class::varSyncBoundary, py::return_value_policy::reference)
		.def("state_position", &Class::statePosition, py::return_value_policy::reference)
		.def("state_velocity", &Class::stateVelocity, py::return_value_policy::reference)
		.def("state_force_density", &Class::stateForceDensity, py::return_value_policy::reference)
		.def("state_attribute", &Class::stateAttribute, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/SemiAnalyticalSummationDensity.h"
template <typename TDataType>
void declare_semi_analytical_summation_density(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalSummationDensity<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;
	std::string pyclass_name = std::string("SemiAnalyticalSummationDensity") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//.def("compute", &Class::compute)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_neighbor_ids", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("in_neighbor_tri_ids", &Class::inNeighborTriIds, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("out_density", &Class::outDensity, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/SemiAnalyticalSurfaceTensionModel.h"
template <typename TDataType>
void declare_semi_analytical_surface_tension_model(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalSurfaceTensionModel<TDataType>;
	using Parent = dyno::GroupModule;
	std::string pyclass_name = std::string("SemiAnalyticalSurfaceTensionModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_smoothing_length", &Class::varSmoothingLength, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_attribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("in_force_density", &Class::inForceDensity, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("var_surface_tension", &Class::varSurfaceTension, py::return_value_policy::reference)
		.def("var_adhesion_intensity", &Class::varAdhesionIntensity, py::return_value_policy::reference)
		.def("var_rest_density", &Class::varRestDensity, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/TriangularMeshBoundary.h"
template <typename TDataType>
void declare_triangular_mesh_boundary(py::module& m, std::string typestr) {
	using Class = dyno::TriangularMeshBoundary<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("TriangularMeshBoundary") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_thickness", &Class::varThickness, py::return_value_policy::reference)
		.def("var_tangential_friction", &Class::varTangentialFriction, py::return_value_policy::reference)
		.def("var_normal_friction", &Class::varNormalFriction, py::return_value_policy::reference)
		.def("get_particle_system", &Class::getParticleSystems)
		.def("add_particle_system", &Class::addParticleSystem)
		.def("remove_particle_system", &Class::removeParticleSystem)
		.def("import_particle_systems", &Class::importParticleSystems, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("state_position", &Class::statePosition, py::return_value_policy::reference)
		.def("state_velocity", &Class::stateVelocity, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/TriangularMeshConstraint.h"
template <typename TDataType>
void declare_triangular_mesh_constraint(py::module& m, std::string typestr) {
	using Class = dyno::TriangularMeshConstraint<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("TriangularMeshConstraint") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_thickness", &Class::varThickness, py::return_value_policy::reference)
		.def("var_tangential_friction", &Class::varTangentialFriction, py::return_value_policy::reference)
		.def("var_normal_friction", &Class::varNormalFriction, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("in_triangle_neighbor_ids", &Class::inTriangleNeighborIds, py::return_value_policy::reference);
}


void declare_semi_analytical_scheme_initializer(py::module& m);

void pybind_semi_analytical_scheme(py::module& m);