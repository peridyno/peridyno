#pragma once
#include "../PyCommon.h"

#include "Peridynamics/TriangularSystem.h"
template <typename TDataType>
void declare_triangular_system(py::module& m, std::string typestr) {
	using Class = dyno::TriangularSystem<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("TriangularSystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("add_fixed_particle", &Class::addFixedParticle)
		.def("translate", &Class::translate)
		.def("scale", &Class::scale)
		//DEF_INSTANCE_STATE
		.def("state_triangle_set", &Class::stateTriangleSet, py::return_value_policy::reference)
		//DEF_ARRAY_STATE
		.def("state_position", &Class::statePosition, py::return_value_policy::reference)
		.def("state_velocity", &Class::stateVelocity, py::return_value_policy::reference)
		.def("state_force", &Class::stateForce, py::return_value_policy::reference)
		//public
		.def("load_surface", &Class::loadSurface);
}

#include "Peridynamics/CodimensionalPD.h"
#include "Peridynamics/EnergyDensityFunction.h"
template <typename TDataType>
void declare_codimensionalPD(py::module& m, std::string typestr) {
	using Class = dyno::CodimensionalPD<TDataType>;
	using Parent = dyno::TriangularSystem<TDataType>;
	std::string pyclass_name = std::string("CodimensionalPD") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<Real, Real, Real, Real, std::string>()) // 构造函数
		.def("set_self_contact", &Class::setSelfContact) // 设置自接触
		.def("translate", &Class::translate) // 平移
		//.def("scale", py::overload_cast<Real>(&Class::scale), py::overload_cast<TDataType::Coord>(&Class::scale))
		.def("load_surface", &Class::loadSurface) // 加载表面
		// 绑定设置能量模型的方法
		//.def("set_energy_model", py::overload_cast<dyno::StVKModel<Real>&>(&Class::setEnergyModel),
		//	"Set energy model to StVKModel")
		//.def("set_energy_model", py::overload_cast<dyno::LinearModel<Real>&>(&Class::setEnergyModel),
		//	"Set energy model to LinearModel")
		//.def("set_energy_model", py::overload_cast<dyno::NeoHookeanModel<Real>&>(&Class::setEnergyModel),
		//	"Set energy model to NeoHookeanModel")
		//.def("set_energy_model", py::overload_cast<dyno::XuModel<Real>&>(&Class::setEnergyModel),
		//	"Set energy model to XuModel")
		//.def("set_energy_model", py::overload_cast<dyno::FiberModel<Real>&>(&Class::setEnergyModel),
		//	"Set energy model to FiberModel")
		.def("setMaxIteNumber", &Class::setMaxIteNumber) // 设置最大迭代次数
		.def("setGrad_ite_eps", &Class::setGrad_ite_eps) // 设置梯度迭代容差
		.def("setContactMaxIte", &Class::setContactMaxIte) // 设置接触最大迭代次数
		.def("setAccelerated", &Class::setAccelerated)
		//DEF_VAR
		.def("var_horizon", &Class::varHorizon, py::return_value_policy::reference)
		.def("var_energy_type", &Class::varEnergyType, py::return_value_policy::reference)
		.def("var_energy_model", &Class::varEnergyModel, py::return_value_policy::reference)
		//DEF_ARRAY_STATE
		.def("state_rest_position", &Class::stateRestPosition, py::return_value_policy::reference)
		.def("state_old_position", &Class::stateOldPosition, py::return_value_policy::reference)
		.def("state_vertices_ref", &Class::stateVerticesRef, py::return_value_policy::reference)
		.def("state_rest_norm", &Class::stateRestNorm, py::return_value_policy::reference)
		.def("state_norm", &Class::stateNorm, py::return_value_policy::reference)
		.def("state_vertex_rotation", &Class::stateVertexRotation, py::return_value_policy::reference)
		.def("state_attribute", &Class::stateAttribute, py::return_value_policy::reference)
		.def("state_volume", &Class::stateVolume, py::return_value_policy::reference)
		.def("state_dynamic_force", &Class::stateDynamicForce, py::return_value_policy::reference)
		.def("state_contact_force", &Class::stateContactForce, py::return_value_policy::reference)
		.def("state_march_position", &Class::stateMarchPosition, py::return_value_policy::reference)
		//DEF_ARRAYLIST_STATE
		.def("state_rest_shape", &Class::stateRestShape, py::return_value_policy::reference)
		//DEF_VAR_STATE
		.def("state_max_length", &Class::stateMaxLength, py::return_value_policy::reference)
		.def("state_min_length", &Class::stateMinLength, py::return_value_policy::reference)
		.def("var_neighbor_searching_adjacent", &Class::varNeighborSearchingAdjacent, py::return_value_policy::reference);
}

#include "Peridynamics/Module/CalculateNormalSDF.h"
template <typename TDataType>
void declare_calculate_normal_sdf(py::module& m, std::string typestr) {
	using Class = dyno::CalculateNormalSDF<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("CalculateNormalSDF") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("compute", &Class::compute)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_normal_sdf", &Class::inNormalSDF, py::return_value_policy::reference)
		.def("in_disrance_SDF", &Class::inDisranceSDF, py::return_value_policy::reference)
		.def("in_tets", &Class::inTets, py::return_value_policy::reference);
}

#include "Peridynamics/Module/ContactRule.h"
template <typename TDataType>
void declare_contact_rule(py::module& m, std::string typestr) {
	using Class = dyno::ContactRule<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("ContactRule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("init_ccd_broad_phase", &Class::initCCDBroadPhase)
		.def("set_contact_max_ite", &Class::setContactMaxIte)
		.def("in_triangular_mesh", &Class::inTriangularMesh, py::return_value_policy::reference)
		.def("in_xi", &Class::inXi, py::return_value_policy::reference)
		.def("in_s", &Class::inS, py::return_value_policy::reference)
		.def("in_unit", &Class::inUnit, py::return_value_policy::reference)
		.def("in_old_position", &Class::inOldPosition, py::return_value_policy::reference)
		.def("in_new_position", &Class::inNewPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def_readwrite("weight", &Class::weight);
	//.def("out_contact_force", &Class:; outContactForce, py::return_value_policy::reference);
//.def("out_weight", &Class::outWeight, py::return_value_policy::reference);
}

#include "Peridynamics/Module/LinearElasticitySolver.h"
template <typename TDataType>
void declare_linear_elasticity_solver(py::module& m, std::string typestr) {
	using Class = dyno::LinearElasticitySolver<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("LinearElasticitySolver") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("solve_elasticity", &Class::solveElasticity)
		.def("in_horizon", &Class::inHorizon, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("in_x", &Class::inX, py::return_value_policy::reference)
		.def("in_y", &Class::inY, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_neighbor_ids", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("in_bonds", &Class::inBonds, py::return_value_policy::reference)
		.def("var_mu", &Class::varMu, py::return_value_policy::reference)
		.def("var_lambda", &Class::varLambda, py::return_value_policy::reference)
		.def("var_iteration_number", &Class::varIterationNumber, py::return_value_policy::reference);
}

#include "Peridynamics/Module/CoSemiImplicitHyperelasticitySolver.h"
template <typename TDataType>
void declare_co_semi_implicit_hyperelasticity_solver(py::module& m, std::string typestr) {
	using Class = dyno::CoSemiImplicitHyperelasticitySolver<TDataType>;
	using Parent = dyno::LinearElasticitySolver<TDataType>;
	std::string pyclass_name = std::string("CoSemiImplicitHyperelasticitySolverr") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("solve_elasticity", &Class::solveElasticity)
		.def("set_object_volume", &Class::setObjectVolume)
		.def("set_particle_volume", &Class::setParticleVolume)
		.def("set_contact_max_ite", &Class::setContactMaxIte)
		.def("in_energy_type", &Class::inEnergyType, py::return_value_policy::reference)
		.def("in__energy_models", &Class::inEnergyModels, py::return_value_policy::reference)
		.def("var_neighbor_searching_adjacent", &Class::varNeighborSearchingAdjacent, py::return_value_policy::reference)
		.def("in_rest_norm", &Class::inRestNorm, py::return_value_policy::reference)
		.def("in_old_position", &Class::inOldPosition, py::return_value_policy::reference)
		.def("in_march_position", &Class::inMarchPosition, py::return_value_policy::reference)
		.def("in_norm", &Class::inNorm, py::return_value_policy::reference)
		.def("in_unit", &Class::inUnit, py::return_value_policy::reference)
		.def("in_triangular_mesh", &Class::inTriangularMesh, py::return_value_policy::reference)
		.def("set_xi", &Class::setXi)
		.def("set_k_bend", &Class::setK_bend)
		.def("set_self_contact", &Class::setSelfContact)
		.def("get_xi", &Class::getXi)
		.def("set_e", &Class::setE)
		.def("get_e", &Class::getE)
		.def("set_s", &Class::setS)
		.def("get_s", &Class::getS)
		.def("get_s0", &Class::getS0)
		.def("get_s1", &Class::getS1)
		.def("get_grad_res_eps", &Class::setGrad_res_eps)
		.def("set_accelerated", &Class::setAccelerated)
		.def("in_attribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("get_contact_rule_ptr", &Class::getContactRulePtr)
		.def("in_dynamic_force", &Class::inDynamicForce, py::return_value_policy::reference)
		.def("in_contact_force", &Class::inContactForce, py::return_value_policy::reference);
}

#include "Peridynamics/Module/DamplingParticleIntegrator.h"
template <typename TDataType>
void declare_dampling_particle_integrator(py::module& m, std::string typestr) {
	using Class = dyno::DamplingParticleIntegrator<TDataType>;
	using Parent = dyno::NumericalIntegrator;
	std::string pyclass_name = std::string("DamplingParticleIntegrator") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>()) // 绑定默认构造函数
		.def("begin", &Class::begin) // 绑定 begin 方法
		.def("end", &Class::end) // 绑定 end 方法
		.def("integrate", &Class::integrate) // 绑定 integrate 方法
		.def("update_velocity", &Class::updateVelocity)
		.def("update_position", &Class::updatePosition)
		.def("in_contact_force", &Class::inContactForce, py::return_value_policy::reference)
		.def("in_dynamic_force", &Class::inDynamicForce, py::return_value_policy::reference)
		.def("in_norm", &Class::inNorm, py::return_value_policy::reference)
		.def("in_mu", &Class::inMu, py::return_value_policy::reference)
		.def("in_air_disspation", &Class::inAirDisspation, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_attribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("in_force_density", &Class::inForceDensity, py::return_value_policy::reference);
}

#include "Peridynamics/Module/DragSurfaceInteraction.h"
template <typename TDataType>
void declare_drag_surface_interaction(py::module& m, std::string typestr) {
	using Class = dyno::DragSurfaceInteraction<TDataType>;
	using Parent = dyno::MouseInputModule;
	std::string pyclass_name = std::string("DragSurfaceInteraction") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>()) // 绑定默认构造函数
		.def("interaction_click", &Class::InteractionClick)
		.def("interaction_drag", &Class::InteractionDrag)
		.def("calc_surface_interact_click", &Class::calcSurfaceInteractClick)
		.def("calc_surface_interact_drag", &Class::calcSurfaceInteractDrag)
		.def("set_tri_fixed", &Class::setTriFixed)
		.def("cancel_velocity", &Class::cancelVelocity)
		.def("in_initial_triangle_set", &Class::inInitialTriangleSet, py::return_value_policy::reference)
		.def("in_attribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("var_interation_radius", &Class::varInterationRadius, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference);
}

#include "Peridynamics/Module/DragVertexInteraction.h"
template <typename TDataType>
void declare_drag_vertex_interaction(py::module& m, std::string typestr) {
	using Class = dyno::DragVertexInteraction<TDataType>;
	using Parent = dyno::MouseInputModule;
	std::string pyclass_name = std::string("DragVertexInteraction") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>()) // 绑定默认构造函数
		.def("interaction_click", &Class::InteractionClick)
		.def("interaction_drag", &Class::InteractionDrag)
		.def("calc_vertex_interact_click", &Class::calcVertexInteractClick)
		.def("calc_vertex_interact_drag", &Class::calcVertexInteractDrag)
		.def("set_vertex_fixed", &Class::setVertexFixed)
		.def("cancel_velocity", &Class::cancelVelocity)
		.def("in_initial_triangle_set", &Class::inInitialTriangleSet, py::return_value_policy::reference)
		.def("in_attribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("var_interation_radius", &Class::varInterationRadius, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference);
}

#include "Peridynamics/Module/ElastoplasticityModule.h"
#include "cmath"
template <typename TDataType>
void declare_elastoplasticity_module(py::module& m, std::string typestr) {
	using Class = dyno::ElastoplasticityModule<TDataType>;
	using Parent = dyno::LinearElasticitySolver<TDataType>;
	std::string pyclass_name = std::string("ElastoplasticityModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("solve_elasticity", &Class::solveElasticity)
		.def("apply_plasticity", &Class::applyPlasticity)
		.def("apply_yielding", &Class::applyYielding)
		.def("rotate_rest_shape", &Class::rotateRestShape)
		.def("reconstruct_rest_shape", &Class::reconstructRestShape)
		.def("set_cohesion", &Class::setCohesion)
		.def("set_friction_angle", &Class::setFrictionAngle)
		.def("enable_fully_reconstruction", &Class::enableFullyReconstruction)
		.def("disable_fully_reconstruction", &Class::disableFullyReconstruction)
		.def("enable_incompressibility", &Class::enableIncompressibility)
		.def("disable_incompressibility", &Class::disableIncompressibility)
		.def("in_neighbor_ids", &Class::inNeighborIds, py::return_value_policy::reference);
}

#include "Peridynamics/Module/FixedPoints.h"
template <typename TDataType>
void declare_fixed_points(py::module& m, std::string typestr) {
	using Class = dyno::FixedPoints<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("FixedPoints") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("add_fixed_point", &Class::addFixedPoint)
		.def("remove_fixed_point", &Class::removeFixedPoint)
		.def("clear", &Class::clear)
		.def("in_position", &Class::inPosition, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference);
}

#include "Peridynamics/Module/FractureModule.h"
template <typename TDataType>
void declare_fracture_module(py::module& m, std::string typestr) {
	using Class = dyno::FractureModule<TDataType>;
	using Parent = dyno::ElastoplasticityModule<TDataType>;
	std::string pyclass_name = std::string("FractureModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("apply_plasticity", &Class::applyPlasticity);
}

#include "Peridynamics/Module/GranularModule.h"
template <typename TDataType>
void declare_granular_module(py::module& m, std::string typestr) {
	using Class = dyno::GranularModule<TDataType>;
	using Parent = dyno::ElastoplasticityModule<TDataType>;
	std::string pyclass_name = std::string("GranularModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

#include "Peridynamics/Module/OneDimElasticityModule.h"
template <typename TDataType>
void declare_one_dim_elasticity_module(py::module& m, std::string typestr) {
	using Class = dyno::OneDimElasticityModule<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("OneDimElasticityModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("solve_elasticity", &Class::solveElasticity)
		.def("set_iteration_number", &Class::setIterationNumber)
		.def("get_iteration_number", &Class::getIterationNumber)
		.def("set_material_stiffness", &Class::setMaterialStiffness);
}

#include "Peridynamics/Module/Peridynamics.h"
template <typename TDataType>
void declare_peridynamics(py::module& m, std::string typestr) {
	using Class = dyno::Peridynamics<TDataType>;
	using Parent = dyno::GroupModule;
	std::string pyclass_name = std::string("Peridynamics") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_horizon", &Class::varHorizon, py::return_value_policy::reference)
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference)
		.def("in_x", &Class::inX, py::return_value_policy::reference)
		.def("in_y", &Class::inY, py::return_value_policy::reference)
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_force", &Class::inForce, py::return_value_policy::reference)
		.def("in_bonds", &Class::inBonds, py::return_value_policy::reference);
}

#include "Peridynamics/Module/SemiImplicitHyperelasticitySolver.h"
template <typename TDataType>
void declare_semi_implicit_hyperelasticity_solver(py::module& m, std::string typestr) {
	using Class = dyno::SemiImplicitHyperelasticitySolver<TDataType>;
	using Parent = dyno::LinearElasticitySolver<TDataType>;
	std::string pyclass_name = std::string("SemiImplicitHyperelasticitySolver") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("solve_elasticity", &Class::solveElasticity)
		.def("var_strain_limiting", &Class::varStrainLimiting, py::return_value_policy::reference)
		.def("in_energy_type", &Class::inEnergyType, py::return_value_policy::reference)
		.def("in_energy_models", &Class::inEnergyModels, py::return_value_policy::reference)
		.def("var_is_alpha_computed", &Class::varIsAlphaComputed, py::return_value_policy::reference)
		.def("in_attribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("in_volume", &Class::inVolume, py::return_value_policy::reference)
		.def("in_volume_pair", &Class::inVolumePair, py::return_value_policy::reference);
}

#include "Peridynamics/Bond.h"
template <typename TDataType>
void declare_bond(py::module& m, std::string typestr) {
	using Class = dyno::TBond<TDataType>;
	std::string pyclass_name = std::string("TBond") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def_readwrite("idx", &Class::idx)
		.def_readwrite("xi", &Class::xi);
}

#include "Peridynamics/Cloth.h"
template <typename TDataType>
void declare_cloth(py::module& m, std::string typestr) {
	using Class = dyno::Cloth<TDataType>;
	using Parent = dyno::TriangularSystem<TDataType>;
	std::string pyclass_name = std::string("Cloth") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("state_rest_rotation", &Class::stateRestPosition, py::return_value_policy::reference)
		.def("state_old_position", &Class::stateOldPosition, py::return_value_policy::reference)
		.def("state_bonds", &Class::stateBonds, py::return_value_policy::reference);
}

#include "Peridynamics/ElasticBody.h"
template <typename TDataType>
void declare_elastic_body(py::module& m, std::string typestr) {
	using Class = dyno::ElasticBody<TDataType>;
	using Parent = dyno::ParticleSystem<TDataType>;
	std::string pyclass_name = std::string("ElasticBody") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//.def("load_particles", &Class::loadParticles, py::arg("filename"))
		//.def("load_particles", &Class::loadParticles)
		.def("translate", &Class::translate)
		.def("scale", &Class::scale)
		.def("rotate", &Class::rotate)
		.def("var_horizon", &Class::varHorizon, py::return_value_policy::reference)
		.def("state_reference_position", &Class::stateReferencePosition, py::return_value_policy::reference)
		.def("state_bonds", &Class::stateBonds, py::return_value_policy::reference);
}

#include "Peridynamics/TetrahedralSystem.h"
template <typename TDataType>
void declare_tetrahedral_system(py::module& m, std::string typestr) {
	using Class = dyno::TetrahedralSystem<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("TetrahedralSystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("state_normal_sdf", &Class::stateNormalSDF, py::return_value_policy::reference)
		.def("var_sdf", &Class::varSDF, py::return_value_policy::reference)
		.def("state_tetrahedron_set", &Class::stateTetrahedronSet, py::return_value_policy::reference)
		.def("state_position", &Class::statePosition, py::return_value_policy::reference)
		.def("state_velocity", &Class::stateVelocity, py::return_value_policy::reference)
		.def("state_force", &Class::stateForce, py::return_value_policy::reference)
		.def("load_vertex_from_file", &Class::loadVertexFromFile)
		.def("load_vertex_from_gmsh_file", &Class::loadVertexFromGmshFile)
		.def("translate", &Class::translate)
		.def("scale", &Class::scale)
		.def("rotate", &Class::rotate);
}

#include "Peridynamics/HyperelasticBody.h"
template <typename TDataType>
void declare_hyperelastic_body(py::module& m, std::string typestr) {
	using Class = dyno::HyperelasticBody<TDataType>;
	using Parent = dyno::TetrahedralSystem<TDataType>;
	std::string pyclass_name = std::string("HyperelasticBody") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("translate", &Class::translate)
		//.def("scale", py::overload_cast<Real>(&Class::scale),
		//	py::overload_cast<Coord>(&Class::scale))
		//.def("rotate", py::overload_cast<Quat<Real>>(&Class::rotate),
		//	py::overload_cast<Coord>(&Class::rotate))
		//.def("set_energy_model", py::overload_cast<StVKModel<Real>>(&Class::setEnergyModel),
		//	py::overload_cast<LinearModel<Real>>(&Class::setEnergyModel),
		//	py::overload_cast<NeoHookeanModel<Real>>(&Class::setEnergyModel),
		//	py::overload_cast<XuModel<Real>>(&Class::setEnergyModel))
		.def("load_sdf", &Class::loadSDF)
		.def("var_location", &Class::varLocation, py::return_value_policy::reference)
		.def("var_rotation", &Class::varRotation, py::return_value_policy::reference)
		.def("var_scale", &Class::varScale, py::return_value_policy::reference)
		.def("var_horizon", &Class::varHorizon, py::return_value_policy::reference)
		.def("var_alpha_computed", &Class::varAlphaComputed, py::return_value_policy::reference)
		.def("var_energy_type", &Class::varEnergyType, py::return_value_policy::reference)
		.def("var_energy_model", &Class::varEnergyModel, py::return_value_policy::reference)
		.def("state_rest_position", &Class::stateRestPosition, py::return_value_policy::reference)
		.def("state_bonds", &Class::stateBonds, py::return_value_policy::reference)
		.def("state_volume_pair", &Class::stateVolumePair, py::return_value_policy::reference)
		.def("state_vertex_rotation", &Class::stateVertexRotation, py::return_value_policy::reference)
		.def("state_attribute", &Class::stateAttribute, py::return_value_policy::reference)
		.def("state_volume", &Class::stateVolume, py::return_value_policy::reference)
		.def("var_neighbor_searching_adjacent", &Class::varNeighborSearchingAdjacent, py::return_value_policy::reference)
		.def("var_file_name", &Class::varFileName, py::return_value_policy::reference)
		.def("state_tets", &Class::stateTets, py::return_value_policy::reference)
		.def("state_disrance_sdf", &Class::stateDisranceSDF, py::return_value_policy::reference);
}

//#include "Peridynamics/initializePeridynamics.h"
//template <typename TDataType>
//void declare_peridynamics_initializer(py::module& m, std::string typestr) {
//	using Class = dyno::PeridynamicsInitializer<TDataType>;
//	using Parent = dyno::PluginEntry;
//	std::string pyclass_name = std::string("PeridynamicsInitializer") + typestr;
//	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
//		.def(py::init<>())
//		.def("instance", &Class::instance);
//}

#include "Peridynamics/ThreadSystem.h"
template <typename TDataType>
void declare_thread_system(py::module& m, std::string typestr) {
	using Class = dyno::ThreadSystem<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("ThreadSystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str())
		.def(py::init<>())
		.def("state_edge_set", &Class::stateEdgeSet, py::return_value_policy::reference)
		.def("state_position", &Class::statePosition, py::return_value_policy::reference)
		.def("state_velocity", &Class::stateVelocity, py::return_value_policy::reference)
		.def("state_force", &Class::stateForce, py::return_value_policy::reference)
		.def("add_thread", &Class::addThread);
}

#include "Peridynamics/Thread.h"
template <typename TDataType>
void declare_thread(py::module& m, std::string typestr) {
	using Class = dyno::Thread<TDataType>;
	using Parent = dyno::ThreadSystem<TDataType>;
	std::string pyclass_name = std::string("Thread") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("translate", &Class::translate)
		.def("scale", &Class::scale)
		.def("var_horizon", &Class::varHorizon, py::return_value_policy::reference)
		.def("state_rest_shape", &Class::stateRestShape, py::return_value_policy::reference);
}



void pybind_peridynamics(py::module& m);