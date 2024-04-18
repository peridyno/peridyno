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

void pybind_peridynamics(py::module& m);