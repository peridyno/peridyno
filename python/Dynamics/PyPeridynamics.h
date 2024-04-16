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

void pybind_peridynamics(py::module& m);