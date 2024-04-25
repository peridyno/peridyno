#pragma once
#include "PyCommon.h"

#include "Node.h"
#include "FInstance.h"
#include "Field.h"
#include "Module/VisualModule.h"
#include "Module/AnimationPipeline.h"
#include "Module/GraphicsPipeline.h"
#include "Module/MouseInputModule.h"

#include "Module/CalculateNorm.h"
#include "Module/ComputeModule.h"

#include "Topology/PointSet.h"
#include "Topology/TriangleSet.h"
#include "Topology/EdgeSet.h"

#include "Module/TopologyMapping.h"
#include "PlaneModel.h"
#include "SphereModel.h"
#include "Module/GroupModule.h"

#include "Mapping/DiscreteElementsToTriangleSet.h"

#include "ParticleSystem/ParticleSystem.h"
#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"
#include "ParticleSystem/ParticleEmitter.h"

#include "Peridynamics/TriangularSystem.h"

#include "HeightField/CapillaryWave.h"
#include "HeightField/Ocean.h"
#include "HeightField/OceanPatch.h"

#include "SceneGraph.h"
#include "Log.h"

#include "Color.h"
#include "FilePath.h"

using FBase = dyno::FBase;
using OBase = dyno::OBase;
using InstanceBase = dyno::InstanceBase;
using Node = dyno::Node;
using NodePort = dyno::NodePort;
using Module = dyno::Module;
using ComputeModule = dyno::ComputeModule;
using TopologyModule = dyno::TopologyModule;
using Pipeline = dyno::Pipeline;
using GraphicsPipeline = dyno::GraphicsPipeline;
using AnimationPipeline = dyno::AnimationPipeline;
using SceneGraph = dyno::SceneGraph;
using VisualModule = dyno::VisualModule;
using Log = dyno::Log;
//new
using Color = dyno::Color;
using ConstraintModule = dyno::ConstraintModule;
using NumericalIntegrator = dyno::NumericalIntegrator;
using InputModule = dyno::InputModule;
using MouseInputModule = dyno::MouseInputModule;
using GroupModule = dyno::GroupModule;
using TopologyMappingdyno = dyno::TopologyMapping;

using uint = unsigned int;
using uchar = unsigned char;
using uint64 = unsigned long long;
using int64 = signed long long;

template<class TNode, class ...Args>
std::shared_ptr<TNode> create_root(SceneGraph& scene, Args&& ... args) {
	return scene.createNewScene<TNode>(std::forward<Args>(args)...);
}

template<typename T>
void declare_var(py::module& m, std::string typestr) {
	using Class = dyno::FVar<T>;
	std::string pyclass_name = std::string("FVar") + typestr;
	py::class_<Class, FBase, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("set_value", &Class::setValue)
		.def("get_value", &Class::getValue);
}

template<typename T, DeviceType deviceType>
void declare_array(py::module& m, std::string typestr) {
	using Class = dyno::FArray<T, deviceType>;
	using Parent = FBase;
	std::string pyclass_name = std::string("Array") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("resize", &Class::resize);
}

template<typename T>
void declare_instance(py::module& m, std::string typestr) {
	using Class = dyno::FInstance<T>;
	using Parent = InstanceBase;
	std::string pyclass_name = std::string("Instance") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("connect", &Class::connect)
		.def("disconnect", &Class::disconnect);
}

//------------------------- New ------------------------------

template <typename TDataType>
void declare_multi_node_port(py::module& m, std::string typestr) {
	using Class = dyno::MultipleNodePort<TDataType>;
	using Parent = dyno::NodePort;
	std::string pyclass_name = std::string("MultipleNodePort_") + typestr;
	py::class_<Class, Parent>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("clear", &Class::clear)
		.def("add_derive_node", &Class::addDerivedNode)
		.def("remove_derive_node", &Class::removeDerivedNode)
		.def("is_kind_of", &Class::isKindOf)
		.def("has_node", &Class::hasNode)
		.def("get_nodes", &Class::getNodes, py::return_value_policy::reference)
		.def("get_derived_node", &Class::getDerivedNodes, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_single_node_port(py::module& m, std::string typestr) {
	using Class = dyno::SingleNodePort<TDataType>;
	using Parent = dyno::NodePort;
	std::string pyclass_name = std::string("SingleNodePort_") + typestr;
	py::class_<Class, Parent>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("is_kind_of", &Class::isKindOf)
		.def("has_node", &Class::hasNode)
		.def("get_nodes", &Class::getNodes, py::return_value_policy::reference)
		.def("get_derived_node", &Class::getDerivedNode, py::return_value_policy::reference)
		.def("set_derived_node", &Class::setDerivedNode);
}

template <typename TDataType>
void declare_parametric_model(py::module& m, std::string typestr) {
	using Class = dyno::ParametricModel<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("ParametricModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("compute_quaternion", &Class::computeQuaternion)
		.def("var_location", &Class::varLocation, py::return_value_policy::reference)
		.def("var_rotation", &Class::varRotation, py::return_value_policy::reference)
		.def("var_scale", &Class::varScale, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/SemiAnalyticalSFINode.h"
template <typename TDataType>
void declare_semiAnalyticalSFI_node(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalSFINode<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("SemiAnalyticalSFINode") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("import_particle_systems", &Class::importParticleSystems, py::return_value_policy::reference)
		.def("in_triangleSet", &Class::inTriangleSet, py::return_value_policy::reference);
}

//Init_static_plugin  - for example_3 WaterPouring
#include "initializeModeling.h"
#include "ParticleSystem/initializeParticleSystem.h"
#include "SemiAnalyticalScheme/initializeSemiAnalyticalScheme.h"

//------------------------- NEW END ------------------------------

void declare_camera(py::module& m);

void pybind_log(py::module& m);

void pybind_framework(py::module& m);
