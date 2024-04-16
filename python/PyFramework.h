#pragma once
#include "PyCommon.h"

#include "Node.h"
#include "FInstance.h"
#include "Field.h"
#include "Module/VisualModule.h"
#include "Module/AnimationPipeline.h"
#include "Module/GraphicsPipeline.h"

#include "Module/CalculateNorm.h"
#include "Module/ComputeModule.h"

#include "Topology/PointSet.h"
#include "Topology/TriangleSet.h"
#include "Topology/EdgeSet.h"

#include "Module/TopologyMapping.h"
#include "PlaneModel.h"
#include "SphereModel.h"

#include "Mapping/DiscreteElementsToTriangleSet.h"

#include "ParticleSystem/ParticleSystem.h"
#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"

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
using Color = dyno::Color;

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

template <typename TDataType>
void declare_calculate_norm(py::module& m, std::string typestr) {
	using Class = dyno::CalculateNorm<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("CalculateNorm") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_vec", &Class::inVec, py::return_value_policy::reference)
		.def("out_norm", &Class::outNorm, py::return_value_policy::reference);
}

template <typename TDataType>
void declare_pointset(py::module& m, std::string typestr) {
	using Class = dyno::PointSet<TDataType>;
	using Parent = dyno::TopologyModule;
	std::string pyclass_name = std::string("PointSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

template <typename TDataType>
void declare_triangleSet(py::module& m, std::string typestr) {
	using Class = dyno::TriangleSet<TDataType>;
	using Parent = dyno::EdgeSet<TDataType>;
	std::string pyclass_name = std::string("TriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

template <typename TDataType>
void declare_edgeSet(py::module& m, std::string typestr) {
	using Class = dyno::EdgeSet<TDataType>;
	using Parent = dyno::PointSet<TDataType>;
	std::string pyclass_name = std::string("EdgeSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

template <typename TDataType>
void declare_discrete_elements_to_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::DiscreteElementsToTriangleSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("DiscreteElementsToTriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_discreteElements", &Class::inDiscreteElements, py::return_value_policy::reference)
		.def("out_triangleSet", &Class::outTriangleSet, py::return_value_policy::reference);
}


//------------------------- New ------------------------------

template <typename TDataType>
void declare_parametric_model(py::module& m, std::string typestr) {
	using Class = dyno::ParametricModel<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("ParametricModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_location", &Class::varLocation, py::return_value_policy::reference)
		.def("var_rotation", &Class::varRotation, py::return_value_policy::reference)
		.def("var_scale", &Class::varScale, py::return_value_policy::reference);
}

#include "Mapping/MergeTriangleSet.h"
template <typename TDataType>
void declare_merge_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::MergeTriangleSet<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("MergeTriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("state_triangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("in_first", &Class::inFirst, py::return_value_policy::reference)
		.def("in_second", &Class::inSecond, py::return_value_policy::reference);
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

/*
void declare_semiAnalyticalScheme_init_static_plugin(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalSchemeInitializer;
	using Parent = dyno::PluginEntry;
	std::string pyclass_name = std::string("SemiAnalyticalSchemeInitializer" + typestr);
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("semiAnalyticalScheme_init_static_plugin", &SemiAnalyticalScheme::initStaticPlugin);
}*/

//------------------------- NEW END ------------------------------

void declare_modeling_init_static_plugin(py::module& m, std::string typestr);

void declare_paticleSystem_init_static_plugin(py::module& m, std::string typestr);

void declare_discrete_topology_mapping(py::module& m, std::string typestr);


void pybind_log(py::module& m);

void pybind_framework(py::module& m);
