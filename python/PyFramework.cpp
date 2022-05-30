#include "PyFramework.h"

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

#include "Mapping/DiscreteElementsToTriangleSet.h"

#include "SceneGraph.h"
#include "Log.h"

using FBase = dyno::FBase;
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

template<class TNode, class ...Args>
std::shared_ptr<TNode> create_root(SceneGraph& scene, Args&& ... args) {
	return scene.createNewScene<TNode>(std::forward<Args>(args)...);
}

void pybind_log(py::module& m)
{
	py::class_<Log>(m, "Log")
		.def(py::init<>())
		.def_static("set_output", &Log::setOutput)
		.def_static("get_output", &Log::getOutput)
		.def_static("send_message", &Log::sendMessage)
		.def_static("set_level", &Log::setLevel);
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
		.def("set_elementCount", &Class::setElementCount);
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

void declare_discrete_topology_mapping(py::module& m, std::string typestr) {
	using Class = dyno::TopologyMapping;
	using Parent = dyno::Module;
	std::string pyclass_name = std::string("TopologyMapping") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str());
}

void pybind_framework(py::module& m)
{
	pybind_log(m);

	py::class_<Node, std::shared_ptr<Node>>(m, "Node")
		.def(py::init<>())
		.def("set_name", &Node::setName)
		.def("is_active", &Node::isActive)
		.def("connect", &Node::connect)
		.def("set_visible", &Node::setVisible)
		.def("disconnect", &Node::disconnect)
		.def("current_topology", &Node::stateTopology, py::return_value_policy::reference)
		.def("graphics_pipeline", &Node::graphicsPipeline, py::return_value_policy::reference)
		.def("animation_pipeline", &Node::animationPipeline, py::return_value_policy::reference)
		.def("var_location", &Node::varLocation, py::return_value_policy::reference);

	py::class_<NodePort>(m, "NodePort");

	py::class_<FBase, std::shared_ptr<FBase>>(m, "FBase")
		.def("connect", &FBase::connect);

	py::class_<InstanceBase, FBase, std::shared_ptr<InstanceBase>>(m, "FInstance");

	py::class_<Module, std::shared_ptr<Module>>(m, "Module")
		.def(py::init<>());

	py::class_<Pipeline, Module, std::shared_ptr<Pipeline>>(m, "Pipeline")
		.def("push_module", &Pipeline::pushModule);

	py::class_<GraphicsPipeline, Pipeline, std::shared_ptr<GraphicsPipeline>>(m, "GraphicsPipeline", py::buffer_protocol(), py::dynamic_attr());

	py::class_<AnimationPipeline, Pipeline, std::shared_ptr<AnimationPipeline>>(m, "AnimationPipeline", py::buffer_protocol(), py::dynamic_attr());

	py::class_<VisualModule, Module, std::shared_ptr<VisualModule>>(m, "VisualModule")
		.def(py::init<>());

	py::class_<TopologyModule, Module, std::shared_ptr<TopologyModule>>(m, "TopologyModule")
		.def(py::init<>());

	py::class_<ComputeModule, Module, std::shared_ptr<ComputeModule>>(m, "ComputeModule");

	py::class_<SceneGraph, std::shared_ptr<SceneGraph>>(m, "SceneGraph")
		.def(py::init<>())
		.def("is_initialized", &SceneGraph::isInitialized)
		.def("initialize", &SceneGraph::initialize)
		.def("set_total_time", &SceneGraph::setTotalTime)
		.def("get_total_time", &SceneGraph::getTotalTime)
		.def("set_frame_rate", &SceneGraph::setFrameRate)
		.def("get_frame_rate", &SceneGraph::getFrameRate)
		.def("get_timecost_perframe", &SceneGraph::getTimeCostPerFrame)
		.def("get_frame_interval", &SceneGraph::getFrameInterval)
		.def("get_frame_number", &SceneGraph::getFrameNumber)
		.def("set_gravity", &SceneGraph::setGravity)
		.def("get_gravity", &SceneGraph::getGravity)
		.def("set_upper_bound", &SceneGraph::setUpperBound)
		.def("get_upper_bound", &SceneGraph::getUpperBound)
		.def("set_lower_bound", &SceneGraph::setLowerBound)
		.def("get_lower_bound", &SceneGraph::getLowerBound)
		.def("add_node", static_cast<std::shared_ptr<Node>(SceneGraph::*)(std::shared_ptr<Node>)>(&SceneGraph::addNode));

	declare_calculate_norm<dyno::DataType3f>(m, "3f");


	declare_pointset<dyno::DataType3f>(m, "3f");
	declare_edgeSet<dyno::DataType3f>(m, "3f");
	declare_triangleSet<dyno::DataType3f>(m, "3f");

	declare_discrete_topology_mapping(m, "3f");
	declare_discrete_elements_to_triangle_set<dyno::DataType3f>(m, "3f");

	declare_var<float>(m, "f");
	declare_var<dyno::Vec3f>(m, "3f");

	declare_array<float, DeviceType::GPU>(m, "1fD");
	declare_array<dyno::Vec3f, DeviceType::GPU>(m, "3fD");

	declare_instance<TopologyModule>(m, "");
	declare_instance<dyno::PointSet<dyno::DataType3f>>(m, "PointSet3f");
	declare_instance<dyno::EdgeSet<dyno::DataType3f>>(m, "EdgeSet3f");
	declare_instance<dyno::TriangleSet<dyno::DataType3f>>(m, "TriangleSet3f");
	declare_instance<dyno::DiscreteElements<dyno::DataType3f>>(m, "DiscreteElements3f");


}
