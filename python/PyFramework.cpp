#include "PyFramework.h"

#include "Node.h"
#include "Module/VisualModule.h"
#include "Module/AnimationPipeline.h"
#include "Module/GraphicsPipeline.h"

#include "Module/CalculateNorm.h"
#include "Module/ComputeModule.h"

#include "SceneGraph.h"
#include "Log.h"

using FBase = dyno::FBase;
using Node = dyno::Node;
using NodePort = dyno::NodePort;
using Module = dyno::Module;
using ComputeModule = dyno::ComputeModule;
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
void declare_var(py::module& m, std::string& typestr) {
	using Class = FVar<T>;
	std::string pyclass_name = std::string("FVar") + typestr;
	py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("connect", &Class::connect)
		.def("disconnect", &Class::disconnect)
		.def("set_value", &Class::setValue)
		.def("get_value", &Class::getValue);
}

template<typename T, DeviceType deviceType>
void declare_array(py::module& m, std::string& typestr) {
	using Class = FArray<T, deviceType>;
	std::string pyclass_name = std::string("Array") + typestr;
	py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
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
		.def("in_vec", &Class::inVec)
		.def("out_norm", &Class::outNorm);
}

void pybind_framework(py::module& m)
{
	pybind_log(m);

	py::class_<Node, std::shared_ptr<Node>>(m, "Node")
		.def(py::init<>())
		.def("set_name", &Node::setName)
		.def("is_active", &Node::isActive)
		.def("connect", &Node::connect)
		.def("disconnect", &Node::disconnect)
		.def("graphics_pipeline", static_cast<std::shared_ptr<GraphicsPipeline> (Node::*)()>(&Node::graphicsPipeline))
		.def("animation_pipeline", static_cast<std::shared_ptr<AnimationPipeline>(Node::*)()>(&Node::animationPipeline));

	py::class_<NodePort, std::shared_ptr<NodePort>>(m, "NodePort");

	py::class_<Module, std::shared_ptr<Module>>(m, "Module")
		.def(py::init<>());

	py::class_<VisualModule, std::shared_ptr<VisualModule>>(m, "VisualModule")
		.def(py::init<>());

	py::class_<ComputeModule, std::shared_ptr<ComputeModule>>(m, "ComputeModule");

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
		.def("get_lower_bound", &SceneGraph::getLowerBound)
		.def("set_upper_bound", &SceneGraph::setUpperBound)
		.def("add_node", static_cast<std::shared_ptr<Node>(SceneGraph::*)(std::shared_ptr<Node>)>(&SceneGraph::addNode));

	declare_calculate_norm<dyno::DataType3f>(m, "3f");
}
