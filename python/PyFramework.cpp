#include "PyFramework.h"

#include "Node.h"
#include "Module/VisualModule.h"
#include "SceneGraph.h"
#include "Log.h"

using Node = dyno::Node;
using Module = dyno::Module;
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

void pybind_framework(py::module& m)
{
	pybind_log(m);

	py::class_<Node, std::shared_ptr<Node>>(m, "Node")
		.def(py::init<>())
		.def("set_name", &Node::setName)
		.def("is_active", &Node::isActive);

	py::class_<Module, std::shared_ptr<Module>>(m, "Module")
		.def(py::init<>());

	py::class_<VisualModule, std::shared_ptr<VisualModule>>(m, "VisualModule")
		.def(py::init<>());

	py::class_<SceneGraph>(m, "SceneGraph")
		.def_static("get_instance", &SceneGraph::getInstance, py::return_value_policy::reference, "Return an instance")
		//.def("set_root_node", &SceneGraph::setRootNode)
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
		.def("set_upper_bound", &SceneGraph::setUpperBound);
}

