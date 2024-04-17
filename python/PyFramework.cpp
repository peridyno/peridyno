#include "PyFramework.h"

void declare_modeling_init_static_plugin(py::module& m, std::string typestr) {
	using Class = dyno::ModelingInitializer;
	using Parent = dyno::PluginEntry;
	std::string pyclass_name = std::string("ModelingInitializer" + typestr);
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("modeling_init_static_plugin", &Modeling::initStaticPlugin);
}

void declare_paticleSystem_init_static_plugin(py::module& m, std::string typestr) {
	using Class = dyno::ParticleSystemInitializer;
	using Parent = dyno::PluginEntry;
	std::string pyclass_name = std::string("ParticleSystemInitializer" + typestr);
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("paticleSystem_init_static_plugin", &PaticleSystem::initStaticPlugin);
}

void declare_discrete_topology_mapping(py::module& m, std::string typestr)
{
	using Class = dyno::TopologyMapping;
	using Parent = dyno::Module;
	std::string pyclass_name = std::string("TopologyMapping") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str());
}

void pybind_log(py::module& m)
{
	//TODO: Log is updated, update the python binding as well
// 	py::class_<Log>(m, "Log")
// 		.def_static("set_output", &Log::setOutput)
// 		.def_static("get_output", &Log::getOutput)
// 		.def_static("set_level", &Log::setLevel);
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
		.def("graphics_pipeline", &Node::graphicsPipeline, py::return_value_policy::reference)
		.def("animation_pipeline", &Node::animationPipeline, py::return_value_policy::reference);

	py::class_<NodePort>(m, "NodePort");
	/*		.def(py::init<>())
			.def("get_port_name", &NodePort::getPortName)
			.def("get_port_type", &NodePort::getPortType)
			.def("set_port_type", &NodePort::setPortType)
			.def("get_parent", &NodePort::getParent, py::return_value_policy::reference)
			.def("attach", &NodePort::attach)*/

	py::class_<OBase, std::shared_ptr<OBase>>(m, "OBase");

	py::class_<FBase, std::shared_ptr<FBase>>(m, "FBase")
		.def("get_template_name", &FBase::getTemplateName)
		.def("get_class_name", &FBase::getClassName)
		.def("get_object_name", &FBase::getObjectName)
		.def("get_description", &FBase::getDescription)
		.def("get_device_type", &FBase::getDeviceType)
		.def("set_object_name", &FBase::setObjectName)
		.def("set_parent", &FBase::setParent)
		.def("is_derived", &FBase::isDerived)
		.def("is_auto_destroyable", &FBase::isAutoDestroyable)
		.def("set_auto_destroy", &FBase::setAutoDestroy)
		.def("set_derived", &FBase::setDerived)
		.def("is_modified", &FBase::isModified)
		.def("tick", &FBase::tick)
		.def("tack", &FBase::tack)
		.def("is_optional", &FBase::isOptional)
		.def("tag_optional", &FBase::tagOptional)
		.def("get_min", &FBase::getMin)
		.def("set_min", &FBase::setMin)
		.def("get_max", &FBase::getMax)
		.def("set_max", &FBase::setMax)
		.def("set_range", &FBase::setRange)
		.def("connect", &FBase::connect)
		.def("disconnect", &FBase::disconnect)
		.def("serialize", &FBase::serialize)
		.def("deserialize", &FBase::deserialize)

		.def("get_top_field", &FBase::getTopField)
		.def("get_source", &FBase::getSource)
		.def("promote_ouput", &FBase::promoteOuput)
		.def("promote_input", &FBase::promoteInput)
		.def("demote_ouput", &FBase::demoteOuput)
		.def("demote_input", &FBase::demoteInput)

		.def("is_empty", &FBase::isEmpty)
		.def("update", &FBase::update)
		.def("attach", &FBase::attach)
		.def("detach", &FBase::detach);

	py::class_<Color>(m, "Color")
		.def(py::init<float, float, float>());

	py::class_<InstanceBase, FBase, std::shared_ptr<InstanceBase>>(m, "FInstance");

	py::class_<Module, std::shared_ptr<Module>>(m, "Module")
		.def(py::init<>());

	py::class_<Pipeline, Module, std::shared_ptr<Pipeline>>(m, "Pipeline")
		.def("push_module", &Pipeline::pushModule)
		.def("disable", &Pipeline::disable)
		.def("enable", &Pipeline::enable);

	py::class_<GraphicsPipeline, Pipeline, std::shared_ptr<GraphicsPipeline>>(m, "GraphicsPipeline", py::buffer_protocol(), py::dynamic_attr());

	py::class_<AnimationPipeline, Pipeline, std::shared_ptr<AnimationPipeline>>(m, "AnimationPipeline", py::buffer_protocol(), py::dynamic_attr());

	py::class_<VisualModule, Module, std::shared_ptr<VisualModule>>(m, "VisualModule")
		.def(py::init<>())
		.def("set_visible", &VisualModule::setVisible)
		.def("is_visible", &VisualModule::isVisible)
		.def("get_module_type", &VisualModule::getModuleType);

	py::class_<TopologyModule, OBase, std::shared_ptr<TopologyModule>>(m, "TopologyModule")
		.def(py::init<>());

	py::class_<ComputeModule, Module, std::shared_ptr<ComputeModule>>(m, "ComputeModule");

	py::class_<SceneGraph, OBase, std::shared_ptr<SceneGraph>>(m, "SceneGraph")
		.def(py::init<>())
		.def("bounding_box", &SceneGraph::boundingBox)
		.def("print_node_info", &SceneGraph::printNodeInfo)
		.def("print_module_info", &SceneGraph::printModuleInfo)
		.def("is_node_info_printable", &SceneGraph::isNodeInfoPrintable)
		.def("is_module_info_printable", &SceneGraph::isModuleInfoPrintable)
		.def("load", &SceneGraph::load)
		.def("invoke", &SceneGraph::invoke)
		.def("delete_node", &SceneGraph::deleteNode)
		.def("propagate_node", &SceneGraph::propagateNode)
		.def("is_empty", &SceneGraph::isEmpty)
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

	//------------------------- New ------------------------------2024
	py::class_<dyno::FilePath>(m, "FilePath")
		.def(py::init<const std::string&>())
		.def("string", &dyno::FilePath::string)
		.def("path", &dyno::FilePath::path)
		.def("__eq__", &dyno::FilePath::operator==)
		.def("__ne__", &dyno::FilePath::operator!=)
		.def("is_path", &dyno::FilePath::is_path)
		.def("extensions", &dyno::FilePath::extensions)
		.def("add_extension", &dyno::FilePath::add_extension)
		.def("set_as_path", &dyno::FilePath::set_as_path)
		.def("set_path", &dyno::FilePath::set_path);

	py::class_<dyno::PluginEntry>(m, "PluginEntry")
		.def(py::init<>())
		.def("name", &dyno::PluginEntry::name)
		.def("version", &dyno::PluginEntry::version)
		.def("description", &dyno::PluginEntry::description)
		.def("setName", &dyno::PluginEntry::setName, py::arg("pluginName"))
		.def("setVersion", &dyno::PluginEntry::setVersion, py::arg("pluginVersion"))
		.def("setDescription", &dyno::PluginEntry::setDescription, py::arg("desc"))
		.def("initialize", &dyno::PluginEntry::initialize);

	declare_calculate_norm<dyno::DataType3f>(m, "3f");

	declare_pointset<dyno::DataType3f>(m, "3f");
	declare_edgeSet<dyno::DataType3f>(m, "3f");
	declare_triangleSet<dyno::DataType3f>(m, "3f");

	declare_discrete_topology_mapping(m, "3f");
	declare_discrete_elements_to_triangle_set<dyno::DataType3f>(m, "3f");

	declare_var<float>(m, "f");
	declare_var<bool>(m, "b");
	declare_var<std::string>(m, "s");
	declare_var<dyno::Vec3f>(m, "3f");
	declare_var<dyno::TOrientedBox3D<Real>>(m, "TOrientedBox3D");
	declare_var<dyno::FilePath>(m, "FilePath");

	declare_array<float, DeviceType::GPU>(m, "1fD");
	declare_array<dyno::Vec3f, DeviceType::GPU>(m, "3fD");

	declare_instance<TopologyModule>(m, "");
	declare_instance<dyno::PointSet<dyno::DataType3f>>(m, "PointSet3f");
	declare_instance<dyno::EdgeSet<dyno::DataType3f>>(m, "EdgeSet3f");
	declare_instance<dyno::TriangleSet<dyno::DataType3f>>(m, "TriangleSet3f");
	declare_instance<dyno::DiscreteElements<dyno::DataType3f>>(m, "DiscreteElements3f");

	// New
	declare_parametric_model<dyno::DataType3f>(m, "3f");

	declare_merge_triangle_set<dyno::DataType3f>(m, "3f");

	declare_semiAnalyticalSFI_node<dyno::DataType3f>(m, "3f");

	declare_modeling_init_static_plugin(m, "");
	declare_paticleSystem_init_static_plugin(m, "");
	//declare_semiAnalyticalScheme_init_static_plugin(m, "");
}