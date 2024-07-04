#include "PyGlfwGUI.h"

void declare_gltf_app(py::module& m) {
	using Class = dyno::GlfwApp;
	//using Parent = dyno::AppBase;
	std::string pyclass_name = std::string("GlfwApp");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init())
		.def("set_scenegraph", &dyno::GlfwApp::setSceneGraph)
		.def("initialize", &dyno::GlfwApp::initialize)
		.def("main_loop", &dyno::GlfwApp::mainLoop)
		.def("render_window", &dyno::GlfwApp::renderWindow)
		.def("set_window_title", &dyno::GlfwApp::setWindowTitle);
}

void declare_gltf_render_window(py::module& m)
{
	using Class = dyno::GlfwRenderWindow;
	//using Parent = dyno::RenderWindow;
	std::string pyclass_name = std::string("GlfwRenderWindow");
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init())
		.def("initialize", &dyno::GlfwRenderWindow::initialize)
		.def("main_loop", &dyno::GlfwRenderWindow::mainLoop)
		.def("name", &dyno::GlfwRenderWindow::name)
		.def("set_window_title", &dyno::GlfwRenderWindow::setWindowTitle)
		.def("set_cursor_pos", &dyno::GlfwRenderWindow::setCursorPos)
		.def("get_cursor_pos_x", &dyno::GlfwRenderWindow::getCursorPosX)
		.def("get_cursor_pos_y", &dyno::GlfwRenderWindow::getCursorPosY)
		.def("set_button_type", &dyno::GlfwRenderWindow::setButtonType)
		.def("set_button_mode", &dyno::GlfwRenderWindow::setButtonMode)
		.def("set_button_action", &dyno::GlfwRenderWindow::setButtonAction)
		.def("set_button_state", &dyno::GlfwRenderWindow::setButtonState)
		.def("get_button_type", &dyno::GlfwRenderWindow::getButtonType)
		.def("get_button_mode", &dyno::GlfwRenderWindow::getButtonMode)
		.def("get_button_action", &dyno::GlfwRenderWindow::getButtonAction)
		.def("get_button_state", &dyno::GlfwRenderWindow::getButtonState)
		.def("turn_on_vsync", &dyno::GlfwRenderWindow::turnOnVSync)
		.def("turn_off_vsync", &dyno::GlfwRenderWindow::turnOffVSync)
		.def("toggle_animation", &dyno::GlfwRenderWindow::toggleAnimation)
		.def("get_width", &dyno::GlfwRenderWindow::getWidth)
		.def("get_height", &dyno::GlfwRenderWindow::getHeight)
		.def("initialize_style", &dyno::GlfwRenderWindow::initializeStyle)
		.def("im_window", &dyno::GlfwRenderWindow::imWindow);
}

void pybind_glfw_gui(py::module& m)
{
	//py::class_<dyno::AppBase>(m, "AppBase")
	//	.def("initialize", &dyno::AppBase::initialize)
	//	.def("main_loop", &dyno::AppBase::mainLoop)
	//	.def("get_scene_graph", &dyno::AppBase::getSceneGraph)
	//	.def("set_scene_graph", &dyno::AppBase::setSceneGraph)
	//	.def("set_scene_graph_creator", &dyno::AppBase::setSceneGraphCreator);

	declare_gltf_app(m);
	declare_gltf_render_window(m);
}