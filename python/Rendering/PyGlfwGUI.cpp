#include "PyGlfwGUI.h"

#include "AppBase.h"
void declare_gltf_app(py::module& m) {
	using Class = dyno::GlfwApp;
	using Parent = dyno::AppBase;
	std::string pyclass_name = std::string("GlfwApp");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init())
		.def("setSceneGraph", &dyno::GlfwApp::setSceneGraph)
		.def("initialize", &dyno::GlfwApp::initialize)
		.def("mainLoop", &dyno::GlfwApp::mainLoop)
		.def("renderWindow", &dyno::GlfwApp::renderWindow)
		.def("setWindowTitle", &dyno::GlfwApp::setWindowTitle);
}

void declare_gltf_render_window(py::module& m)
{
	using Class = dyno::GlfwRenderWindow;
	using Parent = dyno::RenderWindow;
	std::string pyclass_name = std::string("GlfwRenderWindow");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init())
		.def("initialize", &dyno::GlfwRenderWindow::initialize)
		.def("mainLoop", &dyno::GlfwRenderWindow::mainLoop)
		.def("name", &dyno::GlfwRenderWindow::name)
		.def("setWindowTitle", &dyno::GlfwRenderWindow::setWindowTitle)

		.def("setCursorPos", &dyno::GlfwRenderWindow::setCursorPos)
		.def("getCursorPosX", &dyno::GlfwRenderWindow::getCursorPosX)
		.def("getCursorPosY", &dyno::GlfwRenderWindow::getCursorPosY)

		.def("setButtonType", &dyno::GlfwRenderWindow::setButtonType)
		.def("setButtonMode", &dyno::GlfwRenderWindow::setButtonMode)
		.def("setButtonAction", &dyno::GlfwRenderWindow::setButtonAction)
		.def("setButtonState", &dyno::GlfwRenderWindow::setButtonState)

		.def("getButtonType", &dyno::GlfwRenderWindow::getButtonType)
		.def("getButtonMode", &dyno::GlfwRenderWindow::getButtonMode)
		.def("getButtonAction", &dyno::GlfwRenderWindow::getButtonAction)
		.def("getButtonState", &dyno::GlfwRenderWindow::getButtonState)

		.def("turnOnVSync", &dyno::GlfwRenderWindow::turnOnVSync)
		.def("turnOffVSync", &dyno::GlfwRenderWindow::turnOffVSync)

		.def("toggleAnimation", &dyno::GlfwRenderWindow::toggleAnimation)

		.def("getWidth", &dyno::GlfwRenderWindow::getWidth)
		.def("getHeight", &dyno::GlfwRenderWindow::getHeight)
		.def("initializeStyle", &dyno::GlfwRenderWindow::initializeStyle)
		.def("imWindow", &dyno::GlfwRenderWindow::imWindow);
}

void pybind_glfw_gui(py::module& m)
{
	py::class_<dyno::AppBase, std::shared_ptr<dyno::AppBase>>(m, "AppBase")
		.def("initialize", &dyno::AppBase::initialize)
		.def("mainLoop", &dyno::AppBase::mainLoop)
		.def("renderWindow", &dyno::AppBase::renderWindow)
		.def("getSceneGraph", &dyno::AppBase::getSceneGraph)
		.def("setSceneGraph", &dyno::AppBase::setSceneGraph)
		.def("setSceneGraphCreator", &dyno::AppBase::setSceneGraphCreator);

	declare_rednder_window(m);
	declare_gltf_app(m);
	declare_gltf_render_window(m);
}