#include "PyRenderCore.h"

#include "RenderWindow.h"
//#include "utility.h"
#include "TrackballCamera.h"

#include "OrbitCamera.h"
void declare_orbit_camera(py::module& m) {
	using Class = dyno::OrbitCamera;
	using Parent = dyno::Camera;
	std::string pyclass_name = std::string("OrbitCamera");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init())
		.def("rotateToPoint", &Class::rotateToPoint)
		.def("translateToPoint", &Class::translateToPoint)
		.def("zoom", &Class::zoom) // °ó¶¨ zoom ·½·¨

		.def("registerPoint", &Class::registerPoint) // °ó¶¨ registerPoint ·½·¨

		.def("getViewDir", &Class::getViewDir)
		.def("getEyePos", &Class::getEyePos) // °ó¶¨ getEyePos ·½·¨
		.def("getTargetPos", &Class::getTargetPos) // °ó¶¨ getTargetPos ·½·¨

		.def("setEyePos", &Class::setEyePos) // °ó¶¨ setEyePos ·½·¨
		.def("setTargetPos", &Class::setTargetPos) // °ó¶¨ setTargetPos ·½·¨

		.def("getCoordSystem", &Class::getCoordSystem)
		.def("getViewMat", &Class::getViewMat)
		.def("getProjMat", &Class::getProjMat);
}

#include "RenderEngine.h"
void declare_render_engine(py::module& m) {
	using Class = dyno::RenderEngine;
	std::string pyclass_name = std::string("RenderEngine");
	py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def("initialize", &Class::initialize)
		.def("terminate", &Class::terminate)
		.def("draw", &Class::draw)
		.def("select", &Class::select)
		.def("name", &Class::name)
		.def("setDefaultEnvmap", &Class::setDefaultEnvmap)
		.def("setUseEnvmapBackground", &Class::setUseEnvmapBackground)
		.def("setEnvmapScale", &Class::setEnvmapScale)
		.def("setEnvStyle", &Class::setEnvStyle)

		.def_readwrite("bgColor0", &Class::bgColor0)
		.def_readwrite("bgColor1", &Class::bgColor1)
		.def_readwrite("showGround", &Class::showGround)
		.def_readwrite("planeScale", &Class::planeScale)
		.def_readwrite("rulerScale", &Class::rulerScale)
		.def_readwrite("planeColor", &Class::planeColor)
		.def_readwrite("rulerColor", &Class::rulerColor)

		.def_readwrite("bDrawEnvmap", &Class::bDrawEnvmap)
		.def_readwrite("enmapScale", &Class::enmapScale)
		.def_readwrite("showSceneBounds", &Class::showSceneBounds)
		.def_readwrite("envStyle", &Class::envStyle);
}

void pybind_render_core(py::module& m)
{
	declare_orbit_camera(m);
	declare_render_engine(m);

	py::enum_<typename dyno::EEnvStyle>(m, "EEnvStyle")
		.value("Standard", dyno::EEnvStyle::Standard)
		.value("Studio", dyno::EEnvStyle::Studio)
		.export_values();

	//py::class_<dyno::RenderWindow, std::shared_ptr<dyno::RenderWindow>>RW(m, "RenderWindow");
	//RW.def("initialize", &dyno::RenderWindow::initialize)
	//	.def("main_loop", &dyno::RenderWindow::mainLoop)
	//	.def("get_render_engine", &dyno::RenderWindow::getRenderEngine)
	//	.def("set_render_engine", &dyno::RenderWindow::setRenderEngine)
	//	.def("get_camera", &dyno::RenderWindow::getCamera)
	//	.def("set_camera", &dyno::RenderWindow::setCamera)
	//	.def("get_render_params", &dyno::RenderWindow::getRenderParams)
	//	.def("set_render_params", &dyno::RenderWindow::setRenderParams)
	//	.def("set_window_size", &dyno::RenderWindow::setWindowSize)
	//	.def("get_selection_mode", &dyno::RenderWindow::getSelectionMode)
	//	.def("set_selection_mode", &dyno::RenderWindow::setSelectionMode)
	//	.def("toggle_im_gui", &dyno::RenderWindow::toggleImGUI)
	//	.def("show_im_gui", &dyno::RenderWindow::showImGUI)
	//	.def("is_screen_recording_on", &dyno::RenderWindow::isScreenRecordingOn)
	//	.def("screen_recording_interval", &dyno::RenderWindow::screenRecordingInterval)
	//	.def("set_screen_recording_path", &dyno::RenderWindow::setScreenRecordingPath)
	//	.def("save_screen", &dyno::RenderWindow::saveScreen)
	//	.def("set_main_light_direction", &dyno::RenderWindow::setMainLightDirection)
	//	.def("select", py::overload_cast<int, int, int, int>(&dyno::RenderWindow::select))
	//	.def("select", py::overload_cast<std::shared_ptr<dyno::Node>, int, int>(&dyno::RenderWindow::select))
	//	.def("get_current_selected_node", &dyno::RenderWindow::getCurrentSelectedNode);
	//py::enum_<typename dyno::RenderWindow::SelectionMode>(RW, "SelectionMode")
	//	.value("OBJECT_MODE", dyno::RenderWindow::SelectionMode::OBJECT_MODE)
	//	.value("PRIMITIVE_MODE", dyno::RenderWindow::SelectionMode::PRIMITIVE_MODE);

	//py::class_<TimeElapse, std::shared_ptr<TimeElapse>>(m, "TimeElapse")
	//	.def(py::init<>())
	//	.def("elapse", &TimeElapse::elapse);

	py::class_<dyno::TrackballCamera, dyno::Camera, std::shared_ptr<dyno::TrackballCamera>>(m, "TrackballCamera")
		.def(py::init<>())
		.def("reset", &dyno::TrackballCamera::reset)
		.def("registerPoint", &dyno::TrackballCamera::registerPoint)
		.def("rotateToPoint", &dyno::TrackballCamera::rotateToPoint)
		.def("translateToPoint", &dyno::TrackballCamera::translateToPoint)
		.def("zoom", &dyno::TrackballCamera::zoom)
		.def("setEyePos", &dyno::TrackballCamera::setEyePos)
		.def("setTargetPos", &dyno::TrackballCamera::setTargetPos)
		.def("getTargetPos", &dyno::TrackballCamera::getTargetPos)
		.def("getViewMat", &dyno::TrackballCamera::getViewMat)
		.def("getProjMat", &dyno::TrackballCamera::getProjMat)

		.def_readwrite("mRegX", &dyno::TrackballCamera::mRegX)
		.def_readwrite("mRegY", &dyno::TrackballCamera::mRegY)
		.def_readwrite("mCameraPos", &dyno::TrackballCamera::mCameraPos)
		.def_readwrite("mCameraTarget", &dyno::TrackballCamera::mCameraTarget)
		.def_readwrite("mCameraUp", &dyno::TrackballCamera::mCameraUp);
}