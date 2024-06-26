#include "PyRenderCore.h"

#include "OrbitCamera.h"
void declare_orbit_camera(py::module& m) {
	using Class = dyno::OrbitCamera;
	using Parent = dyno::Camera;
	std::string pyclass_name = std::string("OrbitCamera");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init())
		.def("rotate_to_point", &Class::rotateToPoint)
		.def("translate_to_point", &Class::translateToPoint)
		.def("zoom", &Class::zoom) // 绑定 zoom 方法
		.def("register_point", &Class::registerPoint) // 绑定 registerPoint 方法
		.def("get_view_dir", &Class::getViewDir)
		.def("get_eye_pos", &Class::getEyePos) // 绑定 getEyePos 方法
		.def("get_target_pos", &Class::getTargetPos) // 绑定 getTargetPos 方法
		.def("set_eye_pos", &Class::setEyePos) // 绑定 setEyePos 方法
		.def("set_target_pos", &Class::setTargetPos) // 绑定 setTargetPos 方法
		.def("get_coord_system", &Class::getCoordSystem)
		.def("get_view_mat", &Class::getViewMat)
		.def("get_proj_mat", &Class::getProjMat);
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
		.def("set_default_envmap", &Class::setDefaultEnvmap)
		.def("set_use_envmap_background", &Class::setUseEnvmapBackground)
		.def("set_envmap_scale", &Class::setEnvmapScale)
		.def("set_env_style", &Class::setEnvStyle);
}

#include "RenderWindow.h"
void declare_render_window(py::module& m) {
	using Class = dyno::RenderWindow;
	std::string pyclass_name = std::string("RenderWindow");
	py::class_<Class>RW(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	RW.def("initialize", &Class::initialize)
		.def("main_loop", &Class::mainLoop)
		.def("get_render_engine", &Class::getRenderEngine)
		.def("set_render_engine", &Class::setRenderEngine)
		.def("get_camera", &Class::getCamera)
		.def("set_camera", &Class::setCamera)
		.def("get_render_params", &Class::getRenderParams)
		.def("set_render_params", &Class::setRenderParams)
		.def("set_window_size", &Class::setWindowSize)
		.def("get_selection_mode", &Class::getSelectionMode)
		.def("set_selection_mode", &Class::setSelectionMode)
		.def("toogle_imgui", &Class::toggleImGUI)
		.def("show_imgui", &Class::showImGUI)
		.def("is_screen_recording_on", &Class::isScreenRecordingOn)
		.def("screen_recording_interval", &Class::screenRecordingInterval)
		.def("set_screen_recording_path", &Class::setScreenRecordingPath)
		.def("save_screen", &Class::saveScreen)
		.def("set_main_light_direction", &Class::setMainLightDirection);




	py::enum_<typename Class::SelectionMode>(RW, "SelectionMode")
		.value("OBJECT_MODE", Class::SelectionMode::OBJECT_MODE)
		.value("PRIMITIVE_MODE", Class::SelectionMode::PRIMITIVE_MODE);
}



void pybind_render_core(py::module& m)
{
	py::class_<dyno::Color>(m)
		.def(py::init())
		.def("hsv_to_rgb", &dyno::Color::HSVtoRGB);

	declare_orbit_camera(m);
	declare_render_engine(m);
	declare_render_window(m);


}