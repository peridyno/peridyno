#include "PyRendering.h"

void declare_point_visual_module(py::module& m, std::string typestr) {
	using Class = dyno::GLPointVisualModule;
	using Parent = dyno::GLVisualModule;

	std::string pyclass_name = std::string("GLPointVisualModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>> GLPV(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	GLPV.def(py::init<>())
		.def("in_point_set", &Class::inPointSet, py::return_value_policy::reference)
		.def("set_color_map_mode", &Class::setColorMapMode)
		.def("in_color", &Class::inColor, py::return_value_policy::reference)
		//DEF_VAR
		.def("var_point_size", &Class::varPointSize, py::return_value_policy::reference)
		//DEF_ENUM
		.def("var_color_mode", &Class::varColorMode, py::return_value_policy::reference);

	//DECLARE_ENUM
	py::enum_<Class::ColorMapMode>(GLPV, "ColorMapMode")
		.value("PER_OBJECT_SHADER", Class::ColorMapMode::PER_OBJECT_SHADER)
		.value("PER_VERTEX_SHADER", Class::ColorMapMode::PER_VERTEX_SHADER);
}

void declare_surface_visual_module(py::module& m, std::string typestr) {
	using Class = dyno::GLSurfaceVisualModule;
	using Parent = dyno::GLVisualModule;

	std::string pyclass_name = std::string("GLSurfaceVisualModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>GLSVM(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	GLSVM.def(py::init<>())
		.def("caption", &Class::caption)
		.def("var_color_mode", &Class::varColorMode, py::return_value_policy::reference)
		.def("var_use_vertex_normal", &Class::varUseVertexNormal, py::return_value_policy::reference)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		//DEF_ARRAY_IN
		.def("in_color", &Class::inColor, py::return_value_policy::reference)
		.def("in_normal", &Class::inNormal, py::return_value_policy::reference)
		.def("in_tex_coord", &Class::inTexCoord, py::return_value_policy::reference)
		.def("in_normal_index", &Class::inNormalIndex, py::return_value_policy::reference)
		.def("in_tex_coord_index", &Class::inTexCoordIndex, py::return_value_policy::reference)
		.def("in_color_texture", &Class::inColorTexture, py::return_value_policy::reference)
		.def("in_bump_map", &Class::inBumpMap, py::return_value_policy::reference);

	py::enum_<Class::EColorMode>(GLSVM, "EColorMode")
		.value("CM_Object", Class::CM_Object)
		.value("CM_Vertex", Class::CM_Vertex)
		.value("CM_Texture", Class::CM_Texture);
}

#include "RenderWindow.h"
#include "Camera.h"
void declare_rednder_window(py::module& m) {
	using Class = dyno::RenderWindow;
	std::string pyclass_name = std::string("RenderWindow");
	py::class_<Class, std::shared_ptr<Class>>RW(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	RW.def(py::init<>())
		.def("initialize", &RenderWindow::initialize) // 绑定 initialize 方法
		.def("mainLoop", &RenderWindow::mainLoop) // 绑定 mainLoop 方法
		.def("get_render_engine", &RenderWindow::getRenderEngine) // 绑定 getRenderEngine 方法
		.def("set_render_engine", &RenderWindow::setRenderEngine) // 绑定 setRenderEngine 方法
		.def("get_camera", &RenderWindow::getCamera) // 绑定 getCamera 方法
		.def("set_camera", &RenderWindow::setCamera) // 绑定 setCamera 方法
		.def("get_render_params", &RenderWindow::getRenderParams) // 绑定 getRenderParams 方法
		.def("set_render_params", &RenderWindow::setRenderParams) // 绑定 setRenderParams 方法
		.def("set_window_size", &RenderWindow::setWindowSize) // 绑定 setWindowSize 方法
		.def("get_selection_mode", &RenderWindow::getSelectionMode) // 绑定 getSelectionMode 方法
		.def("set_selection_mode", &RenderWindow::setSelectionMode);// 绑定 setSelectionMode 方法

	py::enum_<Class::SelectionMode>(RW, "SelectionMode")
		.value("OBJECT_MODE", Class::SelectionMode::OBJECT_MODE)
		.value("PRIMITIVE_MODE", Class::SelectionMode::PRIMITIVE_MODE);
}

void pybind_rendering(py::module& m)
{
	py::class_<GLVisualModule, VisualModule, std::shared_ptr<GLVisualModule>>GLVM(m, "GLVisualModule");
	GLVM.def("set_color", &GLVisualModule::setColor)
		.def("set_metallic", &GLVisualModule::setMetallic)
		.def("set_roughness", &GLVisualModule::setRoughness)
		.def("set_alpha", &GLVisualModule::setAlpha)
		.def("is_transparent", &GLVisualModule::isTransparent);

	// 	py::class_<GLPointVisualModuleWrap, GLVisualModule, std::shared_ptr<GLPointVisualModuleWrap>>
	// 		(m, "GLPointVisualModule", py::buffer_protocol(), py::dynamic_attr())
	// 		.def(py::init<>())
	// 		.def("set_point_size", &GLPointVisualModuleWrap::setPointSize)
	// 		.def("get_point_size", &GLPointVisualModuleWrap::getPointSize)
	// 		.def("in_pointset", &GLPointVisualModuleWrap::inPointSet, py::return_value_policy::reference)
	// 		.def("in_color", &GLPointVisualModuleWrap::inColor);

	//py::class_<GLSurfaceVisualModuleWrap, GLVisualModule, std::shared_ptr<GLSurfaceVisualModuleWrap>>
	//	(m, "GLSurfaceVisualModule", py::buffer_protocol(), py::dynamic_attr())
	//	.def(py::init<>());

	declare_color_mapping<dyno::DataType3f>(m, "3f");

	declare_point_visual_module(m, "");

	declare_surface_visual_module(m, "");

	declare_rednder_window(m);

	//declare_surface_visual_module(m, "3f");
}