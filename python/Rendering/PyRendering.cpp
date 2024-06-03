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

#include "GLWireframeVisualModule.h"
void declare_gl_wireframe_visual_module(py::module& m) {
	using Class = dyno::GLWireframeVisualModule;
	using Parent = dyno::GLVisualModule;
	std::string pyclass_name = std::string("GLWireframeVisualModule");
	py::class_<Class, Parent, std::shared_ptr<Class>>GLWVM(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	GLWVM.def(py::init<>())
		.def("caption", &Class::caption)
		.def("var_radius", &Class::varRadius, py::return_value_policy::reference)
		.def("var_line_width", &Class::varLineWidth, py::return_value_policy::reference)
		.def("var_render_mode", &Class::varRenderMode, py::return_value_policy::reference)
		.def("in_edge_set", &Class::inEdgeSet, py::return_value_policy::reference);

	py::enum_<typename Class::EEdgeMode>(GLWVM, "EEdgeMode")
		.value("LINE", Class::EEdgeMode::LINE)
		.value("CYLINDER", Class::EEdgeMode::CYLINDER);
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

//backend-module
#include "Backend/Cuda/Module/ConstructTangentSpace.h"
void declare_construct_tangent_space(py::module& m) {
	using Class = dyno::ConstructTangentSpace;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("ConstructTangentSpace");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_texture_mesh", &Class::inTextureMesh, py::return_value_policy::reference)
		.def("out_normal", &Class::outNormal, py::return_value_policy::reference)
		.def("out_tangent", &Class::outTangent, py::return_value_policy::reference)
		.def("out_bitangent", &Class::outBitangent, py::return_value_policy::reference);
}

#include "Backend/Cuda/Module/GLElementVisualModule.h"
void declare_gl_element_visual_module(py::module& m) {
	using Class = dyno::GLElementVisualModule;
	using Parent = dyno::GLVisualModule;
	std::string pyclass_name = std::string("GLElementVisualModule");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_time_step", &Class::inTimeStep, py::return_value_policy::reference);
}

#include "Backend/Cuda/Module/GLInstanceVisualModule.h"
void declare_gl_instance_visual_module(py::module& m) {
	using Class = dyno::GLInstanceVisualModule;
	using Parent = dyno::GLSurfaceVisualModule;
	std::string pyclass_name = std::string("GLInstanceVisualModule");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("in_instance_transform", &Class::inInstanceTransform, py::return_value_policy::reference)
		.def("in_instance_color", &Class::inInstanceColor, py::return_value_policy::reference);
}

#include "Backend/Cuda/Module/GLPhotorealisticRender.h"
void declare_gl_photorealistic_render(py::module& m) {
	using Class = dyno::GLPhotorealisticRender;
	using Parent = dyno::GLVisualModule;
	std::string pyclass_name = std::string("GLPhotorealisticRender");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("in_texture_mesh", &Class::inTextureMesh, py::return_value_policy::reference);
}

#include "Backend/Cuda/Module/GLPhotorealisticInstanceRender.h"
void declare_gl_photorealistic_instance_render(py::module& m) {
	using Class = dyno::GLPhotorealisticInstanceRender;
	using Parent = dyno::GLPhotorealisticRender;
	std::string pyclass_name = std::string("GLPhotorealisticInstanceRender");
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("in_transform", &Class::inTransform, py::return_value_policy::reference);
}

void pybind_rendering(py::module& m)
{
	py::class_<GLVisualModule, VisualModule, std::shared_ptr<GLVisualModule>>GLVM(m, "GLVisualModule");
	GLVM.def("set_color", &GLVisualModule::setColor)
		.def("set_metallic", &GLVisualModule::setMetallic)
		.def("set_roughness", &GLVisualModule::setRoughness)
		.def("set_alpha", &GLVisualModule::setAlpha)
		.def("is_transparent", &GLVisualModule::isTransparent)
		.def("draw", &GLVisualModule::draw)
		.def("release", &GLVisualModule::release)
		.def("var_base_color", &GLVisualModule::varBaseColor, py::return_value_policy::reference)
		.def("var_metallic", &GLVisualModule::varMetallic, py::return_value_policy::reference)
		.def("var_roughness", &GLVisualModule::varRoughness, py::return_value_policy::reference)
		.def("var_alpha", &GLVisualModule::varAlpha, py::return_value_policy::reference);

	declare_color_mapping<dyno::DataType3f>(m, "3f");

	declare_point_visual_module(m, "");

	declare_surface_visual_module(m, "");

	declare_gl_wireframe_visual_module(m);

	declare_rednder_window(m);

	declare_construct_tangent_space(m);

	declare_gl_element_visual_module(m);

	declare_gl_instance_visual_module(m);

	declare_gl_photorealistic_render(m);

	declare_gl_photorealistic_instance_render(m);

	py::class_<RenderTools, std::shared_ptr<RenderTools>>(m, "RenderTools")
		.def(py::init<>())
		.def("setup_color", &RenderTools::setupColor);

	declare_gl_surface_visual_node<dyno::DataType3f>(m, "3f");
}