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

	py::class_<GLSurfaceVisualModuleWrap, GLVisualModule, std::shared_ptr<GLSurfaceVisualModuleWrap>>
		(m, "GLSurfaceVisualModule", py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());

	declare_color_mapping<dyno::DataType3f>(m, "3f");

	declare_point_visual_module(m, "");

	declare_surface_visual_module(m, "3f");
}