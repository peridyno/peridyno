#include "PyPeridyno.h"

#include <Node.h>
#include <ColorMapping.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include "glad/glad.h"
#include "Topology/TriangleSet.h"
using namespace dyno;


/**
	Wrap the visual module to make sure the GLAD is initialzied.
*/
static int glInitialized = 0;

#define WRAP_VISUAL_MODULE(T) \
class T##Wrap : public T {\
protected:\
bool initializeGL() override{\
    if(glInitialized==0) glInitialized=gladLoadGL();\
    if(glInitialized) return T::initializeGL();\
    return false; }\
};

WRAP_VISUAL_MODULE(GLPointVisualModule)
WRAP_VISUAL_MODULE(GLSurfaceVisualModule)

template <typename TDataType>
void declare_color_mapping(py::module& m, std::string typestr) {
	using Class = dyno::ColorMapping<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("ColorMapping") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_scalar", &Class::inScalar, py::return_value_policy::reference)
		.def("out_color", &Class::outColor, py::return_value_policy::reference)
		.def("var_max", &Class::varMax, py::return_value_policy::reference);
}

void declare_point_visual_module(py::module& m, std::string typestr) {
	using Class = dyno::GLPointVisualModule;
	using Parent = dyno::GLVisualModule;

	std::string pyclass_name = std::string("GLPointVisualModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>> GLPV(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	GLPV.def(py::init<>())
		.def("in_pointSet", &Class::inPointSet, py::return_value_policy::reference)
		.def("set_colorMapMode", &GLPointVisualModule::setColorMapMode)
		.def("set_colorMapRange", &GLPointVisualModule::setColorMapRange)
		.def("in_color", &Class::inColor, py::return_value_policy::reference);



	py::enum_<GLPointVisualModule::ColorMapMode>(GLPV, "ColorMapMode")
		.value("PER_OBJECT_SHADER", GLPointVisualModule::ColorMapMode::PER_OBJECT_SHADER)
		.value("PER_VERTEX_SHADER", GLPointVisualModule::ColorMapMode::PER_VERTEX_SHADER);
	
}

void declare_surface_visual_module(py::module& m, std::string typestr) {
	using Class = dyno::GLSurfaceVisualModule;
	using Parent = dyno::GLVisualModule;

	std::string pyclass_name = std::string("GLSurfaceVisualModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("in_triangleSet", &Class::inTriangleSet, py::return_value_policy::reference);



}

void pybind_rendering(py::module& m)
{
	py::class_<GLVisualModule, VisualModule, std::shared_ptr<GLVisualModule>>(m, "GLVisualModule")
		.def("set_color", &GLVisualModule::setColor)
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
		.def(py::init<>())
		.def("set_node", [](std::shared_ptr<GLSurfaceVisualModuleWrap> r, std::shared_ptr<Node> n)
			{
				n->currentTopology()->connect(r->inTriangleSet());
				n->graphicsPipeline()->pushModule(r);
			});

	declare_color_mapping<dyno::DataType3f>(m, "3f");

	declare_point_visual_module(m, "3f");

	declare_surface_visual_module(m, "3f");
}