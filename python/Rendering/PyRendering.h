#pragma once
#include "../PyCommon.h"

#include <Node.h>
#include <ColorMapping.h>
#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include "glad/glad.h"
#include "Topology/TriangleSet.h"
#include "Backend/Cuda/Module/RenderTools.h"
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


#include "Backend/Cuda/Node/GLSurfaceVisualNode.h"
template <typename TDataType>
void declare_gl_surface_visual_node(py::module& m, std::string typestr) {
	using Class = dyno::GLSurfaceVisualNode<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("GLSurfaceVisualNode") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("caption", &Class::caption)
		.def("get_node_type", &Class::getNodeType)
		.def("in_triangle_set", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("var_color", &Class::varColor, py::return_value_policy::reference);
}

void declare_point_visual_module(py::module& m, std::string typestr);

void declare_surface_visual_module(py::module& m, std::string typestr);

void declare_gl_wireframe_visual_module(py::module& m);

void declare_rednder_window(py::module& m);

void declare_construct_tangent_space(py::module& m);

void declare_gl_instance_visual_module(py::module& m);

void declare_gl_photorealistic_render(py::module& m);

void declare_gl_photorealistic_instance_render(py::module& m);

void pybind_rendering(py::module& m);