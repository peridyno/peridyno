#include "PyInteraction.h"

void pybind_Interaction(py::module& m)
{
	declare_edge_interaction<dyno::DataType3f>(m, "3f");
	declare_point_interaction<dyno::DataType3f>(m, "3f");
	declare_surface_interaction<dyno::DataType3f>(m, "3f");
	declare_edge_picker_node<dyno::DataType3f>(m, "3f");
	declare_point_picker_node<dyno::DataType3f>(m, "3f");
	declare_quad_picker_node<dyno::DataType3f>(m, "3f");
	declare_triangle_picker_node<dyno::DataType3f>(m, "3f");
}