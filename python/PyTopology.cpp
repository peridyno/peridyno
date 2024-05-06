#include "PyTopology.h"

void pybind_topology(py::module& m)
{
	declare_pointset<dyno::DataType3f>(m, "3f");
	declare_edgeSet<dyno::DataType3f>(m, "3f");
	declare_triangleSet<dyno::DataType3f>(m, "3f");
	declare_calculate_norm<dyno::DataType3f>(m, "3f");
	declare_height_field_to_triangle_set<dyno::DataType3f>(m, "3f");
	declare_discrete_elements_to_triangle_set<dyno::DataType3f>(m, "3f");
	declare_merge_triangle_set<dyno::DataType3f>(m, "3f");

	declare_neighbor_element_query<dyno::DataType3f>(m, "3f");
	declare_contacts_to_edge_set<dyno::DataType3f>(m, "3f");
	declare_contacts_to_point_set<dyno::DataType3f>(m, "3f");
	declare_neighbor_point_query<dyno::DataType3f>(m, "3f");

}