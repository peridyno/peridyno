#include "PyVolume.h"

void pybind_volume(py::module& m)
{
	// Module
	declare_fast_marching_method_GPU<dyno::DataType3f>(m, "3f");
	declare_fast_sweeping_method<dyno::DataType3f>(m, "3f");
	declare_fast_sweeping_method_GPU<dyno::DataType3f>(m, "3f");
	declare_marching_cubes_helper<dyno::DataType3f>(m, "3f");
	declare_volume_to_triangle_set<dyno::DataType3f>(m, "3f");

	// Volume
	declare_volume<dyno::DataType3f>(m, "3f");
	declare_basic_shape_to_volume<dyno::DataType3f>(m, "3f");
	declare_marching_cubes<dyno::DataType3f>(m, "3f");
	declare_volume_bool<dyno::DataType3f>(m, "3f");
	declare_volume_clipper<dyno::DataType3f>(m, "3f");
	declare_volume_generator<dyno::DataType3f>(m, "3f");
	declare_volume_loader<dyno::DataType3f>(m, "3f");
}