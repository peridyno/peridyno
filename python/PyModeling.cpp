#include "PyModeling.h"


void pybind_modeling(py::module& m) {
	//declare_var<dyno::TOrientedBox3D<Real>>(m, "TOrientedBox3D");
	declare_cube_model <dyno::DataType3f>(m, "3f");
	declare_plane_model<dyno::DataType3f>(m, "3f");
	declare_sphere_model<dyno::DataType3f>(m, "3f");
	declare_static_triangular_mesh<dyno::DataType3f>(m, "3f");
}