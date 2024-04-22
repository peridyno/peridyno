#include "PyMultiphysics.h"




void pybind_multiphysics(py::module& m)
{
	declare_volume_boundary<dyno::DataType3f>(m, "3f");
}