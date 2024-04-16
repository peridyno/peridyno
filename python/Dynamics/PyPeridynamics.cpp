#include "PyPeridynamics.h"

void pybind_peridynamics(py::module& m)
{
	declare_triangular_system<dyno::DataType3f>(m, "3f");
	declare_codimensionalPD<dyno::DataType3f>(m, "3f");
}