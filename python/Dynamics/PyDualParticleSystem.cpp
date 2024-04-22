#include "PyDualParticleSystem.h"

void pybind_dual_particle_system(py::module& m)
{
	declare_dual_particle_fluid_system<dyno::DataType3f>(m, "3f");
}