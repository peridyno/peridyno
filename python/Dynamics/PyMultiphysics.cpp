#include "PyMultiphysics.h"

void pybind_multiphysics(py::module& m)
{
	// Module
	declare_poission_disk_position_shifting<dyno::DataType3f>(m, "3f");

	// Multiphysics
	declare_compute_surface_level_set<dyno::DataType3f>(m, "3f");
	declare_particle_skinning<dyno::DataType3f>(m, "3f");
	declare_volume_boundary<dyno::DataType3f>(m, "3f");
	declare_sdf_sampler<dyno::DataType3f>(m, "3f");
	declare_device_poisson_disk_sampler<dyno::DataType3f>(m, "3f");
	declare_poisson_disk_sampling<dyno::DataType3f>(m, "3f");
}