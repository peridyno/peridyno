#include "PyHeightField.h"

void pybind_height_field(py::module& m)
{
	declare_apply_bump_map_2_triangle_set<dyno::DataType3f>(m, "3f");
	declare_steer<dyno::DataType3f>(m, "3f");
	declare_capillary_wave<dyno::DataType3f>(m, "3f");
	declare_granular_media<dyno::DataType3f>(m, "3f");
	declare_land_scape<dyno::DataType3f>(m, "3f");
	declare_ocean_base<dyno::DataType3f>(m, "3f");
	declare_large_ocean<dyno::DataType3f>(m, "3f");
	declare_mountain_torrents<dyno::DataType3f>(m, "3f");
	declare_ocean<dyno::DataType3f>(m, "3f");
	declare_ocean_patch<dyno::DataType3f>(m, "3f");
	declare_rigid_sand_coupling<dyno::DataType3f>(m, "3f");
	declare_rigid_water_coupling<dyno::DataType3f>(m, "3f");
	declare_surface_particle_tracking<dyno::DataType3f>(m, "3f");
	declare_vessel<dyno::DataType3f>(m, "3f");
	declare_wake<dyno::DataType3f>(m, "3f");
}