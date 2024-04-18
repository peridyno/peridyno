#include "PyHeightField.h"

//#include "HeightField/initializeHeightField.h"
//void declare_height_field_initializer(py::module& m, std::string typestr = "")
//{
//	using Class = dyno::HeightFieldInitializer;
//	using Parent = dyno::PluginEntry;
//	std::string pyclass_name = std::string("Attribute") + typestr;
//	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
//	//.def(py::init<>());
////.def("instance", &Class::instance);
//}

void pybind_height_field(py::module& m)
{
	declare_ocean<dyno::DataType3f>(m, "3f");
	declare_ocean_patch<dyno::DataType3f>(m, "3f");
	declare_capillary_wave<dyno::DataType3f>(m, "3f");
	declare_coupling<dyno::DataType3f>(m, "3f");
	declare_granular_media<dyno::DataType3f>(m, "3f");
	//declare_height_field_initializer(m);
	declare_land_scape<dyno::DataType3f>(m, "3f");
	declare_surface_particle_tracking<dyno::DataType3f>(m, "3f");
	declare_vessel<dyno::DataType3f>(m, "3f");
	declare_wake<dyno::DataType3f>(m, "3f");
}