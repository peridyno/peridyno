#include "PyCloth.h"
#include "Peridynamics/Cloth.h"

template <typename TDataType>
void declare_cloth(py::module& m, std::string typestr) {
	using Class = dyno::Cloth<TDataType>;
	using Parent = dyno::ParticleSystem<TDataType>;
	std::string pyclass_name = std::string("Cloth") + typestr;

	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("load_particles", py::detail::overload_cast_impl<std::string>()(&Parent::loadParticles))
		.def("load_particles", py::detail::overload_cast_impl<TDataType::Coord, TDataType::Real, TDataType::Real >()(&Parent::loadParticles))
		.def("load_particles", py::detail::overload_cast_impl<TDataType::Coord , TDataType::Coord , TDataType::Real>()(&Parent::loadParticles))
		.def("load_surface", &Class::loadSurface)
		.def("state_velocity", &Class::stateVelocity, py::return_value_policy::reference);

}

void pybind_cloth(py::module& m) {
	declare_cloth<dyno::DataType3f>(m, "3f");

}