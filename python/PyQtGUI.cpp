#include "PyCommon.h"

#include "ImColorbar.h"
#include "ImWidget.h"


template<typename T>
void declare_instance(py::module& m, std::string typestr) {
	using Class = dyno::FInstance<T>;
	using Parent = InstanceBase;
	std::string pyclass_name = std::string("Instance") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("connect", &Class::connect)
		.def("disconnect", &Class::disconnect);
}

void declare_im_colorbar(py::module& m, std::string typestr) {
	using Class = dyno::ImColorbar;
	using Parent = dyno::ImWidget;

	std::string pyclass_name = std::string("ImColorbar") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>> (m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_max", &Class::varMax, py::return_value_policy::reference)
		.def("in_scalar", &Class::inScalar, py::return_value_policy::reference);
}

void declare_im_widget(py::module& m, std::string typestr) {
	using Class = dyno::ImWidget;
	using Parent = dyno::VisualModule;

	std::string pyclass_name = std::string("ImWidget") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str());
}

void pybind_qt_gui(py::module& m)
{
	declare_im_widget(m, "3f");
	declare_im_colorbar(m, "3f");

	//declare_instance<dyno::ImColorbar>(m, "ImColorbar3f");
}
