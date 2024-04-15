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
	py::class_<Class, Parent, std::shared_ptr<Class>> ImColorBar(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	ImColorBar.def(py::init<>())
		.def("set_coord", &Class::setCoord)
		.def("get_coord", &Class::getCoord)
		//DEF_VAR
		.def("var_is_fix", &Class::varIsfix, py::return_value_policy::reference)
		.def("var_min", &Class::varMin, py::return_value_policy::reference)
		.def("var_max", &Class::varMax, py::return_value_policy::reference)
		.def("var_field_name", &Class::varFieldName, py::return_value_policy::reference)
		.def("in_scalar", &Class::inScalar, py::return_value_policy::reference)
		//DEF_ENUM
		.def("var_type", &Class::varType, py::return_value_policy::reference)
		.def("var_number_type", &Class::varNumberType, py::return_value_policy::reference);

	//DECLARE_ENUM
	py::enum_<Class::ColorTable>(ImColorBar, "ColorTable")
		.value("Jet", Class::ColorTable::Jet)
		.value("Heat", Class::ColorTable::Heat);

	py::enum_<Class::NumberTypeSelection>(ImColorBar, "NumberTypeSelection")
		.value("Dec", Class::NumberTypeSelection::Dec)
		.value("Exp", Class::NumberTypeSelection::Exp);
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