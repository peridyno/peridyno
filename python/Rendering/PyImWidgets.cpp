#include "PyImWidgets.h"


void declare_im_colorbar(py::module& m, std::string typestr) {
	using Class = dyno::ImColorbar;
	using Parent = dyno::ImWidget;
	std::string pyclass_name = std::string("ImColorbar") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>> ImColorBar(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	ImColorBar.def(py::init<>())
		.def("setCoord", &Class::setCoord)
		.def("getCoord", &Class::getCoord)
		//DEF_VAR
		.def("varIsfix", &Class::varIsfix, py::return_value_policy::reference)
		.def("varMin", &Class::varMin, py::return_value_policy::reference)
		.def("varMax", &Class::varMax, py::return_value_policy::reference)
		.def("varFieldName", &Class::varFieldName, py::return_value_policy::reference)
		.def("inScalar", &Class::inScalar, py::return_value_policy::reference)
		//DEF_ENUM
		.def("varType", &Class::varType, py::return_value_policy::reference)
		.def("varNumberType", &Class::varNumberType, py::return_value_policy::reference);

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

void pybind_im_widgets(py::module& m)
{
	py::class_<dyno::ImChart>(m, "ImChart")
		.def(py::init())
		.def("set_coord", &dyno::ImChart::setCoord, py::return_value_policy::reference)
		.def("get_coord", &dyno::ImChart::getCoord, py::return_value_policy::reference)
		.def("in_frame_number", &dyno::ImChart::inFrameNumber, py::return_value_policy::reference)
		.def("in_value", &dyno::ImChart::inValue, py::return_value_policy::reference)
		.def("in_array", &dyno::ImChart::inArray, py::return_value_policy::reference)
		.def("var_min", &dyno::ImChart::varMin, py::return_value_policy::reference)
		.def("var_max", &dyno::ImChart::varMax, py::return_value_policy::reference)
		.def("var_fix_height", &dyno::ImChart::varFixHeight, py::return_value_policy::reference)
		.def("var_count", &dyno::ImChart::varCount, py::return_value_policy::reference)
		.def("var_title", &dyno::ImChart::varTitle, py::return_value_policy::reference)
		.def("var_output_file", &dyno::ImChart::varOutputFile, py::return_value_policy::reference)
		.def("var_output_path", &dyno::ImChart::varOutputPath, py::return_value_policy::reference)
		.def("var_input_mode", &dyno::ImChart::varInputMode, py::return_value_policy::reference);

	declare_im_widget(m, "3f");
	declare_im_colorbar(m, "3f");
}
