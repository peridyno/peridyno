#include "PyImWidgets.h"

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
}
