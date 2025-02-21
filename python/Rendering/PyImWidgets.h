#pragma once
#include "../PyCommon.h"
#include "ImWidgets/ImChart.h"
#include "ImColorbar.h"
#include "ImWidget.h"

void declare_im_colorbar(py::module& m, std::string typestr);

void declare_im_widget(py::module& m, std::string typestr);

void pybind_im_widgets(py::module& m);

