#pragma once
#include "../PyCommon.h"

#include "QtGUI/QtApp.h"

void declare_qt_app(py::module& m);

void pybind_qt_gui(py::module& m);