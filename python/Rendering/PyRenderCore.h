#pragma once
#include "../PyCommon.h"
#include "Color.h"

void declare_orbit_camera(py::module& m);

void pybind_render_core(py::module& m);