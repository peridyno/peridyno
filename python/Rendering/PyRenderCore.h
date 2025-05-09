#pragma once
#include "../PyCommon.h"
#include "SceneGraph.h"
#include "Field/Color.h"

void declare_orbit_camera(py::module& m);

void pybind_render_core(py::module& m);