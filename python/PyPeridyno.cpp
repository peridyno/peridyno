#include "PyPeridyno.h"

// void init_GlutGUI(py::module &);
// void init_Core(py::module &);

PYBIND11_MODULE(PyPeridyno, m) {
	m.doc() = "Python binding of Peridyno";

	pybind_glfw_gui(m);
	pybind_core(m);
	pybind_framework(m);
	pybind_particle_system(m);
	pybind_rendering(m);
	pybind_qt_gui(m);
	pybind_cloth(m);
	pybind_rigid_body_system(m);
	pybind_modeling(m);
	pybind_multiphysics(m);
	pybind_peridynamics(m);
	pybind_dual_particle_system(m);
	pybind_height_field(m);
}