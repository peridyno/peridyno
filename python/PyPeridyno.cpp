#include "PyPeridyno.h"


PYBIND11_MODULE(PyPeridyno, m) {
	m.doc() = "Python binding of Peridyno";
	pybind_core(m);
	pybind_framework(m);
	pybind_io(m);
	pybind_modeling(m);
	pybind_topology(m);
	pybind_objIO(m);
	pybind_Interaction(m);

	pybind_particle_system(m);
	pybind_rigid_body(m);
	pybind_multiphysics(m);
	pybind_peridynamics(m);
	pybind_dual_particle_system(m);
	pybind_height_field(m);
	pybind_semi_analytical_scheme(m);
	pybind_volume(m);

	pybind_render_core(m);
	pybind_rendering(m);
	pybind_im_widgets(m);
	pybind_glfw_gui(m);


	m.def("getAssetPath", &getAssetPath, "Get the asset path");
	m.def("getPluginPath", &getPluginPath, "Get the plugin path");
}