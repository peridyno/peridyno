#include "PyCore.h"

void pybind_core(py::module& m)
{
	py::class_<dyno::CTimer>(m, "CTimer")
		.def(py::init())
		.def("start", &dyno::CTimer::start)
		.def("stop", &dyno::CTimer::stop)
		.def("get_elapsed_time", &dyno::CTimer::getElapsedTime);

	declare_vector<float, 2>(m, "2f");
	declare_vector<float, 3>(m, "3f");
	declare_vector<float, 4>(m, "4f");

	declare_vector<double, 2>(m, "2d");
	declare_vector<double, 3>(m, "3d");
	declare_vector<double, 4>(m, "4d");

	//declare_vector<int, 2>(m, "2i");
	declare_vector<int, 3>(m, "3i");
	//declare_vector<int, 4>(m, "4i");

	declare_matrix<float, 2>(m, "2f");
	declare_matrix<float, 3>(m, "3f");
	declare_matrix<float, 4>(m, "4f");

	declare_matrix<double, 2>(m, "2d");
	declare_matrix<double, 3>(m, "3d");
	declare_matrix<double, 4>(m, "4d");

	declare_quat<float>(m, "1f");

	typedef  dyno::Vector<Real, 2> Coord2D;
	typedef  dyno::Vector<Real, 3> Coord3D;
	typedef  dyno::SquareMatrix<Real, 3> Matrix3D;

	py::class_<dyno::TOrientedBox3D<Real>>(m, "TOrientedBox3D")
		.def(py::init<const Coord3D&, const Coord3D&, const Coord3D&, const Coord3D&, const Coord3D&>())
		.def(py::init<const Coord3D&, const dyno::Quat<Real>&, const Coord3D&>())
		.def(py::init<const dyno::TOrientedBox3D<Real>&>())
		.def("volume", &dyno::TOrientedBox3D<Real>::volume)
		.def("isValid", &dyno::TOrientedBox3D<Real>::isValid)
		.def("rotate", &dyno::TOrientedBox3D<Real>::rotate)
		.def("aabb", &dyno::TOrientedBox3D<Real>::aabb)
		.def_readwrite("center", &dyno::TOrientedBox3D<Real>::center)
		.def_readwrite("u", &dyno::TOrientedBox3D<Real>::u)
		.def_readwrite("v", &dyno::TOrientedBox3D<Real>::v)
		.def_readwrite("w", &dyno::TOrientedBox3D<Real>::w)
		.def_readwrite("extent", &dyno::TOrientedBox3D<Real>::extent);

	//for GL_GhostSPH sample
	py::bind_vector<std::vector<dyno::Vec3f>>(m, "VectorVec3f", py::module_local(false))
		.def(py::init());
	py::bind_vector<std::vector<dyno::Attribute>>(m, "VectorAttribute", py::module_local(false));
}