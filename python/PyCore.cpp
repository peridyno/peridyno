#include "PyCore.h"

#include "Timer.h"
#include "Vector.h"
#include "Matrix.h"

template <typename T, int dim>
void declare_vector(py::module &m, std::string typestr) {
	using Class = dyno::Vector<T, dim>;
	std::string pyclass_name = std::string("Vector") + typestr;
	py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<const Class&>())
		.def(py::init([](const std::vector<T>& v) {
			if (v.size() != dim) throw py::index_error();
			Class* vec = new Class;
			for (size_t i = 0; i < dim; i++)
				(*vec)[i] = v[i];
			return vec;
			}), "Default constructor")
		.def(py::self + py::self)
		.def(py::self += py::self)
		.def(py::self - py::self)
		.def(py::self -= py::self)
		.def(py::self *= py::self)
		.def(py::self * py::self)
		.def(py::self /= py::self)
		.def(py::self / py::self)
		.def(py::self + T())
		.def(py::self - T())
		.def(py::self * T())
		.def(py::self / T())
		.def(py::self += T())
		.def(py::self -= T())
		.def(py::self *= T())
		.def(py::self /= T())
		.def(py::self == py::self)
		.def(py::self != py::self)
		.def(-py::self)
		.def("norm",			&Class::norm)
		.def("norm_squared",	&Class::normSquared)
		.def("normalize",		&Class::normalize)
		.def("dot",				&Class::dot)
		.def("minimum",			&Class::minimum)
		.def("maximum",			&Class::maximum)
		.def_static("dims",		&Class::dims)
		.def("__getitem__", [](const Class &v, size_t i) {
			if (i >= Class::dims()) throw py::index_error();
			return v[i];
			})
		.def("__setitem__", [](Class &v, size_t i, T s) {
				if (i >= Class::dims()) throw py::index_error();
				v[i] = s;
			});
}

template <typename T, int dim>
void declare_matrix(py::module &m, std::string typestr) {
	using Vector = dyno::Vector<T, dim>;
	using Matrix = dyno::SquareMatrix<T, dim>;
	std::string pyclass_name = std::string("Matrix") + typestr;
	py::class_<Matrix>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<const Matrix&>())
		.def(py::self + py::self)
		.def(py::self += py::self)
		.def(py::self - py::self)
		.def(py::self -= py::self)
		.def(py::self * py::self)
		.def(py::self *= py::self)
		.def(py::self / py::self)
		.def(py::self /= py::self)
		.def(-py::self)
//		.def("__mul__",			(Matrix		(Matrix::*)(const T&) const)		&Matrix::operator*)
		.def("__imul__",		(Matrix&	(Matrix::*)(const T&))				&Matrix::operator*=)
//		.def("__truediv__",		(Matrix		(Matrix::*)(const T&) const)		&Matrix::operator/)
		.def("__itruediv__",	(Matrix&	(Matrix::*)(const T&))				&Matrix::operator/=)
//		.def("__mul__",			(Vector		(Matrix::*)(const Vector&) const)	&Matrix::operator*)
		.def("set_row",		&Matrix::setRow)
		.def("set_col",		&Matrix::setCol)
		.def("determinant", &Matrix::determinant)
		.def("trace",		&Matrix::trace)
		.def("double_contraction", &Matrix::doubleContraction)
		.def("frobenius_norm", &Matrix::frobeniusNorm)
		.def("one_norm",	&Matrix::oneNorm)
		.def("inf_norm",	&Matrix::infNorm)
		.def("transpose",	&Matrix::transpose)
		.def("inverse",		&Matrix::inverse)
		.def_static("rows", &Matrix::rows)
		.def_static("cols", &Matrix::cols)
		.def_static("identity_matrix", &Matrix::identityMatrix);
}

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

// 	declare_vector<double, 2>(m, "2d");
// 	declare_vector<double, 3>(m, "3d");
// 	declare_vector<double, 4>(m, "4d");

	declare_matrix<float, 2>(m, "2f");
	declare_matrix<float, 3>(m, "3f");
	declare_matrix<float, 4>(m, "4f");
// 
// 	declare_matrix<double, 2>(m, "2d");
// 	declare_matrix<double, 3>(m, "3d");
// 	declare_matrix<double, 4>(m, "4d");
}
