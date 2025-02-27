#pragma once
#include "PyCommon.h"

#include <pybind11/stl.h>

#include "Collision/Attribute.h"
#include "Platform.h"

#include "Timer.h"
#include "Vector.h"
#include "Matrix.h"

template <typename T, int dim>
void declare_vector(py::module& m, std::string typestr) {
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
				//.def("norm", &Class::norm)
				//.def("norm_squared", &Class::normSquared)
				//.def("normalize", &Class::normalize)
				//.def("dot", &Class::dot)
				.def("minimum", &Class::minimum)
				.def("maximum", &Class::maximum)
				.def_static("dims", &Class::dims)
				.def("__getitem__", [](const Class& v, size_t i) {
				if (i >= Class::dims()) throw py::index_error();
				return v[i];
					})
				.def("__setitem__", [](Class& v, size_t i, T s) {
						if (i >= Class::dims()) throw py::index_error();
						v[i] = s;
					});
}

template <typename T, int dim>
void declare_matrix(py::module& m, std::string typestr) {
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
		.def("__imul__", (Matrix & (Matrix::*)(const T&)) & Matrix::operator*=)
		//		.def("__truediv__",		(Matrix		(Matrix::*)(const T&) const)		&Matrix::operator/)
		.def("__itruediv__", (Matrix & (Matrix::*)(const T&)) & Matrix::operator/=)
		//		.def("__mul__",			(Vector		(Matrix::*)(const Vector&) const)	&Matrix::operator*)
		.def("set_row", &Matrix::setRow)
		.def("set_col", &Matrix::setCol)
		.def("determinant", &Matrix::determinant)
		.def("trace", &Matrix::trace)
		.def("double_contraction", &Matrix::doubleContraction)
		.def("frobenius_norm", &Matrix::frobeniusNorm)
		.def("one_norm", &Matrix::oneNorm)
		.def("inf_norm", &Matrix::infNorm)
		.def("transpose", &Matrix::transpose)
		.def("inverse", &Matrix::inverse)
		.def_static("rows", &Matrix::rows)
		.def_static("cols", &Matrix::cols)
		.def_static("identity_matrix", &Matrix::identityMatrix);
}

#include "Quat.h"
template <typename Real>
void declare_quat(py::module& m, std::string typestr) {
	using Quat = dyno::Quat<Real>;
	std::string pyclass_name = std::string("Quat") + typestr;
	py::class_<Quat>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<Real, Real, Real, Real>())
		.def(py::init<Real, const dyno::Vector<Real, 3>&>());
}

#include "Array/Array.h"
template<typename T, DeviceType deviceType>
void declare_array(py::module& m, std::string typestr, std::string type) {
	using Class = dyno::Array<T, deviceType>;
	std::string pyclass_name = typestr + std::string("Array") + type;
	using uint = unsigned int;
	py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<uint>())
		.def("resize", &Class::resize)
		.def("reset", &Class::reset)
		.def("clear", &Class::clear)
		//.def("begin", &Class::begin)
		.def("deviceType", &Class::deviceType)
		.def("size", &Class::size)
		.def("is_CPU", &Class::isCPU)
		.def("is_GPU", &Class::isGPU)
		.def("is_empty", &Class::isEmpty);
	//.def("assign", &Class::assign);
//.def("assign", py::overload_cast<const T&>(&Class::assign))
//.def("assign", py::overload_cast<const std::vector<T>&>(&Class::assign))
//.def("assign", py::overload_cast<const dyno::Array<T, DeviceType::GPU>&>(&Class::assign))
//.def("assign", py::overload_cast<const dyno::Array<T, DeviceType::CPU>&>(&Class::assign))

}

#include "Primitive/Primitive3D.h"

void pybind_core(py::module& m);