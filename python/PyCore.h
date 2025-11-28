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
		//.def(py::init<T>)
		//.def(py::init<T, T, T>)
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
				//.def("normSquared", &Class::normSquared)
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


//#include <Matrix/Matrix3x3.inl>
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
		//.def("__mul__",			(Vector		(Matrix::*)(const Vector&) const)	&Matrix::operator*)
		.def("__mul__", [](const Matrix& self, const Vector& v) {return self * v; })
		.def("setRow", &Matrix::setRow)
		.def("setCol", &Matrix::setCol)
		.def("determinant", &Matrix::determinant)
		.def("trace", &Matrix::trace)
		.def("doubleContraction", &Matrix::doubleContraction)
		.def("frobeniusNorm", &Matrix::frobeniusNorm)
		.def("oneNorm", &Matrix::oneNorm)
		.def("infNorm", &Matrix::infNorm)
		.def("transpose", &Matrix::transpose)
		.def("inverse", &Matrix::inverse)
		.def_static("rows", &Matrix::rows)
		.def_static("cols", &Matrix::cols)
		.def_static("identityMatrix", &Matrix::identityMatrix);
}

#include "Quat.h"
template <typename Real>
void declare_quat(py::module& m, std::string typestr) {
	using Quat = dyno::Quat<Real>;
	std::string pyclass_name = std::string("Quat") + typestr;
	py::class_<Quat>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<Real, Real, Real, Real>())
		.def(py::init<Real, const dyno::Vector<Real, 3>&>())
		.def(py::init<const dyno::Vector<Real, 3>, const dyno::Vector < Real, 3 >>())
		.def(py::init<const dyno::Quat<Real>&>())
		.def(py::init<const dyno::SquareMatrix<Real, 3>&>())
		.def(py::init<const dyno::SquareMatrix<Real, 4>&>())
		.def(py::init<const Real, const Real, const Real>())
		/* Assignment operators */
		//.def(py::self = py::self)
		.def(py::self += py::self)
		.def(py::self -= py::self)
		/* Special functions */
		.def("norm", &Quat::norm)
		.def("normSquared", &Quat::normSquared)
		.def("normalize", &Quat::normalize)
		.def("inverse", &Quat::inverse)
		//override
		//.def("angle", &Quat::angle)
		//.def("angle", &Quat::angle)
		.def("conjugate", &Quat::conjugate)
		.def("rotate", &Quat::rotate)
		.def("toRotationAxis", &Quat::toRotationAxis)
		.def("toEulerAngle", &Quat::toEulerAngle)
		.def("toMatrix3x3", &Quat::toMatrix3x3)
		.def("toMatrix4x4", &Quat::toMatrix4x4)
		/* Operator overloading */
		.def("dot", &Quat::dot)
		.def("identity", &Quat::identity)
		.def("fromEulerAngles", &Quat::fromEulerAngles);
}

#include "Array/Array.h"
template<typename T>
void declare_array_CPU(py::module& m, std::string type)
{
	using Class = dyno::Array<T, DeviceType::CPU>;
	using uint = unsigned int;
	std::string pyclass_name = std::string("CArray") + type;
	py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<uint>())
		.def("resize", &Class::resize)
		.def("reset", &Class::reset)
		.def("clear", &Class::clear)
		.def("deviceType", &Class::deviceType)
		.def("__getitem__", [](Class& self, unsigned int idx) -> T& {
		if (idx >= self.size()) {
			throw py::index_error("Index out of range");
		}
		return self[idx];
			}, py::return_value_policy::reference_internal)
		.def("size", &Class::size)
		.def("isCPU", &Class::isCPU)
		.def("isGPU", &Class::isGPU)
		.def("isEmpty", &Class::isEmpty)
		.def("assign", py::overload_cast<const T&>(&Class::assign))
		.def("assign", py::overload_cast<uint, const T&>(&Class::assign))
		.def("assign", py::overload_cast<const dyno::Array<T, DeviceType::GPU>&>(&Class::assign))
		.def("assign", py::overload_cast<const dyno::Array<T, DeviceType::CPU>&>(&Class::assign))
		.def("assign", py::overload_cast<const std::vector<T>&>(&Class::assign));
}

template<typename T>
void declare_array_GPU(py::module& m, std::string type)
{
	using Class = dyno::Array<T, DeviceType::GPU>;
	using uint = unsigned int;
	std::string pyclass_name = std::string("DArray") + type;
	py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<uint>())
		.def("resize", &Class::resize)
		.def("reset", &Class::reset)
		.def("clear", &Class::clear)
		//.def("begin", &Class::begin)
		.def("deviceType", &Class::deviceType)
		.def("__getitem__", [](Class& self, unsigned int idx) -> T& {
		if (idx >= self.size()) {
			throw py::index_error("Index out of range");
		}
		return self[idx];
			}, py::return_value_policy::reference_internal)
		.def("size", &Class::size)
				.def("isCPU", &Class::isCPU)
				.def("isGPU", &Class::isGPU)
				.def("isEmpty", &Class::isEmpty)
				.def("assign", py::overload_cast<const dyno::Array<T, DeviceType::GPU>&>(&Class::assign))
				.def("assign", py::overload_cast<const dyno::Array<T, DeviceType::CPU>&>(&Class::assign))
				.def("assign", py::overload_cast<const dyno::Array<T, DeviceType::GPU>&, const uint, const uint, const uint>(&Class::assign))
				.def("assign", py::overload_cast<const dyno::Array<T, DeviceType::CPU>&, const uint, const uint, const uint>(&Class::assign));
		//.def("assign", py::overload_cast<const std::vector<T>&, const uint count, const uint, const uint>(&Class::assign));
}

#include "Primitive/Primitive3D.h"

void pybind_core(py::module& m);