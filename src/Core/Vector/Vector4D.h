#pragma once
#include <glm/vec4.hpp>
#include <iostream>

#include "VectorBase.h"

namespace dyno {

	template <typename T, int Dim> class SquareMatrix;

	/*
	 * Vector<T,4> are defined for C++ fundamental integer types and floating-point types
	 */

	template <typename T>
	class Vector<T, 4>
	{
	public:
		typedef T VarType;

		DYN_FUNC Vector();
		DYN_FUNC explicit Vector(T);
		DYN_FUNC Vector(T x, T y, T z, T w);
		DYN_FUNC Vector(const Vector<T, 4>&);
		DYN_FUNC ~Vector();

		DYN_FUNC  static int dims() { return 4; }

		DYN_FUNC T& operator[] (unsigned int);
		DYN_FUNC const T& operator[] (unsigned int) const;

		DYN_FUNC const Vector<T, 4> operator+ (const Vector<T, 4> &) const;
		DYN_FUNC Vector<T, 4>& operator+= (const Vector<T, 4> &);
		DYN_FUNC const Vector<T, 4> operator- (const Vector<T, 4> &) const;
		DYN_FUNC Vector<T, 4>& operator-= (const Vector<T, 4> &);
		DYN_FUNC const Vector<T, 4> operator* (const Vector<T, 4> &) const;
		DYN_FUNC Vector<T, 4>& operator*= (const Vector<T, 4> &);
		DYN_FUNC const Vector<T, 4> operator/ (const Vector<T, 4> &) const;
		DYN_FUNC Vector<T, 4>& operator/= (const Vector<T, 4> &);

		DYN_FUNC Vector<T, 4>& operator= (const Vector<T, 4> &);

		DYN_FUNC bool operator== (const Vector<T, 4> &) const;
		DYN_FUNC bool operator!= (const Vector<T, 4> &) const;

		DYN_FUNC const Vector<T, 4> operator+ (T) const;
		DYN_FUNC const Vector<T, 4> operator- (T) const;
		DYN_FUNC const Vector<T, 4> operator* (T) const;
		DYN_FUNC const Vector<T, 4> operator/ (T) const;

		DYN_FUNC Vector<T, 4>& operator+= (T);
		DYN_FUNC Vector<T, 4>& operator-= (T);
		DYN_FUNC Vector<T, 4>& operator*= (T);
		DYN_FUNC Vector<T, 4>& operator/= (T);

		DYN_FUNC const Vector<T, 4> operator - (void) const;

		DYN_FUNC T norm() const;
		DYN_FUNC T normSquared() const;
		DYN_FUNC Vector<T, 4>& normalize();
		DYN_FUNC T dot(const Vector<T, 4>&) const;
		//    DYN_FUNC const SquareMatrix<T,4> outerProduct(const Vector<T,4>&) const;

		DYN_FUNC Vector<T, 4> minimum(const Vector<T, 4> &) const;
		DYN_FUNC Vector<T, 4> maximum(const Vector<T, 4> &) const;

		DYN_FUNC T* getDataPtr() { return &data_.x; }

		friend std::ostream& operator<<(std::ostream &out, const Vector<T, 4>& vec)
		{
			out << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ", " << vec[3] << ")";
			return out;
		}
	public:
		union
		{
			glm::tvec4<T> data_; //default: zero vector
			struct { T x, y, z, w; };
		};
		
	};

	template class Vector<float, 4>;
	template class Vector<double, 4>;
	//convenient typedefs
	typedef Vector<float, 4> Vec4f;
	typedef Vector<double, 4> Vec4d;

} //end of namespace dyno

#include "Vector4D.inl"