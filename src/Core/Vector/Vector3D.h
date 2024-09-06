#pragma once
#include <glm/vec3.hpp>
#include <iostream>

#include "VectorBase.h"

namespace dyno {

	template <typename T, int Dim> class SquareMatrix;

	/*
	 * Vector<T,3> are defined for C++ fundamental integer types and floating-point types
	 */

	template <typename T>
	class Vector<T, 3>
	{
	public:
		typedef T VarType;

		DYN_FUNC Vector();
		DYN_FUNC explicit Vector(T);
		DYN_FUNC Vector(T x, T y, T z);
		DYN_FUNC Vector(const Vector<T, 3>&);
		DYN_FUNC ~Vector();

		DYN_FUNC static int dims() { return 3; }

		DYN_FUNC T& operator[] (unsigned int);
		DYN_FUNC const T& operator[] (unsigned int) const;

		DYN_FUNC const Vector<T, 3> operator+ (const Vector<T, 3> &) const;
		DYN_FUNC Vector<T, 3>& operator+= (const Vector<T, 3> &);
		DYN_FUNC const Vector<T, 3> operator- (const Vector<T, 3> &) const;
		DYN_FUNC Vector<T, 3>& operator-= (const Vector<T, 3> &);
		DYN_FUNC const Vector<T, 3> operator* (const Vector<T, 3> &) const;
		DYN_FUNC Vector<T, 3>& operator*= (const Vector<T, 3> &);
		DYN_FUNC const Vector<T, 3> operator/ (const Vector<T, 3> &) const;
		DYN_FUNC Vector<T, 3>& operator/= (const Vector<T, 3> &);


		DYN_FUNC Vector<T, 3>& operator= (const Vector<T, 3> &);

		DYN_FUNC bool operator== (const Vector<T, 3> &) const;
		DYN_FUNC bool operator!= (const Vector<T, 3> &) const;

		DYN_FUNC const Vector<T, 3> operator* (T) const;
		DYN_FUNC const Vector<T, 3> operator- (T) const;
		DYN_FUNC const Vector<T, 3> operator+ (T) const;
		DYN_FUNC const Vector<T, 3> operator/ (T) const;

		DYN_FUNC Vector<T, 3>& operator+= (T);
		DYN_FUNC Vector<T, 3>& operator-= (T);
		DYN_FUNC Vector<T, 3>& operator*= (T);
		DYN_FUNC Vector<T, 3>& operator/= (T);

		DYN_FUNC const Vector<T, 3> operator - (void) const;

		DYN_FUNC T norm() const;
		DYN_FUNC T normSquared() const;
		DYN_FUNC Vector<T, 3>& normalize();
		DYN_FUNC Vector<T, 3> cross(const Vector<T, 3> &) const;
		DYN_FUNC T dot(const Vector<T, 3>&) const;
		//    DYN_FUNC const SquareMatrix<T,3> outerProduct(const Vector<T,3>&) const;

		DYN_FUNC Vector<T, 3> minimum(const Vector<T, 3>&) const;
		DYN_FUNC Vector<T, 3> maximum(const Vector<T, 3>&) const;

		DYN_FUNC T* getDataPtr() { return &data_.x; }

		friend std::ostream& operator<<(std::ostream &out, const Vector<T, 3>& vec)
		{
			out << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")";
			return out;
		}

	public:
		union
		{
#ifdef VK_BACKEND
			DYN_ALIGN_16 glm::tvec3<T> data_; //default: zero vector
			struct { T x, y, z, dummy; };
#else
			glm::tvec3<T> data_; //default: zero vector
			struct { T x, y, z; };
#endif // VK_BACKEND
		};
	};

	template class Vector<float, 3>;
	template class Vector<double, 3>;
	//convenient typedefs 
	typedef Vector<float, 3>	Vec3f;
	typedef Vector<double, 3>	Vec3d;
	typedef Vector<int, 3>		Vec3i;
	typedef Vector<uint, 3>		Vec3u;
	typedef Vector<char, 3>		Vec3c;
	typedef Vector<uchar, 3> Vec3uc;

} //end of namespace dyno

#include "Vector3D.inl"
