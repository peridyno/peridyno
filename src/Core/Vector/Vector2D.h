#pragma once
#include <glm/vec2.hpp>
#include <iostream>

#include "VectorBase.h"

namespace dyno {

	template <typename T, int Dim> class SquareMatrix;

	template <typename T>
	class Vector<T, 2>
	{
	public:
		typedef T VarType;

		DYN_FUNC Vector();
		DYN_FUNC explicit Vector(T);
		DYN_FUNC Vector(T x, T y);
		DYN_FUNC Vector(const Vector<T, 2>&);
		DYN_FUNC ~Vector();

		DYN_FUNC static int dims() { return 2; }

		DYN_FUNC T& operator[] (unsigned int);
		DYN_FUNC const T& operator[] (unsigned int) const;

		DYN_FUNC const Vector<T, 2> operator+ (const Vector<T, 2> &) const;
		DYN_FUNC Vector<T, 2>& operator+= (const Vector<T, 2> &);
		DYN_FUNC const Vector<T, 2> operator- (const Vector<T, 2> &) const;
		DYN_FUNC Vector<T, 2>& operator-= (const Vector<T, 2> &);
		DYN_FUNC const Vector<T, 2> operator* (const Vector<T, 2> &) const;
		DYN_FUNC Vector<T, 2>& operator*= (const Vector<T, 2> &);
		DYN_FUNC const Vector<T, 2> operator/ (const Vector<T, 2> &) const;
		DYN_FUNC Vector<T, 2>& operator/= (const Vector<T, 2> &);

		DYN_FUNC Vector<T, 2>& operator= (const Vector<T, 2> &);

		DYN_FUNC bool operator== (const Vector<T, 2> &) const;
		DYN_FUNC bool operator!= (const Vector<T, 2> &) const;

		DYN_FUNC const Vector<T, 2> operator* (T) const;
		DYN_FUNC const Vector<T, 2> operator- (T) const;
		DYN_FUNC const Vector<T, 2> operator+ (T) const;
		DYN_FUNC const Vector<T, 2> operator/ (T) const;

		DYN_FUNC Vector<T, 2>& operator+= (T);
		DYN_FUNC Vector<T, 2>& operator-= (T);
		DYN_FUNC Vector<T, 2>& operator*= (T);
		DYN_FUNC Vector<T, 2>& operator/= (T);

		DYN_FUNC const Vector<T, 2> operator - (void) const;

		DYN_FUNC T norm() const;
		DYN_FUNC T normSquared() const;
		DYN_FUNC Vector<T, 2>& normalize();
		DYN_FUNC T cross(const Vector<T, 2> &)const;
		DYN_FUNC T dot(const Vector<T, 2>&) const;
		DYN_FUNC Vector<T, 2> minimum(const Vector<T, 2> &) const;
		DYN_FUNC Vector<T, 2> maximum(const Vector<T, 2> &) const;

		DYN_FUNC T* getDataPtr() { return &data_.x; }

		friend std::ostream& operator<<(std::ostream &out, const Vector<T, 2>& vec)
		{
			out << "(" << vec[0] << ", " << vec[1] << ")";
			return out;
		}

	public:
		union
		{
			glm::tvec2<T> data_; //default: zero vector
			struct { T x, y; };
		};
	};

	template class Vector<float, 2>;
	template class Vector<double, 2>;

	typedef Vector<float, 2> Vec2f;
	typedef Vector<double, 2> Vec2d;
	typedef Vector<uint32_t, 2> Vec2u;

} //end of namespace dyno

#include "Vector2D.inl"
