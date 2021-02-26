#pragma once
#include <cstddef>
#include <stdexcept>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <cassert>
#include "Platform.h"

namespace dyno
{
	template<class T, unsigned int N>
	class VectorND
	{
	public:
		// type definitions
		typedef T              value_type;
		typedef T*             iterator;
		typedef const T*       const_iterator;
		typedef T&             reference;
		typedef const T&       const_reference;
		typedef unsigned int   size_type;

		DYN_FUNC VectorND()
		{
		}

		/// Specific constructor for 1-element vectors.
		template<int NN = N, typename std::enable_if<NN == 1, int>::type = 0>
		DYN_FUNC explicit VectorND(value_type r1)
		{
			static_assert(N == 1, "");
			this->data[0] = r1;
		}

		/// Specific constructor for 2-elements vectors.
		template<int NN = N, typename std::enable_if<NN == 2, int>::type = 0>
		DYN_FUNC VectorND(value_type r1, value_type r2)
		{
			static_assert(N == 2, "");
			this->data[0] = r1;
			this->data[1] = r2;
		}

		/// Specific constructor for 3-elements vectors.
		template<int NN = N, typename std::enable_if<NN == 3, int>::type = 0>
		DYN_FUNC VectorND(value_type r1, value_type r2, value_type r3)
		{
			static_assert(N == 3, "");
			this->data[0] = r1;
			this->data[1] = r2;
			this->data[2] = r3;
		}

		/// Specific constructor for 4-elements vectors.
		template<int NN = N, typename std::enable_if<NN == 4, int>::type = 0>
		DYN_FUNC VectorND(value_type r1, value_type r2, value_type r3, value_type r4)
		{
			static_assert(N == 4, "");
			this->data[0] = r1;
			this->data[1] = r2;
			this->data[2] = r3;
			this->data[3] = r4;
		}

		/// Specific constructor for 5-elements vectors.
		template<int NN = N, typename std::enable_if<NN == 5, int>::type = 0>
		DYN_FUNC VectorND(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5)
		{
			static_assert(N == 5, "");
			this->data[0] = r1;
			this->data[1] = r2;
			this->data[2] = r3;
			this->data[3] = r4;
			this->data[4] = r5;
		}

		/// Specific constructor for 6-elements vectors.
		template<int NN = N, typename std::enable_if<NN == 6, int>::type = 0>
		DYN_FUNC VectorND(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5, value_type r6)
		{
			static_assert(N == 6, "");
			this->data[0] = r1;
			this->data[1] = r2;
			this->data[2] = r3;
			this->data[3] = r4;
			this->data[4] = r5;
			this->data[5] = r6;
		}

		/// Specific constructor for 7-elements vectors.
		template<int NN = N, typename std::enable_if<NN == 7, int>::type = 0>
		DYN_FUNC VectorND(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5, value_type r6, value_type r7)
		{
			static_assert(N == 7, "");
			this->data[0] = r1;
			this->data[1] = r2;
			this->data[2] = r3;
			this->data[3] = r4;
			this->data[4] = r5;
			this->data[5] = r6;
			this->data[6] = r7;
		}

		/// Specific constructor for 8-elements vectors.
		template<int NN = N, typename std::enable_if<NN == 8, int>::type = 0>
		DYN_FUNC VectorND(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5, value_type r6, value_type r7, value_type r8)
		{
			static_assert(N == 8, "");
			this->data[0] = r1;
			this->data[1] = r2;
			this->data[2] = r3;
			this->data[3] = r4;
			this->data[4] = r5;
			this->data[5] = r6;
			this->data[6] = r7;
			this->data[7] = r8;
		}

		/// Specific constructor for 9-elements vectors.
		template<int NN = N, typename std::enable_if<NN == 9, int>::type = 0>
		DYN_FUNC VectorND(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5, value_type r6, value_type r7, value_type r8, value_type r9)
		{
			static_assert(N == 9, "");
			this->data[0] = r1;
			this->data[1] = r2;
			this->data[2] = r3;
			this->data[3] = r4;
			this->data[4] = r5;
			this->data[5] = r6;
			this->data[6] = r7;
			this->data[7] = r8;
			this->data[8] = r9;
		}

		/// Specific constructor for 10-elements vectors.
		template<int NN = N, typename std::enable_if<NN == 10, int>::type = 0>
		DYN_FUNC VectorND(value_type r1, value_type r2, value_type r3, value_type r4, value_type r5, value_type r6, value_type r7, value_type r8, value_type r9, value_type r10)
		{
			static_assert(N == 10, "");
			this->data[0] = r1;
			this->data[1] = r2;
			this->data[2] = r3;
			this->data[3] = r4;
			this->data[4] = r5;
			this->data[5] = r6;
			this->data[6] = r7;
			this->data[7] = r8;
			this->data[8] = r9;
			this->data[9] = r10;
		}


		// iterator support
		DYN_FUNC iterator begin()
		{
			return data;
		}
		DYN_FUNC const_iterator begin() const
		{
			return data;
		}
		DYN_FUNC iterator end()
		{
			return data + N;
		}
		DYN_FUNC const_iterator end() const
		{
			return data + N;
		}

		// operator[]
		DYN_FUNC reference operator[](size_type i)
		{
#ifndef NDEBUG
			assert(i < N && "index in fixed_array must be smaller than size");
#endif
			return data[i];
		}
		DYN_FUNC const_reference operator[](size_type i) const
		{
#ifndef NDEBUG
			assert(i < N && "index in fixed_array must be smaller than size");
#endif
			return data[i];
		}

		// front() and back()
		DYN_FUNC reference front()
		{
			return data[0];
		}
		DYN_FUNC const_reference front() const
		{
			return data[0];
		}
		DYN_FUNC reference back()
		{
			return data[N - 1];
		}
		DYN_FUNC const_reference back() const
		{
			return data[N - 1];
		}

		// size is constant
		DYN_FUNC static size_type size()
		{
			return N;
		}
		DYN_FUNC static bool empty()
		{
			return N == 0;
		}
		DYN_FUNC static size_type max_size()
		{
			return N;
		}

		// direct access to data
		DYN_FUNC const T* getDataPtr() const
		{
			return data;
		}

		// assignment with type conversion
		template <typename T2>
		DYN_FUNC VectorND<T, N>& operator= (const VectorND<T2, N>& rhs)
		{
			//std::copy(rhs.begin(),rhs.end(), begin());
			for (size_type i = 0; i < N; i++)
				data[i] = rhs[i];
			return *this;
		}

		// assign one value to all elements
		DYN_FUNC inline void assign(const T& value)
		{
			//std::fill_n(begin(),size(),value);
			for (size_type i = 0; i < N; i++)
				data[i] = value;
		}

		//template<int NN = N, typename std::enable_if<NN>0,int>::type = 0>
		CPU_FUNC inline friend std::ostream& operator << (std::ostream& out, const VectorND<T, N>& a)
		{
			static_assert(N > 0, "Cannot create a zero size arrays");
			for (size_type i = 0; i < N - 1; i++)
				out << a.elems[i] << " ";
			out << a.elems[N - 1];
			return out;
		}

		CPU_FUNC inline friend std::istream& operator >> (std::istream& in, VectorND<T, N>& a)
		{
			for (size_type i = 0; i < N; i++)
				in >> a.elems[i];
			return in;
		}

		DYN_FUNC inline bool operator < (const VectorND& v) const
		{
			for (size_type i = 0; i < N; i++)
			{
				if (data[i] < v[i])
					return true;  // (*this)<v
				else if (data[i] > v[i])
					return false; // (*this)>v
			}
			return false; // (*this)==v
		}

		public:
			T data[N];

	};

	template<class T>
	DYN_FUNC inline VectorND<T, 2> makeFixedVector(const T& v0, const T& v1)
	{
		VectorND<T, 2> v;
		v[0] = v0;
		v[1] = v1;
		return v;
	}

	template<class T>
	DYN_FUNC inline VectorND<T, 3> makeFixedVector(const T& v0, const T& v1, const T& v2)
	{
		VectorND<T, 3> v;
		v[0] = v0;
		v[1] = v1;
		v[2] = v2;
		return v;
	}

	template<class T>
	DYN_FUNC inline VectorND<T, 4> makeFixedVector(const T& v0, const T& v1, const T& v2, const T& v3)
	{
		VectorND<T, 4> v;
		v[0] = v0;
		v[1] = v1;
		v[2] = v2;
		v[3] = v3;
		return v;
	}

	template<class T>
	DYN_FUNC inline VectorND<T, 5> makeFixedVector(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4)
	{
		VectorND<T, 5> v;
		v[0] = v0;
		v[1] = v1;
		v[2] = v2;
		v[3] = v3;
		v[4] = v4;
		return v;
	}

	template<class T>
	DYN_FUNC inline VectorND<T, 6> makeFixedVector(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5)
	{
		VectorND<T, 6> v;
		v[0] = v0;
		v[1] = v1;
		v[2] = v2;
		v[3] = v3;
		v[4] = v4;
		v[5] = v5;
		return v;
	}

	template<class T>
	DYN_FUNC inline VectorND<T, 7> makeFixedVector(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5, const T& v6)
	{
		VectorND<T, 7> v;
		v[0] = v0;
		v[1] = v1;
		v[2] = v2;
		v[3] = v3;
		v[4] = v4;
		v[5] = v5;
		v[6] = v6;
		return v;
	}

	template<class T>
	DYN_FUNC inline VectorND<T, 8> makeFixedVector(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5, const T& v6, const T& v7)
	{
		VectorND<T, 8> v;
		v[0] = v0;
		v[1] = v1;
		v[2] = v2;
		v[3] = v3;
		v[4] = v4;
		v[5] = v5;
		v[6] = v6;
		v[7] = v7;
		return v;
	}

	template<class T>
	DYN_FUNC inline VectorND<T, 9> makeFixedVector(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5, const T& v6, const T& v7, const T& v8)
	{
		VectorND<T, 9> v;
		v[0] = v0;
		v[1] = v1;
		v[2] = v2;
		v[3] = v3;
		v[4] = v4;
		v[5] = v5;
		v[6] = v6;
		v[7] = v7;
		v[8] = v8;
		return v;
	}

	template<class T>
	DYN_FUNC inline VectorND<T, 10> makeFixedVector(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5, const T& v6, const T& v7, const T& v8, const T& v9)
	{
		VectorND<T, 10> v;
		v[0] = v0;
		v[1] = v1;
		v[2] = v2;
		v[3] = v3;
		v[4] = v4;
		v[5] = v5;
		v[6] = v6;
		v[7] = v7;
		v[8] = v8;
		v[9] = v9;
		return v;
	}
} // namespace helper
