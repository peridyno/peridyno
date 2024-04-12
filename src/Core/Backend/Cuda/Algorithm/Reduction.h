#pragma once

#include "Vector.h"

namespace dyno {

	template<typename T>
	class Reduction
	{
	public:
		Reduction();

		static Reduction* Create(const uint n);
		~Reduction();

		T accumulate(const T * val, const uint num);

		T maximum(const T* val, const uint num);

		T minimum(const T* val, const uint num);

		T average(const T* val, const uint num);

	private:
		Reduction(const uint num);

		void allocAuxiliaryArray(const uint num);

		uint getAuxiliaryArraySize(const uint n);
		
		uint m_num;
		
		T* m_aux;
		uint m_auxNum;
	};

	template<>
	class Reduction<Vec3f>
	{
	public:
		Reduction();

		static Reduction* Create(const uint n);
		~Reduction();

		Vec3f accumulate(const Vec3f * val, const uint num);

		Vec3f maximum(const Vec3f* val, const uint num);

		Vec3f minimum(const Vec3f* val, const uint num);

		Vec3f average(const Vec3f* val, const uint num);

	private:
		void allocAuxiliaryArray(const uint num);


		uint m_num;
		
		float* m_aux;
		Reduction<float> m_reduce_float;
	};

	template<>
	class Reduction<Vec3d>
	{
	public:
		Reduction();

		static Reduction* Create(const uint n);
		~Reduction();

		Vec3d accumulate(const Vec3d * val, const uint num);

		Vec3d maximum(const Vec3d* val, const uint num);

		Vec3d minimum(const Vec3d* val, const uint num);

		Vec3d average(const Vec3d* val, const uint num);

	private:
		void allocAuxiliaryArray(const uint num);


		uint m_num;

		double* m_aux;
		Reduction<double> m_reduce_double;
	};
}
