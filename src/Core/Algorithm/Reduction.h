#pragma once

#include "Vector.h"

namespace dyno {

	template<typename T>
	class Reduction
	{
	public:
		Reduction();

		static Reduction* Create(uint n);
		~Reduction();

		T accumulate(T * val, uint num);

		T maximum(T* val, uint num);

		T minimum(T* val, uint num);

		T average(T* val, uint num);

	private:
		Reduction(uint num);

		void allocAuxiliaryArray(uint num);

		uint getAuxiliaryArraySize(uint n);
		
		uint m_num;
		
		T* m_aux;
		uint m_auxNum;
	};

	template<>
	class Reduction<Vec3f>
	{
	public:
		Reduction();

		static Reduction* Create(uint n);
		~Reduction();

		Vec3f accumulate(Vec3f * val, uint num);

		Vec3f maximum(Vec3f* val, uint num);

		Vec3f minimum(Vec3f* val, uint num);

		Vec3f average(Vec3f* val, uint num);

	private:
		void allocAuxiliaryArray(uint num);


		uint m_num;
		
		float* m_aux;
		Reduction<float> m_reduce_float;
	};

	template<>
	class Reduction<Vec3d>
	{
	public:
		Reduction();

		static Reduction* Create(uint n);
		~Reduction();

		Vec3d accumulate(Vec3d * val, uint num);

		Vec3d maximum(Vec3d* val, uint num);

		Vec3d minimum(Vec3d* val, uint num);

		Vec3d average(Vec3d* val, uint num);

	private:
		void allocAuxiliaryArray(uint num);


		uint m_num;

		double* m_aux;
		Reduction<double> m_reduce_double;
	};
}
