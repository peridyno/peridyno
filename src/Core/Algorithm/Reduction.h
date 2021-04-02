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
	class Reduction<Vector3f>
	{
	public:
		Reduction();

		static Reduction* Create(uint n);
		~Reduction();

		Vector3f accumulate(Vector3f * val, uint num);

		Vector3f maximum(Vector3f* val, uint num);

		Vector3f minimum(Vector3f* val, uint num);

		Vector3f average(Vector3f* val, uint num);

	private:
		void allocAuxiliaryArray(uint num);


		uint m_num;
		
		float* m_aux;
		Reduction<float> m_reduce_float;
	};

	template<>
	class Reduction<Vector3d>
	{
	public:
		Reduction();

		static Reduction* Create(uint n);
		~Reduction();

		Vector3d accumulate(Vector3d * val, uint num);

		Vector3d maximum(Vector3d* val, uint num);

		Vector3d minimum(Vector3d* val, uint num);

		Vector3d average(Vector3d* val, uint num);

	private:
		void allocAuxiliaryArray(uint num);


		uint m_num;

		double* m_aux;
		Reduction<double> m_reduce_double;
	};
}
