#pragma once

#include "Vector.h"

namespace dyno {

	template<typename T>
	class Reduction
	{
	public:
		Reduction();

		static Reduction* Create(size_t n);
		~Reduction();

		T accumulate(T * val, size_t num);

		T maximum(T* val, size_t num);

		T minimum(T* val, size_t num);

		T average(T* val, size_t num);

	private:
		Reduction(size_t num);

		void allocAuxiliaryArray(size_t num);

		int getAuxiliaryArraySize(size_t n);
		
		uint m_num;
		
		T* m_aux;
		size_t m_auxNum;
	};

	template<>
	class Reduction<Vector3f>
	{
	public:
		Reduction();

		static Reduction* Create(size_t n);
		~Reduction();


		Vector3f accumulate(Vector3f * val, size_t num);

		Vector3f maximum(Vector3f* val, size_t num);

		Vector3f minimum(Vector3f* val, size_t num);

		Vector3f average(Vector3f* val, size_t num);

	private:
		void allocAuxiliaryArray(size_t num);


		uint m_num;
		
		float* m_aux;
		Reduction<float> m_reduce_float;
	};

	template<>
	class Reduction<Vector3d>
	{
	public:
		Reduction();

		static Reduction* Create(size_t n);
		~Reduction();


		Vector3d accumulate(Vector3d * val, size_t num);

		Vector3d maximum(Vector3d* val, size_t num);

		Vector3d minimum(Vector3d* val, size_t num);

		Vector3d average(Vector3d* val, size_t num);

	private:
		void allocAuxiliaryArray(size_t num);


		uint m_num;

		double* m_aux;
		Reduction<double> m_reduce_double;
	};
}
