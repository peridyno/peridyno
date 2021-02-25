#pragma once

#include "Vector.h"

namespace dyno {

	template<typename T>
	class Reduction
	{
	public:
		Reduction();

		static Reduction* Create(int n);
		~Reduction();

		T accumulate(T * val, int num);

		T maximum(T* val, int num);

		T minimum(T* val, int num);

		T average(T* val, int num);

	private:
		Reduction(unsigned num);

		void allocAuxiliaryArray(int num);

		int getAuxiliaryArraySize(int n);
		
		unsigned m_num;
		
		T* m_aux;
		int m_auxNum;
	};


	template class Reduction<int>;
	template class Reduction<float>;
	template class Reduction<double>;

	template<>
	class Reduction<Vector3f>
	{
	public:
		Reduction();

		static Reduction* Create(int n);
		~Reduction();


		Vector3f accumulate(Vector3f * val, int num);

		Vector3f maximum(Vector3f* val, int num);

		Vector3f minimum(Vector3f* val, int num);

		Vector3f average(Vector3f* val, int num);

	private:
		void allocAuxiliaryArray(int num);


		unsigned m_num;
		
		float* m_aux;
		Reduction<float> m_reduce_float;
	};

	template<>
	class Reduction<Vector3d>
	{
	public:
		Reduction();

		static Reduction* Create(int n);
		~Reduction();


		Vector3d accumulate(Vector3d * val, int num);

		Vector3d maximum(Vector3d* val, int num);

		Vector3d minimum(Vector3d* val, int num);

		Vector3d average(Vector3d* val, int num);

	private:
		void allocAuxiliaryArray(int num);


		unsigned m_num;

		double* m_aux;
		Reduction<double> m_reduce_double;
	};
}
