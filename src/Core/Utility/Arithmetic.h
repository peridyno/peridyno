#pragma once
#include "Reduction.h"
#include "Array/Array.h"

namespace dyno 
{
	template<typename T>
	class Arithmetic
	{
	public:
		Arithmetic(const Arithmetic &) = delete;
		Arithmetic& operator=(const Arithmetic &) = delete;

		static Arithmetic* Create(int n);
		
		T Dot(GArray<T>& xArr, GArray<T>& yArr);
		
		~Arithmetic();
	private:
		Arithmetic(int n);

		Reduction<T>* m_reduce;
		GArray<T> m_buf;
	};

	template class Arithmetic<int>;
	template class Arithmetic<float>;
	template class Arithmetic<double>;
}
