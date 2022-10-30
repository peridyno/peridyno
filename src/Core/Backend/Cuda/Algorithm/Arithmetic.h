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
		
		T Dot(DArray<T>& xArr, DArray<T>& yArr);
		
		~Arithmetic();
	private:
		Arithmetic(int n);

		Reduction<T>* m_reduce;
		DArray<T> m_buf;
	};

}
