#include "Arithmetic.h"
#include "Function2Pt.h"

namespace dyno {

	template<typename T>
	Arithmetic<T>::Arithmetic(int n)
		: m_reduce(NULL)
	{
		m_reduce = Reduction<T>::Create(n);
		m_buf.resize(n);
	}

	template<typename T>
	Arithmetic<T>::~Arithmetic()
	{
		if (m_reduce != NULL)
		{
			delete m_reduce;
		}
		//if (m_buf != NULL)
		m_buf.clear();
	}


	template<typename T>
	Arithmetic<T>* Arithmetic<T>::Create(int n)
	{
		return new Arithmetic<T>(n);
	}

	template<typename T>
	T Arithmetic<T>::Dot(DArray<T>& xArr, DArray<T>& yArr)
	{
		if (m_buf.size() != xArr.size())
		{
			m_buf.resize(xArr.size());
		}
		Function2Pt::multiply(m_buf, xArr, yArr);
		return m_reduce->accumulate(m_buf.begin(), m_buf.size());
	}

	template class Arithmetic<int>;
	template class Arithmetic<float>;
	template class Arithmetic<double>;
}