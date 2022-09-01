#include <cuda_runtime.h>
#include "Algorithm/Reduction.h"
#include "Algorithm/Arithmetic.h"
#include "Algorithm/Function2Pt.h"
#include "SMAlgorithm.h"
#include <windows.h>

namespace dyno
{
	template <typename VarType>
	void SparseMatrix<VarType>::clear()
	{
		my_A.clear();
		my_transposedA.clear();
		my_b.clear();
		my_x.clear();
	}

	template <typename VarType>
	void SparseMatrix<VarType>::assign_cgls(CArray<VarType>& s_b, std::vector<std::map<int, VarType>>& s_matrix, std::vector<std::map<int, VarType>>& s_matrix_transposed)
	{
		my_A.assign(s_matrix);
		my_transposedA.assign(s_matrix_transposed);
		my_b.assign(s_b);

		my_x.resize(s_b.size());
		my_x.reset();
	}

	template <typename VarType>
	void SparseMatrix<VarType>::CGLS(int i_max, VarType threshold)
	{
		int system_size = my_b.size();
		//printf("system size is: %d \r\n", system_size);

		Arithmetic<VarType>*m_arithmetic = Arithmetic<VarType>::Create(system_size);

		//x_0
		SparseV x_0(system_size);
		x_0.reset();

		int itor = 0;

		VarType delta_0 = 10, delta_new = 10, delta_old = 10;
		SparseV b_new, temp1, temp2, my_r, my_d, my_q;
		b_new.resize(system_size); temp1.resize(system_size); temp2.resize(system_size); my_r.resize(system_size); my_d.resize(system_size); my_q.resize(system_size); 
		b_new.reset(); temp1.reset(); temp2.reset(); my_r.reset(); my_d.reset(); my_q.reset();

		//compute b_new
		multiply_SM_by_vector<VarType>(my_transposedA, my_b, b_new);

		//compute r=b_new-transposed(A)*A*x_0
		multiply_SM_by_vector<VarType>(my_A, x_0, temp1);
		multiply_SM_by_vector<VarType>(my_transposedA, temp1, temp2);
		Function2Pt::subtract(my_r, b_new, temp2);

		my_d.assign(my_r);

		//compute delta
		delta_new = m_arithmetic->Dot(my_r, my_r);
		delta_0 = delta_new;
		while ((itor < i_max) && (delta_new > (threshold*threshold*delta_0)))
		{
			//compute alpha and x(i+1)
			multiply_SM_by_vector<VarType>(my_A, my_d, my_q);
			VarType alpha = delta_new / m_arithmetic->Dot(my_q, my_q);
			Function2Pt::saxpy(my_x, my_d, my_x, alpha);
			//compute r
			if (itor % 50 == 0)
			{
				//compute r=b_new-transposed(A)*A*x
				temp1.reset(); temp2.reset();
				multiply_SM_by_vector<VarType>(my_A, my_x, temp1);
				multiply_SM_by_vector<VarType>(my_transposedA, temp1, temp2);
				Function2Pt::subtract(my_r, b_new, temp2);
			}
			else
			{
				temp1.reset();
				multiply_SM_by_vector<VarType>(my_transposedA, my_q, temp1);
				Function2Pt::saxpy(my_r, temp1, my_r, -alpha);
			}

			delta_old = delta_new;
			delta_new = m_arithmetic->Dot(my_r, my_r);
			VarType beta = delta_new / delta_old;
			Function2Pt::saxpy(my_d, my_d, my_r, beta);

			itor++;
		}
		//std::printf("the iterations of CGLS is: %d \n",itor);
		delete m_arithmetic;

		x_0.clear();b_new.clear();temp1.clear();temp2.clear();my_r.clear();my_d.clear();my_q.clear();
	}
}
