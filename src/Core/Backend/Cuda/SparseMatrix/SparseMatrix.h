/**
 * Copyright 2021 Lixin Ren
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "Array/Array.h"
#include "Array/ArrayMap.h"



namespace dyno 
{
	template <typename VarType>
	class SparseMatrix
	{
	public:
		typedef DArrayMap<VarType> SparseM;
		typedef DArray<VarType> SparseV;

		SparseMatrix() {};

		/*!
		*	\brief	Do not release memory here, call clear() explicitly.
		*/
		~SparseMatrix() {};

		/*!
		*	\brief	Free allocated memory.	Should be called before the object is deleted.
		*/
		void clear();

		void assign_cgls(CArray<VarType>& s_b, std::vector<std::map<int, VarType>>& s_matrix, std::vector<std::map<int, VarType>>& s_matrix_transposed);

	    void CGLS(int i_max, VarType threshold);
		
		const SparseV& X() const { return my_x; }

	private:
		SparseM my_A;
		SparseM my_transposedA;

		SparseV my_x;
		SparseV my_b;
	};
}  

#include "SparseMatrix.inl"