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
	template<typename VarType>
	void multiply_transposedSM_by_vector(DArrayMap<VarType>& matrix_a, DArray<VarType>& a, DArray<VarType>& Aa);

	template<typename VarType>
	void multiply_SM_by_vector(DArrayMap<VarType>& matrix_a, DArray<VarType>& a, DArray<VarType>& Aa);

}  