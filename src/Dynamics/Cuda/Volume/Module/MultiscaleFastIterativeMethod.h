/**
 * Copyright 2024-2025 Jingqi Zhang & Zihao Zhou
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
#include "Topology/LevelSet.h"
#include "Topology/TriangleSet.h"

#include "Module/ComputeModule.h"

namespace dyno {
	typedef Vector<unsigned int, 3> Vec3ui;
	typedef Vector<int, 3> Vec3i;

	typedef CArray3D<unsigned int> CArray3ui;
	typedef CArray3D<float> CArray3f;
	typedef CArray3D<int> CArray3i;

	/**
	 * @brief Generation of a level set. Implementation of Zhang et al. "A Parallel Multiscale FIM Approach in Solving the Eikonal Equation on GPU", Computer Aided Design, 2025
	 */
	template<typename TDataType>
	class MultiscaleFastIterativeMethod : public ComputeModule
	{
		DECLARE_TCLASS(MultiscaleFastIterativeMethod, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		MultiscaleFastIterativeMethod();
		~MultiscaleFastIterativeMethod() override;

		DEF_VAR(Real, Spacing, 0.01f, "");

		DEF_VAR(uint, Padding, 30, "");

		DEF_VAR(uint, VCircle, 1, "");

	public:
		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "");

		DEF_INSTANCE_OUT(LevelSet<TDataType>, LevelSet, "");

	protected:
		void compute() override;

	private:
		void makeLevelSet();
	};
}

