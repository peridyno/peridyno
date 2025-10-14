/**
 * Copyright 2024-2025 Jingqi Zhang
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
#include "VolumeMacros.h"
#include "Module/ComputeModule.h"

namespace dyno {

	/**
	 * @brief Boolean operation of two level sets. Implementation of Zhang et al. "A Parallel Multiscale FIM Approach in Solving the Eikonal Equation on GPU", Computer Aided Design, 2025
	 */
	template<typename TDataType>
	class MultiscaleFastIterativeMethodForBoolean : public ComputeModule
	{
		DECLARE_TCLASS(MultiscaleFastIterativeMethodForBoolean, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		MultiscaleFastIterativeMethodForBoolean();
		~MultiscaleFastIterativeMethodForBoolean() override;

		DEF_ENUM(BoolType, BoolType, BoolType::Union, "Volume Bool Type");

		DEF_VAR(Real, Spacing, 0.01, "");  

		DEF_VAR(uint, Padding, 10, "");  

		DEF_VAR(uint, VCircle, 1, "");


	public:
		DEF_INSTANCE_IN(LevelSet<TDataType>, LevelSetA, "");
		DEF_INSTANCE_IN(LevelSet<TDataType>, LevelSetB, "");

		DEF_INSTANCE_OUT(LevelSet<TDataType>, LevelSet, "");

	protected:
		void compute() override;

	private:
		void makeLevelSet();

	};
}

