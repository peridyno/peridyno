/**
 * Copyright 2024 Xiaowei He
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
#include "VolumeMacros.h"

#include "Topology/LevelSet.h"

#include "Module/ComputeModule.h"

namespace dyno {
	/**
	 * @brief this class implements a GPU-based fast marching method to do boolean between two distance fields.
	 */
	template<typename TDataType>
	class FastMarchingMethodGPU : public ComputeModule
	{
		DECLARE_TCLASS(FastMarchingMethodGPU, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		FastMarchingMethodGPU();
		~FastMarchingMethodGPU() override;

		DEF_VAR(Real, Spacing, 0.05, "");

		DEF_ENUM(BoolType, BoolType, BoolType::Union, "Volume Bool Type");

		/**
		 * @brief MarchingNumber controls the distance to advance, in case its value is small while the domain is large, it is not guaranteed the whole domain is properly initialized.
		 *			Nevertheless, it it ensured that the value near the 0-level set is accurate with a proper value of MarchingNumber.
		 */
		DEF_VAR(uint, MarchingNumber, 20, "");

	public:
		DEF_INSTANCE_IN(LevelSet<TDataType>, LevelSetA, "");
		DEF_INSTANCE_IN(LevelSet<TDataType>, LevelSetB, "");

		DEF_INSTANCE_OUT(LevelSet<TDataType>, LevelSet, "");

	protected:
		void compute() override;

	private:
		DArray3D<GridType> mGridType;
		DArray3D<bool> mOutside;
	};
}
