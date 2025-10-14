/**
 * Copyright 2021~2025 Shusen Liu
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
#include "Module/ComputeModule.h"
#include "Collision/Attribute.h"
#include "ParticleSystem/Module/ParticleApproximation.h"
#include "ParticleSystem/Module/SummationDensity.h"

namespace dyno {



	template<typename TDataType> class SummationDensity;

	template<typename TDataType>
	class ThinFeature : public ParticleApproximation<TDataType>
	{

		/*
		*@Brief : Determin whether a fluid particle is located in a thin sheet.
		*@Paper2£ºLiu et al.2025 An Adaptive Particle Fission-Fusion Approach for Dual-Particle SPH Fluid. 
		*/

		DECLARE_TCLASS(ThinFeature, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		ThinFeature();
		~ThinFeature() override;

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");

		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "Neighboring particles' ids");

		DEF_ARRAY_OUT(bool, ThinSheet, DeviceType::GPU, "");

		DEF_ARRAY_OUT(Real, ThinFeature, DeviceType::GPU, "");

		DEF_ARRAY_OUT(Real, Density, DeviceType::GPU, "");

		DEF_VAR(Real, Threshold, 0.01f, "");

		DEF_VAR(Real, RestDensity, 1000.0f, "");

		void compute() override;

	protected:

		void resizeArray(int num);

	private:

		DArray<Matrix> mDistributMat;
		DArray<Coord> mEigens;

		std::shared_ptr<SummationDensity<TDataType>> mSummation;
	};
}