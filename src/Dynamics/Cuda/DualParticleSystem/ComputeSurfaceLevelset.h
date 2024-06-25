/**
 * Copyright 2023 Shusen Liu
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
#include "Module/ConstraintModule.h"
#include "Topology/SignedDistanceField.h"
#include "ParticleSystem/Module/Kernel.h"


namespace dyno
{
	template<typename TDataType>
	class ComputeSurfaceLevelset : public ConstraintModule
	{
		DECLARE_TCLASS(ComputeSurfaceLevelset, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ComputeSurfaceLevelset();
		~ComputeSurfaceLevelset();

		void constrain() override;
	
		DEF_ARRAY_IN(Coord, Points, DeviceType::GPU, "Point positions");

		DEF_INSTANCE_IN(SignedDistanceField<TDataType>, LevelSet, "A 3D signed distance field");

		DEF_VAR_IN(Real, GridSpacing, "Grid spacing");

	private:
		SpikyKernel<Real> m_kernel;
	
	};

	IMPLEMENT_TCLASS(ComputeSurfaceLevelset, TDataType)
}