/**
 * Copyright 2022 Nurshat Menglik
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
 * 
 * @author: Xiaowei He
 * @date: 2023/5/14
 * @description: renamed to SemiAnalyticalParticleShifting
 */


#pragma once
#include "Module/TopologyModule.h"

#include "Collision/Attribute.h"
#include "ParticleSystem/Module/ParticleApproximation.h"

namespace dyno
{
	template<typename TDataType> class SemiAnalyticalSummationDensity;
	
	template<typename TDataType>
	class SemiAnalyticalParticleShifting : public virtual ParticleApproximation<TDataType>
	{
		DECLARE_TCLASS(SemiAnalyticalParticleShifting, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		SemiAnalyticalParticleShifting();
		~SemiAnalyticalParticleShifting() override;

		void compute()override;

	public:
		DEF_VAR(uint, InterationNumber, 10, "");

		DEF_VAR(Real, Inertia, Real(0.1), "inertia");

		DEF_VAR(Real, Bulk, Real(0.5), "bulk");

		DEF_VAR(Real, SurfaceTension, Real(0.03), "surface tension");

		DEF_VAR(Real, AdhesionIntensity, Real(30.0), "adhesion");

		DEF_VAR(Real, RestDensity, Real(1000.0), "Rest Density");

	public:
		DEF_VAR_IN(Real, TimeStep, "Time step size");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Particle position");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Particle velocity");

		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "Particle attribute");

		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "");

		DEF_ARRAYLIST_IN(int, NeighborTriIds, DeviceType::GPU, "triangle neighbors");

		DEF_ARRAY_IN(Triangle, TriangleInd, DeviceType::GPU, "triangle_index");
		DEF_ARRAY_IN(Coord, TriangleVer, DeviceType::GPU, "triangle_vertex");

	private:
		DArray<Real> mLambda;
		DArray<Real> mTotalW;
		DArray<Coord> mBoundaryDir;
		DArray<Real> mBoundaryDis;
		DArray<Coord> mDeltaPos;
		DArray<Coord> mPosBuf;
		DArray<Coord> mAdhesionEP;
		std::shared_ptr<SemiAnalyticalSummationDensity<TDataType>> mCalculateDensity;
	};
}