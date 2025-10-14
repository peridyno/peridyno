/**
 * Copyright 2025 Shusen Liu
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
#include "VirtualParticleGenerator.h"
#include "ParticleSystem/Module/Kernel.h"
#include "ParticleSystem/Module/SummationDensity.h"
#include "ParticleSystem/Module/Kernel.h"
#include "Collision/NeighborPointQuery.h"
#include "VirtualParticleStructure.h"


namespace dyno {

	template<typename TDataType> class SummationDensity;

	template<typename TDataType>
	class VirtualFissionFusionStrategy : public VirtualParticleGenerator<TDataType>
	{
	/*
	*@Brief: Adaptive Fission Fusion Strategy in Dual-particle SPH method. (Virtual paritlce genorator).
	*@Paper: Liu et al., Pacific Graphics 2025 (Conference Track). An Adaptive Particle Fission-Fusion Approach for Dual-Particle SPH Fluid. 
	*/



		DECLARE_TCLASS(VirtualFissionFusionStrategy, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VirtualFissionFusionStrategy();
		~VirtualFissionFusionStrategy() override;

		void constrain() override;

		void fissionJudger();		

		void splitParticleArray();

		void constructFissionVirtualParticles();

		void mergeVirtualParticles();

		DEF_VAR_IN(Real, TimeStep, "Time Step");

		DEF_VAR_IN(Real, SmoothingLength, "Smoothing Length");

		DEF_VAR_IN(Real, SamplingDistance, "Particle sampling distance");

		DEF_VAR(Real, RestDensity, 1000, "Reference density");

		DEF_ARRAY_OUT(Real, Density, DeviceType::GPU, "Final particle density");

		DEF_ARRAY_IN(Coord, RPosition, DeviceType::GPU, "Input real particle position");

		DEF_ARRAY_IN(Coord, RVelocity, DeviceType::GPU, "Input real particle velocity");

		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "Neighboring particles' ids");

		DEF_ARRAY_IN(bool, ThinSheet, DeviceType::GPU, "");

		DEF_ARRAY_IN(Real, ThinFeature, DeviceType::GPU, "");

		DEF_VAR(Real, TransitionRegionThreshold, 0.001, "");

		DEF_VAR_IN(uint, FrameNumber, "");

		DEF_VAR(Real, MinDist, 0.002, "");

		DEF_ARRAY_OUT(Coord, CandidateVirtualPoints, DeviceType::GPU, "");

		DEF_ARRAY_OUT(bool, VirtualPointType, DeviceType::GPU, "");

		DEF_VAR(bool, deleteRepeatPoints, true, "");

		DECLARE_ENUM(CandidatePointCount,
		neighbors_8 = 8,
			neighbors_27 = 27,
			neighbors_33 = 33,
			neighbors_125 = 125
			);

		DECLARE_ENUM(StretchedRegionCriteria,
			Divergecne = 0,
			ThinSheet = 1,
			Hybrid = 2
			);

		DEF_ENUM(CandidatePointCount, CandidatePointCount, CandidatePointCount::neighbors_33, "Candidate Point Count");

		DEF_ENUM(StretchedRegionCriteria, StretchedRegionCriteria, StretchedRegionCriteria::Hybrid, "Stretched Region Criteria");


	private:

		void DeleteRepeatVirtualPosition();

		void resizeArrays(int num);

		CubicKernel<Real> kernel;

		SpikyKernel<Real> SpikyKernel;

		DArray<Real> mDivergence;					/*@brief Velocity Divergence of real particle*/

		DArray<uint> mCurrentParticleStates;		/*@brief Particle state:: 0: unclear; 1: compressed/fusion; 2. stretched/fisson*/; 

		DArray<uint> mPreParticleStates;			/*@brief Particle in previrous frame*/

		DArray<Coord> mVirtualPoints;		/*@brief Virtual Particles*/

		DArray<Coord> mAnchorPoints;

		DArray<uint32_t> mAnchorPointCodes;

		DArray<uint32_t> mNonRepeatedCount;

		DArray<uint32_t> mCandidateCodes;

		DArray<Coord> mFissionParticles;

		DArray<Coord> mFussionParticles;

		DArray<uint> mFissionParticleIds;

		DArray<uint> mFussionParticleIds;

		DArray<Coord> mFissionVirtualParicles;

		DArray<Coord> mFussionVirtualParticles;

		
		Coord origin;

		uint FissionVirtualNum;

		uint FussionVirtualNum;


		DArray<uint> fissions;

		DArray<uint> fussions;

		DArray<uint> ArrayPointer;

		std::shared_ptr<SummationDensity<TDataType>> mSummation;

	};
}