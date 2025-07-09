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
#include "SdfSampler.h"
#include "Collision/NeighborPointQuery.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"
#include "Module/PoissionDiskPositionShifting.h"

namespace dyno
{
	/*
	*@brief Using the position-based constant-density constraint method, implement Poisson-disk sampling on GPU.
	* 
	*/

	template<typename TDataType>
	class DevicePoissonDiskSampler : public SdfSampler<TDataType>
	{
		DECLARE_TCLASS(DevicePoissonDiskSampler, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DevicePoissonDiskSampler();
		~DevicePoissonDiskSampler();

		void resetStates() override;

		DEF_ARRAY_STATE(Coord, Position, DeviceType::GPU, "A set of points whose neighbors will be required for");

		//DEF_ARRAY_STATE(Coord, VirtualVelocity, DeviceType::GPU, "");

		DEF_VAR(Real, Delta, 0.001, "");

		DEF_VAR_STATE(Real, NeighborLength, 0.01f, "Length of Neighborhood-searching");

		DEF_ARRAYLIST_STATE(int, NeighborIds, DeviceType::GPU, "Return neighbor ids");

		DEF_ARRAY_OUT(Real, Density, DeviceType::GPU, "Return the particle density");

		DEF_VAR(bool, ConstraintDisable, false, "Disable the constant-density ");

		DEF_VAR(int, MaxIteration, 20, "");

	private:

		//void updatePositions();

		void imposeConstraint();

		void deleteOutsidePoints();

		void deleteCollisionPoints();

		Real minimumDistanceEstimation();

		void resizeArrays(int num);


		//DArray<Coord> m_Positions;

		std::shared_ptr<NeighborPointQuery<TDataType>> m_neighbor;

		std::shared_ptr<PoissionDiskPositionShifting<TDataType>> m_constraint;
		
		std::shared_ptr<ImplicitViscosity<TDataType>> ptr_viscosity;

		int m_seed_offset = 0;


		DArray<Real> mMinimumDistances;
		DArray<Coord> mPointsInsideSdf;
		DArray<int> mInsideSdfCounters;

		Reduction<Real> mReduceReal;
		Reduction<int> mReduceInt;
		Scan<int> mScan;

		std::shared_ptr<DistanceField3D<TDataType>> m_inputSDF;


	};

IMPLEMENT_TCLASS(DevicePoissonDiskSampler, TDataType);
}