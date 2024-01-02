/**
 * Copyright 2017-2021 Xiaowei He
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

#include "Collision/Attribute.h"

//Framework
#include "Module/ConstraintModule.h"

namespace dyno 
{
	template<typename TDataType> class SummationDensity;
	template<typename TDataType> class Reduction;
	template<typename TDataType> class Arithmetic;

	/**
	 *	\class	VariationalApproximateProjection
	 *	\brief	Projection-based solver.
	 *
	 *  @brief Implementation of an approximate projection method for incompressible free-surface flows under a variational staggered particle framework.
	 * 			Refer to "A Variational Staggered Particle Framework for Incompressible Free-Surface Flows", [arXiv:2001.09421], 2020.
	 */
	template<typename TDataType>
	class VariationalApproximateProjection : public ConstraintModule
	{
		DECLARE_TCLASS(VariationalApproximateProjection, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VariationalApproximateProjection();
		~VariationalApproximateProjection() override;
		
		void constrain() override;

	public:
		DEF_VAR(Real, RestDensity, Real(1000), "");

		DEF_VAR_IN(Real, TimeStep, "Time step size");

		DEF_VAR_IN(Real, SamplingDistance, "");

		DEF_VAR_IN(Real, SmoothingLength, "");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, Normal, DeviceType::GPU, "");

		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "");
		
		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "");


// 	protected:
// 		bool initializeImpl() override;

	private:
		Real mAlphaMax;
		Real mAMax;
		Real mAirPressure = 0.0f;

		Real mTangential = 0.1f;
		Real mSeparation = 0.1f;
		//Real mRestDensity = 1000.0f;

		//Refer to "A Nonlocal Variational Particle Framework for Incompressible Free Surface Flows" for their exact meanings
		DArray<Real> mAlpha;
		DArray<Real> mAii;
		DArray<Real> mAiiFluid;
		DArray<Real> mAiiTotal;

		DArray<Real> mPressure;
		DArray<Real> mDivergence;

		//Indicate whether a particle is near the free surface boundary.
		DArray<bool> mIsSurface;

		//Internal variables used to solve the linear system of equations with a conjugate gradient method.
		DArray<Real> m_y;
		DArray<Real> m_r;
		DArray<Real> m_p;

		Reduction<Real>* m_reduce;
		Arithmetic<Real>* m_arithmetic;

		std::shared_ptr<SummationDensity<TDataType>> mDensityCalculator;
	};

	IMPLEMENT_TCLASS(VariationalApproximateProjection, TDataType)
}