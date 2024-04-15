/**
 * Copyright 2024 Shusen Liu
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
#include "Algorithm/Reduction.h"
#include "Algorithm/Functional.h"
#include "Algorithm/Arithmetic.h"
#include "Collision/Attribute.h"

namespace dyno {

	template<typename TDataType>
	class ApproximateImplicitViscosity : public ConstraintModule
	{
		DECLARE_TCLASS(ApproximateImplicitViscosity, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;


		ApproximateImplicitViscosity() ;
		~ApproximateImplicitViscosity() override;

		void constrain() override;

		bool SetCross();

	public:

		DEF_VAR(Real, Viscosity, Real(10), "Dynamic Viscosity");

		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Input real particle position");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Input particle velocity");
	
		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "Input particle velocity");

		DEF_VAR(Real, LowerBoundViscosity, 1.0f, "Viscosity lower bound for Non-Newtonian fluid");

		DEF_VAR(Real, CrossK, 1, "K for Non-Newtonian fluid (Cross model)");

		DEF_VAR(Real, CrossN, -1, "N for Non-Newtonian fluid (Cross model)");

		DECLARE_ENUM(FluidType,
			NewtonianFluid = 0,
			NonNewtonianFluid = 1
			);

		DEF_ENUM(FluidType, FluidType, FluidType::NewtonianFluid, "Fluid type.");

	private:
		DArray<Real> m_viscosity;


		/*
		* @brief Important coefficients in the Cross model.
		*/
		Real CrossVisCeil;
		Real CrossVisFloor;
		Real CrossVisMag;
		Real Cross_K;
		Real Cross_N;

		DEF_VAR(Real, SmoothingLength, Real(0.0125), "");
		
		DEF_VAR(Real, RestDensity, Real(1000), "Reference density");

		DEF_VAR(Real, SamplingDistance, Real(0.005), "");

		DEF_VAR_IN(Real, TimeStep, "");

	private:

		bool IsCrossReady = false;

		Real m_maxAlpha;
		Real m_maxA;

		DArray<Real> m_alpha;
		DArray<Coord> velOld;
		DArray<Coord> velBuf;
		DArray<Real> v_y;
		DArray<Real> v_r;
		DArray<Real> v_p;
		DArray<Coord> v_pv;
		DArray<Real> m_VelocityReal;
		Reduction<Real>* m_reduce;
		Arithmetic<Real>* m_arithmetic_v;
	};


	IMPLEMENT_TCLASS(ApproximateImplicitViscosity, TDataType)
}