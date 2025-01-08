/**
 * Copyright 2021~2024 Shusen Liu
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
#include "VirtualParticleShiftingStrategy.h"
#include "ParticleSystem/Module/Kernel.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
namespace dyno {

	class Attribute;
	template<typename TDataType> class SummationDensity;


	/*
	*@Brief: Dual particle SPH solver to model free surface flow.
	*@Note : 1. This solver does not contain the virtual particle generator; 2. The graphic memory of the NVDIA GPU should be larger than 4GB.
	*@Paper: Liu et al., ACM Trans Graph (TOG). 2024. (A Dual-Particle Approach for Incompressible SPH Fluids) doi.org/10.1145/3649888
	*/

	template<typename TDataType>
	class DualParticleIsphModule : public ConstraintModule
	{
		DECLARE_TCLASS(DualParticleIsphModule, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		DualParticleIsphModule();
		~DualParticleIsphModule() override;

		void constrain() override;


	public:

		DEF_VAR(Real, RestDensity, Real(1000), "Reference density");

		DEF_VAR(Real, SamplingDistance, Real(0.005), "");

		DEF_VAR(Real, SmoothingLength, Real(0.0125), "Smoothing length in most cases");

		DEF_VAR(Real, PpeSmoothingLength, Real(0.0125), "Smoothing length in PPE solving")

		DEF_VAR_IN(Real, TimeStep, "Time Step");

		/**
		 * @brief Real Particle position
		 */
		DEF_ARRAY_IN(Coord, RPosition, DeviceType::GPU, "Input real particle position");

		/**
		* @brief Virtual Particle position
		*/
		DEF_ARRAY_IN(Coord, VPosition, DeviceType::GPU, "Input virtual particle position");

		/**
		 * @brief Real Particle velocitie
		 */
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Input particle velocity");

		/**
		* @brief Real Particle attribute 
		*/
		DEF_ARRAY_IN(Attribute, ParticleAttribute, DeviceType::GPU, "Real particle attribute");


		/**
		* @brief Real-Solid Particle Normal
		*/
		DEF_ARRAY_IN(Coord, BoundaryNorm, DeviceType::GPU, "Real-solid particle normal");


		/**
		* @brief Real Particle's Neghoboring Real Partilce
		*/
		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "Return real neighbor ids of real particles");

		/*
		*@brief Virtual Particle's Neghoboring Real Partilce 
		*/
		DEF_ARRAYLIST_IN(int, VRNeighborIds, DeviceType::GPU, "Return real neighbor ids of Virtual particles");

		/*
		*@brief Real Particle's Neghoboring Virtual Partilce
		*/
		DEF_ARRAYLIST_IN(int, RVNeighborIds, DeviceType::GPU, "Return Virtual neighbor ids of real particles");

		/*
		*@brief Virtual Particle's Neghoboring Virtual Partilces
		*/
		DEF_ARRAYLIST_IN(int, VVNeighborIds, DeviceType::GPU, "Return Virtual neighbor ids of virtual particles");

		/*
		*@brief Virtual Particle's Boolean Quantity (Debug) 
		*/
		DEF_ARRAY_OUT(bool, VirtualBool, DeviceType::GPU, "Virtual Particle's Boolean Quantity");

		/*
		*@brief Virtual Particle's Float Quantity (Debug)
		*/
		DEF_ARRAY_OUT(Real, VirtualWeight, DeviceType::GPU, "Virtual Particle's Float Quantity");


		DEF_VAR(Real, ResidualThreshold, 0.001f, "Convergence threshold for the pressure Poisson Equation");

	private:

		bool initializeImpl() override;
		bool virtualArraysResize();
		bool realArraysResize();

		bool init_flag = false;


		CubicKernel<Real> kernel;
		Real m_maxAii;
		Real m_particleMass;
		Real m_v_particleMass;
		Real m_airPressure;

		DArray<bool> m_solidVirtualPaticleFlag;
		DArray<Coord> m_virtualVelocity;
		DArray<Real> m_source;
		DArray<Real> m_Ax;
		DArray<bool> m_virtualAirFlag;
		DArray<Real> m_r;
		DArray<Real> m_p;
		DArray<Real> m_pressure;
		DArray<Real> m_residual;
		DArray<Coord> m_Gp;
		DArray<Coord> m_GpNearSolid;

		unsigned int virtualNumber_old = 0;

		DArray<Real> m_virtualAirWeight;
		DArray<Real> m_Aii;

		Reduction<Real>* m_reduce;
		Arithmetic<Real>* m_arithmetic;
		//Arithmetic<Real>* m_arithmetic_r;

		unsigned int frag_number = 0;
		Real max_Aii;

		std::shared_ptr<SummationDensity<TDataType>> m_summation;

		std::shared_ptr<SummationDensity<TDataType>> m_vv_summation;

		std::shared_ptr<SummationDensity<TDataType>> m_vr_summation;

	};
}