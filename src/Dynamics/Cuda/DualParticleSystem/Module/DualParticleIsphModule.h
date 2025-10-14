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
	*@Brief : Dual particle SPH solver to model free surface flow.
	*@Note  : 1. This solver does not contain the virtual particle generator; 2. The graphic memory of the NVDIA GPU should be larger than 4GB.
	*@Paper1: Liu et al., ACM Trans Graph (TOG). 2024. A Dual-Particle Approach for Incompressible SPH Fluids. doi.org/10.1145/3649888
	*@Paper2£ºLiu et al., Pacific Graphics 2025 (Conference Track). An Adaptive Particle Fission-Fusion Approach for Dual-Particle SPH Fluid.  https://diglib.eg.org/handle/10.2312/pg20251269
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
		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "Real neighbor ids of real particles");

		/*
		*@brief Virtual Particle's Neghoboring Real Partilce 
		*/
		DEF_ARRAYLIST_IN(int, VRNeighborIds, DeviceType::GPU, "Real neighbor ids of Virtual particles");

		/*
		*@brief Real Particle's Neghoboring Virtual Partilce
		*/
		DEF_ARRAYLIST_IN(int, RVNeighborIds, DeviceType::GPU, "Virtual neighbor ids of real particles");

		/*
		*@brief Virtual Particle's Neghoboring Virtual Partilces
		*/
		DEF_ARRAYLIST_IN(int, VVNeighborIds, DeviceType::GPU, "Virtual neighbor ids of virtual particles");

		/*
		*@brief Virtual Particle's Boolean Quantity (Debug) 
		*/
		DEF_ARRAY_OUT(bool, VirtualBool, DeviceType::GPU, "Virtual Particle's Boolean Quantity");

		/*
		*@brief Virtual Particle's Float Quantity (Debug)
		*/
		DEF_ARRAY_OUT(Real, VirtualWeight, DeviceType::GPU, "Virtual Particle's Float Quantity");

		DEF_VAR(Real, ResidualThreshold, 0.0001f, "Convergence threshold for the pressure Poisson Equation");

		DEF_VAR(bool, WarmStart, true, "");

	private:

		bool initializeImpl() override;

		bool virtualArraysResize();

		bool realArraysResize();

		Real m_particleMass;
		Real m_v_particleMass;
		Real m_airPressure;
		Real max_Aii;

		DArray<Coord> m_virtualVelocity;
		DArray<Real> m_source;
		DArray<Real> m_Ax;
		DArray<bool> m_virtualAirFlag;
		DArray<Real> m_r;
		DArray<Real> m_p;
		DArray<Real> m_pressure;
		DArray<Coord> m_Gp;
		DArray<Coord> m_GpNearSolid;
		DArray<Real> m_RealPressure;
		DArray<Real> m_Aii;

		CubicKernel<Real> kernel;

		Reduction<Real>* m_reduce;
		Reduction<Real>* m_reduce_r;
		Arithmetic<Real>* m_arithmetic;
		Arithmetic<Real>* m_arithmetic_r;

		unsigned int frag_number = 0;

		/// Fluid density on real particle.
		std::shared_ptr<SummationDensity<TDataType>> m_summation;
		
		/// Virtual particle density estimated through virtual particle neighborhoods.
		std::shared_ptr<SummationDensity<TDataType>> m_vv_summation;
		
		/// Fluid density on virtual particle.
		std::shared_ptr<SummationDensity<TDataType>> m_vr_summation;

		//std::ofstream outfile_iter;
		//std::ofstream outfile_virtualNumber;
		//std::ofstream outfile_density;

		//DArray<bool> m_solidVirtualPaticleFlag;
		//DArray<Real> m_RealVolumeEst;
		//DArray<Real> m_VirtualVolumeEst;

	};
}