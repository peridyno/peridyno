/**
 * Copyright 2021 Shusen Liu
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
#include "ParticleSystem/Module/Kernel.h"
#include "ParticleSystem/Module/SummationDensity.h"

 /**
  * @brief This is an implementation of Adapted SIMPLE Algorithm for SPH Fluids based on peridyno.
  *
  * For details, refer to "Shusen Liu, Xiaowei He, Wencheng Wang, Enhua Wu.
  *	 Adapted SIMPLE Algorithm for Incompressible SPH Fluids With a Broad Range Viscosity[J],
  *	 IEEE TRANSACTIONS ON VISUALIZATION AND COMPUTER GRAPHICS,2022:28(9).
  *
  */

namespace dyno {

	class Attribute;
	template<typename TDataType> class SummationDensity;

	template<typename TDataType>
	class SimpleVelocityConstraint : public ConstraintModule
	{

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		SimpleVelocityConstraint();
		~SimpleVelocityConstraint() override;

		void constrain() override;
		bool initialize();

		bool resizeVector();
		void initialAttributes();
		bool visValueSet()
		{
			visValueSet(this->varViscosity()->getValue());
			return true;
		}

		bool visValueSet(Real vis)
		{
			std::vector<Real> vis_temp;
			for (int i = 0; i < m_viscosity.size(); i++)
			{
				vis_temp.push_back(vis);
			}
			m_viscosity.assign(vis_temp);
			return true;
		};

		bool visVectorSet(CArray<Real> vis)
		{
			m_viscosity.assign(vis);
			return true;
		};

		bool SIMPLE_IterNumSet(int i) {
			SIMPLE_IterNum = i;
			return true;
		};


		bool SetCross(Real visT, Real mag, Real k1, Real n1)
		{
			CrossVisCeil = visT;
			CrossVisMag = mag;
			CrossVisFloor = visT / mag;
			Cross_K = k1;
			Cross_N = n1;
			std::cout << "Cross modle is setted!  Viscosity:" << CrossVisFloor
				<< "~" << CrossVisCeil << ", K:"
				<< Cross_K << ", N:" << Cross_N
				<< std::endl;

			IsCrossReady = true;
			return IsCrossReady;
		};



	public:

		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Input real particle position");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Input particle velocity");

		DEF_ARRAY_IN(Coord, Normal, DeviceType::GPU, "Input particle velocity");

		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "Input particle velocity");

		DEF_VAR_IN(Real, SmoothingLength, "");

		DEF_VAR(Real, RestDensity, Real(1000), "Reference density");

		DEF_VAR_IN(Real, SamplingDistance, "");

		DEF_VAR_IN(Real, TimeStep, "");

		DEF_VAR(Real, Viscosity, Real(5000.0), "Initial Viscosity Value");

		DEF_VAR(bool, SimpleIterationEnable, true, "");

	private:

		DArray<Real> m_viscosity;
		Real Cross_N;
		Real CrossVisCeil;
		Real CrossVisMag;
		Real CrossVisFloor;
		Real Cross_K;
		bool m_bConfigured = false;
		bool IsCrossReady = false;

		Real m_maxAlpha;
		Real m_maxA;
		Real m_airPressure = 0.0f;

		Real m_particleMass = 1.0f;
		Real m_tangential = 0.1f;
		Real m_separation = 0.1f;

		//Refer to "A Nonlocal Variational Particle Framework for Incompressible Free Surface Flows" for their exact meanings
		DArray<Real> m_alpha;
		DArray<Real> m_Aii;
		DArray<Real> m_AiiFluid;
		DArray<Real> m_AiiTotal;

		DArray<Coord> P_dv;
		DArray<Coord> velOld;
		DArray<Coord> velBuf;


		DArray<Real> m_deltaPressure;
		DArray<Real> m_pressBuf;
		DArray<Real> m_crossViscosity;

		DArray<Real> m_pressure;
		DArray<Real> m_divergence;
		//Indicate whether a particle is near the free surface boundary.
		DArray<bool> m_bSurface;

		//Used to solve the linear system of PPE with a conjugate gradient method.
		DArray<Real> m_y;
		DArray<Real> m_r;
		DArray<Real> m_p;

		//Used to solve the linear system of velocity constraint with a conjugate gradient method.
		DArray<Real> v_y;
		DArray<Real> v_r;
		DArray<Real> v_p;
		DArray<Coord> v_pv;
		DArray<Real> m_VelocityReal;

		Reduction<Real>* m_reduce;
		Arithmetic<Real>* m_arithmetic;
		Arithmetic<Real>* m_arithmetic_v;

		std::shared_ptr<SummationDensity<TDataType>>  m_densitySum;

		bool init_flag = false;
		int SIMPLE_IterNum;

	};


}