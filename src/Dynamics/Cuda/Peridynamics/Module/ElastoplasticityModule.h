/**
 * @file ElastoplasticityModule.h
 * @author Xiaowei He (xiaowei@iscas.ac.cn)
 * @brief This is an implementation of elastoplasticity based on projective peridynamics.
 * 		  For more details, please refer to [He et al. 2017] "Projective Peridynamics for Modeling Versatile Elastoplastic Materials"
 * @version 0.1
 * @date 2019-06-18
 *
 * @copyright Copyright (c) 2019
 *
 */
#pragma once
#include "LinearElasticitySolver.h"

#include "ParticleSystem/Module/IterativeDensitySolver.h"

namespace dyno {

	template<typename TDataType>
	class ElastoplasticityModule : public LinearElasticitySolver<TDataType>
	{
		DECLARE_TCLASS(ElastoplasticityModule, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename ::dyno::TBond<TDataType> Bond;

		ElastoplasticityModule();
		~ElastoplasticityModule() override;

	public:
		DEF_VAR(Real, Cohesion, 0.0, "Cohesion between particles");

		DEF_VAR(Real, FrictionAngle, Real(1.0f / 3.0f), "Cohesion between particles");

		DEF_VAR(bool, Incompressible, true, "Incompressible or not");

		DEF_VAR(bool, RenewNeighborhood, false, "Whether to renew particle neighbors every time step");

		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "Neighboring particles' ids");
	
	protected:
		void constrain() override;

		void solveElasticity() override;

		virtual void applyPlasticity();

		void applyYielding();

		void rotateRestShape();
		void reconstructRestShape();

	protected:
		inline Real computeA()
		{
			Real phi = this->varFrictionAngle()->getValue() * M_PI;
			return (Real)6.0 * this->varCohesion()->getValue() * cos(phi) / (3.0f + sin(phi)) / std::sqrt(3.0f);
		}


		inline Real computeB()
		{
			Real phi = this->varFrictionAngle()->getValue() * M_PI;
			return (Real)2.0f * sin(phi) / (3.0f + sin(phi)) / std::sqrt(3.0f);
		}

	private:
		DArray<bool> m_bYield;
		DArray<Matrix> m_invF;
		DArray<Real> m_yiled_I1;
		DArray<Real> m_yield_J2;
		DArray<Real> m_I1;

		std::shared_ptr<IterativeDensitySolver<TDataType>> mDensityPBD;
	};
}