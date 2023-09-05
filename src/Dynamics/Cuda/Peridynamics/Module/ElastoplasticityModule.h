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
		typedef typename TBond<TDataType> Bond;

		ElastoplasticityModule();
		~ElastoplasticityModule() override {};
		
		void constrain() override;

		void solveElasticity() override;

		virtual void applyPlasticity();

		void applyYielding();

		void rotateRestShape();
		void reconstructRestShape();

		void setCohesion(Real c);
		void setFrictionAngle(Real phi);

		void enableFullyReconstruction();
		void disableFullyReconstruction();

		void enableIncompressibility();
		void disableIncompressibility();

		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "Neighboring particles' ids");

	protected:
		inline Real computeA()
		{
			Real phi = m_phi.getData();
			return (Real)6.0*m_c.getData()*cos(phi) / (3.0f + sin(phi)) / sqrt(3.0f);
		}


		inline Real computeB()
		{
			Real phi = m_phi.getData();
			return (Real)2.0f*sin(phi) / (3.0f + sin(phi)) / sqrt(3.0f);
		}

	private:

		FVar<Real> m_c;
		FVar<Real> m_phi;

		FVar<bool> m_reconstuct_all_neighborhood;
		FVar<bool> m_incompressible;

		DArray<bool> m_bYield;
		DArray<Matrix> m_invF;
		DArray<Real> m_yiled_I1;
		DArray<Real> m_yield_J2;
		DArray<Real> m_I1;

		std::shared_ptr<IterativeDensitySolver<TDataType>> mDensityPBD;
	};
}