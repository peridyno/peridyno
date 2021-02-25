/**
 * @file HyperelastoplasticityModule.h
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
#include "ParticleSystem/ElasticityModule.h"
#include "ParticleSystem/HyperelasticityModule_test.h"
#include "ParticleSystem/DensityPBD.h"

namespace dyno {

	template<typename TDataType>
	class HyperelastoplasticityModule : public HyperelasticityModule_test<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef TPair<TDataType> NPair;

		HyperelastoplasticityModule();
		~HyperelastoplasticityModule() override {};
		
		bool constrain() override;

//		void solveElasticity() override;

		virtual void applyPlasticity();

		void applyYielding();


		void setCohesion(Real c);
		void setFrictionAngle(Real phi);

		DEF_EMPTY_IN_ARRAY(PrincipleYielding, Coord, DeviceType::GPU, "Storing the plastic yielding along three principle stretches.");

	protected:
		bool initializeImpl() override;
		void begin() override;

		inline Real computeA()
		{
			Real phi = m_phi.getValue();
			return (Real)6.0*m_c.getValue()*cos(phi) / (3.0f + sin(phi)) / sqrt(3.0f);
		}


		inline Real computeB()
		{
			Real phi = m_phi.getValue();
			return (Real)2.0f*sin(phi) / (3.0f + sin(phi)) / sqrt(3.0f);
		}

	private:

		VarField<Real> m_c;
		VarField<Real> m_phi;

		DeviceArray<Coord> m_yielding;

		DeviceArray<bool> m_bYield;
		DeviceArray<Matrix> m_invF;
		DeviceArray<Real> m_yiled_I1;
		DeviceArray<Real> m_yield_J2;
		DeviceArray<Real> m_I1;
	};

#ifdef PRECISION_FLOAT
	template class HyperelastoplasticityModule<DataType3f>;
#else
	template class HyperelastoplasticityModule<DataType3d>;
#endif
}