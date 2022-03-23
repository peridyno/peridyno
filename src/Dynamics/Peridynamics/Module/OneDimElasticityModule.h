/**
 * @file ElasticityModule.h
 * @author Xiaowei He (xiaowei@iscas.ac.cn)
 * @brief This is an implementation of elasticity based on projective peridynamics.
 * 		  For more details, please refer to [He et al. 2017] "Projective Peridynamics for Modeling Versatile Elastoplastic Materials"
 * @version 0.1
 * @date 2019-06-18
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once
#include "Module/ConstraintModule.h"

namespace dyno {

	template<typename TDataType>
	class OneDimElasticityModule : public ConstraintModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		OneDimElasticityModule();
		~OneDimElasticityModule() override;
		
		void constrain() override;

		void solveElasticity();

		void setIterationNumber(int num) { m_iterNum.setValue(num); }
		int getIterationNumber() { return m_iterNum.getData(); }

		void setMaterialStiffness(Real stiff) { m_lambda.setValue(stiff); }

	protected:
		bool initializeImpl() override;

		void updateVelocity();

	public:
		/**
		 * @brief Horizon
		 * A positive number represents the radius of neighborhood for each point
		 */
		FVar<Real> m_distance;

		/**
		 * @brief Particle position
		 */
		DeviceArrayField<Coord> m_position;
		/**
		 * @brief Particle velocity
		 */
		DeviceArrayField<Coord> m_velocity;

		DeviceArrayField<Real> m_mass;

	protected:
		/**
		* @brief Lame parameters
		* m_lambda controls the isotropic part while mu controls the deviatoric part.
		*/
		FVar<Real> m_lambda;

		DArray<Coord> m_position_old;
		DArray<Coord> m_position_buf;

	private:
		FVar<int> m_iterNum;
	};
}