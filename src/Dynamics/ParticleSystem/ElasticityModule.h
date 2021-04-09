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
#include "Framework/ModuleConstraint.h"
#include "NeighborData.h"

namespace dyno {

	template<typename TDataType>
	class ElasticityModule : public ConstraintModule
	{
		DECLARE_CLASS_1(ElasticityModule, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef TPair<TDataType> NPair;

		ElasticityModule();
		~ElasticityModule() override;
		
		bool constrain() override;

		virtual void solveElasticity();

		void setMu(Real mu) { m_mu.setValue(mu); }
		void setLambda(Real lambda) { m_lambda.setValue(lambda); }

//		void setHorizon(Real len) { m_horizon.setValue(len); }
		void setIterationNumber(int num) { m_iterNum.setValue(num); }
		int getIterationNumber() { return m_iterNum.getData(); }

		void resetRestShape();

	protected:
		bool initializeImpl() override;

		void begin() override;

		/**
		 * @brief Correct the particle position with one iteration
		 * Be sure computeInverseK() is called as long as the rest shape is changed
		 */
		virtual void enforceElasticity();
		virtual void computeMaterialStiffness();

		void updateVelocity();
		void computeInverseK();

	public:
		/**
			* @brief Horizon
			* A positive number represents the radius of neighborhood for each point
			*/
		DEF_EMPTY_IN_VAR(Horizon, Real, "");
		//VarField<Real> m_horizon;

		/**
		 * @brief Particle position
		 */
		DEF_EMPTY_IN_ARRAY(Position, Coord, DeviceType::GPU, "Particle position");
		//DeviceArrayField<Coord> m_position;

		/**
			* @brief Particle velocity
			*/
		DEF_EMPTY_IN_ARRAY(Velocity, Coord, DeviceType::GPU, "Particle velocity");
		//DeviceArrayField<Coord> m_velocity;

		/**
		 * @brief Neighboring particles
		 * 
		 */
		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "Neighboring particles' ids");
		//NeighborField<int> m_neighborhood;


		DEF_ARRAYLIST_IN(NPair, RestShape, DeviceType::GPU, "Reference shape");
		//NeighborField<int> m_neighborhood;

//		NeighborField<NPair> m_restShape;

	protected:
		/**
		* @brief Lame parameters
		* m_lambda controls the isotropic part while mu controls the deviatoric part.
		*/
		VarField<Real> m_mu;
		VarField<Real> m_lambda;

		DArray<Real> m_bulkCoefs;
		DArray<Coord> m_position_old;

		DArray<Real> m_weights;
		DArray<Coord> m_displacement;
		DArray<Matrix> m_invK;
	private:
		VarField<int> m_iterNum;

		DArray<Real> m_stiffness;
		DArray<Matrix> m_F;
	};
}