#pragma once
/**
* @file HyperelasticityModule.h
* @author Xiaowei He (xiaowei@iscas.ac.cn)
* @brief This is an implementation of hyperelasticity based on a set of basis functions.
* 		  For more details, please refer to [Xu et al. 2018] "Reformulating Hyperelastic Materials with Peridynamic Modeling"
* @version 0.1
* @date 2019-06-18
*
* @copyright Copyright (c) 2019
*
*/
#pragma once
#include "ElasticityModule.h"

namespace dyno {

	template<typename TDataType>
	class HyperelasticityModule_NewtonMethod : public ElasticityModule<TDataType>
	{
	public:
		HyperelasticityModule_NewtonMethod();
		~HyperelasticityModule_NewtonMethod() override {};

		enum EnergyType
		{
			Linear,
			StVK
		};

		/**
		* @brief Set the energy function
		*
		*/
		void setEnergyFunction(EnergyType type) { m_energyType = type; }

		void solveElasticity() override;
		void solveElasticity_NewtonMethod();
		void solveElasticity_NewtonMethod_StVK();

		void setInfluenceWeightScale(Real w_scale) { this->weightScale = w_scale; };

	protected:
		bool initializeImpl() override;


		//void previous_enforceElasticity();

	private:
		bool ImplicitMethod = true;

		EnergyType m_energyType;

		GArray<Real> m_totalWeight;
		Real weightScale = 110;

		GArray<Coord> m_Sum_delta_x;
		GArray<Coord> m_source_items;

		GArray<Matrix> m_invK;
		Matrix common_K;

		GArray<Matrix> m_F;
		GArray<Matrix> m_firstPiolaKirchhoffStress;

		bool debug_pos_isNaN = false;
		bool debug_v_isNaN = false;
		bool debug_invL_isNaN = false;
		bool debug_F_isNaN = false;
		bool debug_invF_isNaN = false;
		bool debug_Piola_isNaN = false;
	};

}