#pragma once
#include "../Bond.h"
#include "../EnergyDensityFunction.h"

#include "Collision/Attribute.h"

#include "LinearElasticitySolver.h"

namespace dyno 
{
	template<typename TDataType>
	class SemiImplicitHyperelasticitySolver : public LinearElasticitySolver<TDataType>
	{
		DECLARE_TCLASS(SemiImplicitHyperelasticitySolver, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename TBond<TDataType> Bond;

		SemiImplicitHyperelasticitySolver();
		~SemiImplicitHyperelasticitySolver() override;

		void solveElasticity() override;

		DEF_VAR(Real, StrainLimiting, 0.1, "");

	public:
		DEF_VAR_IN(EnergyType, EnergyType, "");
		DEF_VAR_IN(EnergyModels<Real>, EnergyModels, "");

		DEF_VAR(bool, IsAlphaComputed, true, "");

		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "Particle Attribute");
		DEF_ARRAY_IN(Real, Volume, DeviceType::GPU, "Particle volume");

		DEF_ARRAYLIST_IN(Real, VolumePair, DeviceType::GPU, "");

	protected:
		void enforceHyperelasticity();

		void resizeAllFields();
	
	private:
		DArray<Real> m_fraction;

		DArray<Real> m_energy;
		DArray<Real> m_alpha;
		DArray<Coord> m_gradient;
		DArray<Coord> mEnergyGradient;

		DArray<Coord> m_eigenValues;

		DArray<Matrix> m_F;
		DArray<Matrix> m_invF;
		DArray<bool> m_validOfK;
		DArray<bool> m_validOfF;
		DArray<Matrix> m_invK;
		DArray<Matrix> m_matU;
		DArray<Matrix> m_matV;
		DArray<Matrix> m_matR;

		DArray<Coord> y_current;
		DArray<Coord> y_next;
		DArray<Coord> y_pre;
		DArray<Coord> y_residual;
		DArray<Coord> y_gradC;

		DArray<Coord> m_source;
		DArray<Matrix> m_A;

		Reduction<Real>* m_reduce;

		bool m_alphaCompute = true; // inversion control
	};
}