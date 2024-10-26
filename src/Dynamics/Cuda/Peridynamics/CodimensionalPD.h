#pragma once
#include "Module/CoSemiImplicitHyperelasticitySolver.h"
#include "Collision/Attribute.h"
#include "ParticleSystem/ParticleSystem.h"
#include "Peridynamics/TriangularSystem.h"
#include "Peridynamics/Bond.h"
#include "Peridynamics/EnergyDensityFunction.h"


namespace dyno
{
	template<typename> class PointSetToPointSet;

	template<typename TDataType> class CoSemiImplicitHyperelasticitySolver;

	template<typename TDataType>
	class CodimensionalPD : public TriangularSystem<TDataType>
	{
		DECLARE_TCLASS(CodimensionalPD, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename TBond<TDataType> Bond;

		CodimensionalPD();
		~CodimensionalPD() override;

		bool translate(Coord t) override;
		bool scale(Real s) override;
		bool scale(Coord s);
		void loadSurface(std::string filename);

		void setEnergyModel(StVKModel<Real> model);
		void setEnergyModel(LinearModel<Real> model);
		void setEnergyModel(NeoHookeanModel<Real> model);
		void setEnergyModel(XuModel<Real> model);
		void setEnergyModel(FiberModel<Real> model);

	public:
		DEF_VAR(Real, Horizon, 0.01, "Horizon");

		DEF_VAR(EnergyType, EnergyType, Xuetal, "");

		DEF_VAR(EnergyModels<Real>, EnergyModel, EnergyModels<Real>(), "");

		DEF_ARRAY_STATE(Coord, RestPosition, DeviceType::GPU, "");
		
		DEF_ARRAY_STATE(Coord, OldPosition, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Coord, RestNorm, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Coord, Norm, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Attribute, Attribute, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Real, Volume, DeviceType::GPU, "");

		DEF_ARRAYLIST_STATE(Bond, RestShape, DeviceType::GPU, "");
		
		DEF_VAR_STATE(Real, MaxLength, DeviceType::GPU, "");

		DEF_VAR_STATE(Real, MinLength, DeviceType::GPU, "");

	protected:
		void resetStates() override;

		void preUpdateStates() override;
    	void updateTopology() override;
		
		virtual void updateRestShape();
		virtual void updateVolume();
	};
}