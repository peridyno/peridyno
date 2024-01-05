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
		CodimensionalPD( Real xi, Real E , Real kb, Real timeSetp=1e-3, std::string name = "default");
		virtual ~CodimensionalPD();


		void setSelfContact(bool s) { (this->mHyperElasticity)->setSelfContact(s); }
		bool translate(Coord t) override;
		bool scale(Real s) override;
		bool scale(Coord s);
		void loadSurface(std::string filename);

		void setEnergyModel(StVKModel<Real> model);
		void setEnergyModel(LinearModel<Real> model);
		void setEnergyModel(NeoHookeanModel<Real> model);
		void setEnergyModel(XuModel<Real> model);
		void setEnergyModel(FiberModel<Real> model);
		void setMaxIteNumber(uint i) {
			if (this->mHyperElasticity) {
				(this->mHyperElasticity)->varIterationNumber()->setValue(i);
			}
		}
		void setGrad_ite_eps(Real r) {
			if (this->mHyperElasticity) {
				(this->mHyperElasticity)->setGrad_res_eps(r);
			}
		}
		void setContactMaxIte(int ite) {
			if (this->mHyperElasticity) {
				(this->mHyperElasticity)->setContactMaxIte(ite);
			}
		}
		void setAccelerated(bool acc) {
			if (this->mHyperElasticity) {
				this->mHyperElasticity->setAccelerated(acc);
			}
		}

	public:
		DEF_VAR(Real, Horizon, 0.01, "Horizon");

		DEF_VAR(EnergyType, EnergyType, Xuetal, "");

		DEF_VAR(EnergyModels<Real>, EnergyModel, EnergyModels<Real>(), "");

		DEF_ARRAY_STATE(Coord, RestPosition, DeviceType::GPU, "");
		
		DEF_ARRAY_STATE(Coord, OldPosition, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Coord, VerticesRef, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Coord, RestNorm, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Coord, Norm, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Matrix, VertexRotation, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Attribute, Attribute, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Real, Volume, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Coord, DynamicForce, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Coord, ContactForce, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Coord, MarchPosition, DeviceType::GPU, "");

		DEF_ARRAYLIST_STATE(Bond, RestShape, DeviceType::GPU, "");
		
		DEF_VAR_STATE(Real, MaxLength, DeviceType::GPU, "");

		DEF_VAR_STATE(Real, MinLength, DeviceType::GPU,  "");

		DEF_VAR(bool, NeighborSearchingAdjacent, true, "");

	private:
		std::shared_ptr< CoSemiImplicitHyperelasticitySolver<TDataType> > mHyperElasticity;

	protected:
		void resetStates() override;
		virtual void updateRestShape();
		virtual void updateVolume();
		virtual void preUpdateStates();
    	void updateTopology() override;
		
	};
}