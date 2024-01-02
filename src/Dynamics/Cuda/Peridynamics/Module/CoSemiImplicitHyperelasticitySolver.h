#pragma once
#include "Peridynamics/Bond.h"
#include "Peridynamics/EnergyDensityFunction.h"
#include "Topology/TriangleSet.h"
#include "Collision/Attribute.h"

#include "Peridynamics/Module/LinearElasticitySolver.h"
#include "Peridynamics/Module/ContactRule.h"

namespace dyno
{
	template<typename TDataType> class ContactRule;

	template<typename TDataType>
	class CoSemiImplicitHyperelasticitySolver : public LinearElasticitySolver<TDataType>
	{
		DECLARE_TCLASS(CoSemiImplicitHyperelasticitySolver, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename TBond<TDataType> Bond;

		CoSemiImplicitHyperelasticitySolver();
		~CoSemiImplicitHyperelasticitySolver() override;

		void solveElasticity() override;

		void setObjectVolume(Real volume) { this->m_objectVolume = volume; m_objectVolumeSet = true; }
		void setParticleVolume(Real volume) { this->m_particleVolume = volume; m_particleVolumeSet = true; }
		void setContactMaxIte(int ite) {
			if (this->mContactRule) {
				(this->mContactRule)->setContactMaxIte(ite);
			}
		}
		DEF_VAR_IN(EnergyType, EnergyType, "");

		DEF_VAR_IN(EnergyModels<Real>, EnergyModels, "");

		DEF_VAR(bool, NeighborSearchingAdjacent, true, "");

		DEF_ARRAY_IN(Coord, RestNorm, DeviceType::GPU, "Vertex Rest Normal");

		DEF_ARRAY_IN(Coord, OldPosition, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, MarchPosition, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, Norm, DeviceType::GPU, "Vertex Normal");

		DEF_VAR_IN(Real, Unit, "mesh unit");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangularMesh, "");

		void setXi(Real xi_) { 
			this->xi = xi_; 
			mContactRule->inXi()->setValue(this->xi);
		}

		void setK_bend(Real k) {
			Real k_old = this->k_bend;
			if (k_old > EPSILON)
				this->k_bend *= (k / k_old);
			else
				this->k_bend = k * this->E;
		}

		void setSelfContact(bool s_){
			this->selfContact = s_;
		}

		Real getXi() { return this->xi; }

		void setE(Real E_) {
			Real E_old = this->E;
			this->E = E_;
			this->k_bend *= (this->E / E_old);
		}
		Real getE() {
			return this->E;
		}
		void setS(Real s_) {
			this->s = s_;
			mContactRule->inS()->setValue(this->s);
		}

		Real getS(Real E, Real nv) { return this->s; }

		Real getS0(Real E, Real nv) {
			return 15 * E / (4 * (1 + nv)); //mu
		}

		Real getS1(Real E, Real nv) {
			return 9 * E * nv / (2 * (1 + nv) * (1 - 2 * nv)); //lambda
		}
		void setGrad_res_eps(Real r) {
			this->grad_res_eps = r;
		}
		void setAccelerated(bool acc_) {
			this->acc = acc_;
		}

	public:
		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "Particle Attribute");
		std::shared_ptr<ContactRule<TDataType>> getContactRulePtr() {
			return mContactRule;
		}
		DEF_ARRAY_IN(Coord, DynamicForce, DeviceType::GPU, "");
		DEF_ARRAY_IN(Coord, ContactForce, DeviceType::GPU, "");
	protected:
		void initializeVolume();
		void enforceHyperelasticity();

		void resizeAllFields();

	private:
		void connectContact();
		Real E = 1e3;
		Real k_bend = 0.0 * E;
		Real s = 0.0;
		Real xi = 0.1;
		Real d = 1.0;
		Real grad_res_eps = 1e-3;
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
		DArray<Coord> y_pre;
		DArray<Coord> y_residual;
		DArray<Coord> y_gradC;
		DArray<Real> m_gradientMagnitude;
		DArray<Coord> m_source;
		DArray<Matrix> m_A;

		Reduction<Real>* m_reduce;

		DArray<bool> m_bFixed;
		DArray<Coord> m_fixedPos;
		DArray<Real> m_volume;
		DArray<Coord> mPosBuf_March;
		Real m_objectVolume;
		bool m_objectVolumeSet = false;
		Real m_particleVolume;
		bool m_particleVolumeSet = false;
		std::shared_ptr<ContactRule<TDataType>> mContactRule;
		bool m_alphaCompute = true; // inversion control
		bool selfContact = true;
		bool acc = false;
	};
}