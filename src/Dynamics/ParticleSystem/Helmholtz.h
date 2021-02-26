#pragma once
#include "Array/Array.h"
#include "Framework/ModuleConstraint.h"
#include "Topology/FieldNeighbor.h"

namespace dyno {

	template<typename TDataType> class SummationDensity;

	/*!
	*	\class	Helmholtz
	*	\brief	This class implements a position-based solver for incompressibility.
	*/
	template<typename TDataType>
	class Helmholtz : public ConstraintModule
	{
		DECLARE_CLASS_1(Helmholtz, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Helmholtz();
		~Helmholtz() override;

		bool constrain() override;

		void setPositionID(FieldID id) { m_posID = id; }
		void setVelocityID(FieldID id) { m_velID = id; }
		void setNeighborhoodID(FieldID id) {m_neighborhoodID = id; }

		void setIterationNumber(int n) { m_maxIteration = n; }
		void setSmoothingLength(Real len) { m_smoothingLength = len; }

		void computeC(GArray<Real>& c, GArray<Coord>& pos, NeighborList<int>& neighbors);
		void computeGC();
		void computeLC(GArray<Real>& lc, GArray<Coord>& pos, NeighborList<int>& neighbors);

		void setReferenceDensity(Real rho) {
			m_referenceRho = rho;
		}

	protected:
		bool initializeImpl() override;

	protected:
		FieldID m_posID;
		FieldID m_velID;
		FieldID m_neighborhoodID;

	private:
		bool m_bSetup;

		int m_maxIteration;
		Real m_smoothingLength;
		Real m_referenceRho;

		Real m_scale;
		Real m_lambda;
		Real m_kappa;

		GArray<Real> m_c;
		GArray<Real> m_lc;
		GArray<Real> m_energy;
		GArray<Coord> m_bufPos;
		GArray<Coord> m_originPos;
	};

#ifdef PRECISION_FLOAT
	template class Helmholtz<DataType3f>;
#else
 	template class Helmholtz<DataType3d>;
#endif
}