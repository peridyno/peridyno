#pragma once
#include "Topology.h"
#include "Algorithm/Functional.h"
#include "Algorithm/Arithmetic.h"
#include "Topology/TriangleSets.h"
#include "Topology/PointSet.h"
#include "ParticleSystem/Module/ParticleApproximation.h"



namespace dyno
{
	template<typename TDataType> class SemiAnalyticalSummationDensity;

	template<typename TDataType>
	class SemiAnalyticalDensitySolver : public virtual ParticleApproximation<TDataType>
	{
		DECLARE_TCLASS(SemiAnalyticalDensitySolver, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename ::dyno::Quat<Real> TQuat;
		typedef typename Topology::Triangle Triangle;

		SemiAnalyticalDensitySolver();
		~SemiAnalyticalDensitySolver() override;


	public:

		DEF_VAR(Real, RestDensity, Real(1000.0), "Rest Density");
		DEF_VAR(Real, ErrorRate, Real(0.001), "ErrorRate");
		DEF_VAR(Real, Mu, Real(1), "Mu");
		DEF_VAR(Real, BoundaryFriction, Real(0.0), "Boundary tangential friction (0..1)");
		DEF_VAR(Real, KappaLower, Real(100.0), "KappaLower");
		DEF_VAR(Real, D_hat, Real(0.005), "D_hat");
		DEF_VAR(uint, IterationNumber, 5, "");
		DEF_VAR(uint, PolynomialNumber, 3, "");
		DEF_VAR(bool, WarmStart, false, "KappaLower warm start flag");

	public:
		DEF_VAR_IN(Real, TimeStep, "Time step size");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Particle position");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Particle velocity");

		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "");

		DEF_ARRAYLIST_IN(int, NeighborTriIdsMerge, DeviceType::GPU, "triangle neighbors");

		DEF_INSTANCE_IN(TriangleSets<TDataType>, TriangleSetMerge, "");

		DEF_ARRAY_IN(Coord, PreTriangleVerMerge, DeviceType::GPU, "Pretriangle_vertex");
		DEF_ARRAY_OUT(Real, Density, DeviceType::GPU, "Density");
		DEF_ARRAY_OUT(Real, Kappas, DeviceType::GPU, "Kappas");

	protected:
		void compute() override;

	public:
		void updatePosition();

	private:
		DArray<Real> mEnergy;
		DArray<Real> mSignDis;
		DArray<uint> mPolyN;
		DArray<Real> mAlpha;
		DArray<Real> mA;
		DArray<Coord> mKappa;
		DArray<Coord> mPosBuf;
		DArray<Coord> mPosOld;
		DArray<Coord> mPosStart;
		DArray<Coord> mVelStart;
		Reduction<uint> mReduce;
		Arithmetic<Real>* m_arithmetic_v = nullptr;
		std::shared_ptr<SemiAnalyticalSummationDensity<TDataType>> mCalculateDensity;
	};
}
