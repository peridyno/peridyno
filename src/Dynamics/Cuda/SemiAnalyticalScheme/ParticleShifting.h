#pragma once
#include "Module/TopologyModule.h"

#include "ParticleSystem/Attribute.h"
#include "ParticleSystem/Module/ParticleApproximation.h"

namespace dyno {

	template<typename TDataType> class SemiAnalyticalSummationDensity;
	typedef typename TopologyModule::Triangle Triangle;
	template<typename TDataType>

	class ParticleShifting : public virtual ParticleApproximation<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;



		ParticleShifting();
		~ParticleShifting();

		void compute()override;
		bool bulkEnergyDensitySet(Real i) { this->varBulk()->setValue(i);	return true; };
		bool surfaceTensionSet(Real i) { this->varSurfaceTension()->setValue(i); return true; };
		bool momentumPotentialSet(Real i) { this->varInertia()->setValue(i); return true; };

	public:
		DEF_VAR_IN(Real, TimeStep, "Time step size");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Particle position");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Particle velocity");

		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "Particle attribute");

		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "");

		DEF_ARRAYLIST_IN(int, NeighborTriIds, DeviceType::GPU, "triangle neighbors");

		DEF_ARRAY_IN(Triangle, TriangleInd, DeviceType::GPU, "triangle_index");
		DEF_ARRAY_IN(Coord, TriangleVer, DeviceType::GPU, "triangle_vertex");



		//DEF_VAR(Real, SmoothingLength, Real(0.0125), "smoothing length");

		DEF_VAR(Real, Inertia, Real(0.1), "inertia");

		DEF_VAR(Real, Bulk, Real(0.5), "bulk");

		DEF_VAR(Real, SurfaceTension, Real(0.055), "surface tension");

		DEF_VAR(Real, AdhesionIntensity, Real(30.0), "adhesion");

		DEF_VAR(Real, RestDensity, Real(1000.0), "Rest Density");

	private:
		int mIterationNumber;
		Real mEnergyDepth;//adhesion energy coefficient
		DArray<Real> mLambda;
		DArray<Real> mTotalW;
		DArray<Coord> mBoundaryDir;
		DArray<Real> mBoundaryDis;
		DArray<Coord> mDeltaPos;
		DArray<Coord> mPosBuf;
		DArray<Coord> mAdhesionEP;
		std::shared_ptr<SemiAnalyticalSummationDensity<TDataType>> mCalculateDensity;
	};
}