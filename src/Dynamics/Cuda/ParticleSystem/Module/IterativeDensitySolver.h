#pragma once
#include "ParticleApproximation.h"
#include "Algorithm/Arithmetic.h"
namespace dyno {

	template<typename TDataType> class SummationDensity;

	/**
	 * @brief This is an implementation of the iterative density solver integrated into the Position Based Dynamics framework.
	 *
	 * For details, refer to "Position Based Fluids" by Macklin and M¡§uller, ACM TOG, 2013
	 *
	*/
	template<typename TDataType>
	class IterativeDensitySolver : public ParticleApproximation<TDataType>
	{
		DECLARE_TCLASS(IterativeDensitySolver, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		IterativeDensitySolver();
		~IterativeDensitySolver() override;

	public:
		DEF_VAR_IN(Real, TimeStep, "Time Step");

		/**
		 * @brief Particle positions
		 */
		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Input particle position");

		/**
		 * @brief Particle velocities
		 */
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Input particle velocity");

		/**
		 * @brief Neighboring particles' ids
		 *
		 */
		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "Neighboring particles' ids");

		/**
		 * @brief Final particle densities
		 */
		DEF_ARRAY_OUT(Real, Density, DeviceType::GPU, "Final particle density");

	public:
		DEF_VAR(int, IterationNumber, 5, "Iteration number of the PBD solver");

		DEF_VAR(Real, RestDensity, 1000, "Reference density");

		DEF_VAR(Real, Kappa, Real(1), "");

	protected:
		void compute() override;

	public:
		void takeOneIteration();

		void updateVelocity();

	private:
		DArray<Real> mLamda;
		DArray<Coord> mDeltaPos;
		DArray<Coord> mPositionOld;


		Arithmetic<Real>* m_arithmetic;


	private:
		std::shared_ptr<SummationDensity<TDataType>> mSummation;
	};

	IMPLEMENT_TCLASS(IterativeDensitySolver, TDataType)
}