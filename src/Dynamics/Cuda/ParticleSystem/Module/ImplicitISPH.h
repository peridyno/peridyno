#pragma once
#include "ParticleApproximation.h"
#include "Algorithm/Arithmetic.h"
namespace dyno {

	template<typename TDataType> class SummationDensity;

	/**
	 * @brief	This is an implementation of the Implicit Incompressible SPH (IISPH) solver based on PeriDyno.
	 *			For details, refer to "Implicit Incompressible SPH" by Ihmsen et al., IEEE TVCG, 2015.
	 *			The code was written by Shusen Liu, ISCAS, Sep, 2024.
	 * @note	Too large a neighborhood radius may lead to instability, due to the use of second-order particle neighborhoods in the Laplacian. 
	*/
	template<typename TDataType>
	class ImplicitISPH : public ParticleApproximation<TDataType>
	{
		DECLARE_TCLASS(ImplicitISPH, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ImplicitISPH();
		~ImplicitISPH() override;

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
		DEF_VAR(int, IterationNumber, 30, "Iteration number of the PBD solver");

		DEF_VAR(Real, RestDensity, 1000, "Reference density");

		DEF_VAR(Real, Kappa, Real(1), "");

		DEF_VAR(Real, RelaxedOmega, Real(0.5f), "");

	protected:
		void compute() override;

	public:
		Real takeOneIteration();

		void PreIterationCompute();

		void updateVelocity();

	private:

		DArray<Real> mSourceTerm;

		DArray<Coord> mDii;  

		DArray<Real> mAii;

		DArray<Real> mAnPn;

		DArray<Coord> mSumDijPj;

		DArray<Real> mPressrue;

		DArray<Real> mOldPressrue;

		DArray<Real> m_Residual;

		DArray<Real> mPredictDensity;

		DArray<Real> mDensityAdv;

		
		Arithmetic<Real>* m_arithmetic;


	private:
		std::shared_ptr<SummationDensity<TDataType>> mSummation;
	};

	IMPLEMENT_TCLASS(ImplicitISPH, TDataType)
}