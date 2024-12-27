
#pragma once
#include "ParticleApproximation.h"
#include "Algorithm/Arithmetic.h"
namespace dyno {

	template<typename TDataType> class SummationDensity;

	/*
	 *
	 * @brief	This is the GPU implementation of the DFSPH (Divergence Free SPH) method based on Peridyno.
	 *			For details, refer to "Divergence-Free SPH for Incompressible and Viscous Fluids" by Bender and Koschier, IEEE TVCG, 2016.
	 *			The code was written by Shusen Liu (liushusen@iscas.ac.cn), ISCAS, December (Christmas), 2024.
	 *			In this method, the incompressiblity is achieved by the combining of the Divergence solver and the Density solver.
	 * 
	 * @note	It is suggested to use the Spiky Kernel for computation, and the smoothing-length(support radius) cannot be less than 2.5 times initial particle spacing.
	 *			If the simulation fails, please turn down the time-step size. 
	 *			
	 *			
	*/

	template<typename TDataType>
	class DivergenceFreeSphSolver : public ParticleApproximation<TDataType>
	{
		DECLARE_TCLASS(DivergenceFreeSphSolver, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DivergenceFreeSphSolver();
		~DivergenceFreeSphSolver() override;

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


		/**
		 * @brief Disable the divergence solver
		 */
		DEF_VAR(bool, DivergenceSolverDisabled, false, "Disable the Divergence solver in the DFSPH method");

		/**
		* @brief Disable the density solver
		*/
		DEF_VAR(bool, DensitySolverDisabled, false, "Disable the Density solver in the DFSPH method");

		/**
		* @brief Rest density
		*/
		DEF_VAR(Real, RestDensity, 1000, "Reference density");

		/**
		* @brief  Error Threshold for the Divergence solver
		*/
		DEF_VAR(Real, DivergenceErrorThreshold, 0.1, "Error Thershold for the Divergence solver");

		/**
		* @brief  Error Threshold for the Density solver
		*/
		DEF_VAR(Real, DensityErrorThreshold, 0.001, "Error Thershold for the Divergence solver");


		/**
		* @brief  Maximum number of iteration of each solver
		*/
		DEF_VAR(Real, MaxIterationNumber, 50, "Maximum number of iteration of each solver");


	public:
		void compute() override;

		void computeAlpha();

		Real takeOneDensityIteration();

		Real takeOneDivergenIteration();

	private:
	
		/*
		* @brief Particle Stiffness parameter for the density solver ("K_i" in the paper)
		*/
		DArray<Real> mKappa_r;

		/*
		* @brief Particle Stiffness parameter for the divervence solver ("K^v_i" in the paper)
		*/
		DArray<Real> mKappa_v;

		/*
		* @brief Alpha factor ("Alpha_i" in the paper)
		*/
		DArray<Real> mAlpha;

		/*
		* @brief The density estimated by the divergence ("\rho^*_i" in the paper)
		*/
		DArray<Real> mPredictDensity;

		/*
		* @brief Velocity divergences or   ("\frac{D \rho_i}{D t}" in the paper)
		*/
		DArray<Real> mDivergence;


	private:
		std::shared_ptr<SummationDensity<TDataType>> mSummation;
	};

	IMPLEMENT_TCLASS(DivergenceFreeSphSolver, TDataType)
}