/*`
* @ Author: Shusen Liu, 2022
* @ FLIP Fulid solver (Explicit fluid solver)
*/


#pragma once
#include "Module/ConstraintModule.h"
#include "ParticleSystem/Module/Kernel.h"

namespace dyno {


	template<typename TDataType> class SummationDensity;

	template<typename TDataType>
	class FlipFluidExplicitSolver : public ConstraintModule
	{
		DECLARE_TCLASS(FlipFluidExplicitSolver, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;


		FlipFluidExplicitSolver();
		~FlipFluidExplicitSolver() override;

		void constrain() override;


		DEF_VAR_IN(Real, TimeStep, "Time Step");

		/**
		 * @brief Particle positions
		 */
		DEF_ARRAY_IN(Coord, ParticlePosition, DeviceType::GPU, "Input particle position");

		/**
		 * @brief Particle velocities
		 */
		DEF_ARRAY_IN(Coord, ParticleVelocity, DeviceType::GPU, "Input particle velocity");

		DEF_ARRAY_IN(Coord, GridVelocity, DeviceType::GPU, "Input grid velocity");

		DEF_ARRAYLIST_IN(int, PGNeighborIds, DeviceType::GPU, "Return neighbor grids ids of particles");

		DEF_ARRAY_IN(Coord, AdaptGridPosition, DeviceType::GPU, "Input adaptive grid position");

		DEF_VAR_IN(Real, GridSpacing, "Spacing distance of grids");
		//DEF_VAR_IN(Vec3i, BoxGridNum, "");

		DEF_VAR_IN(Real, SamplingDistance, "");

		DEF_VAR(Real, FlipAlpha, 0.95, "FLIP-PIC Blender Ratio");

		DEF_VAR(Real, Stiffness, 5.0, "Stiffness of fluid particle");

		DEF_VAR_IN(uint, FrameNumber, "Frame number");

		DECLARE_ENUM(InterpolationModel,
			PIC = 0,
			APIC = 1,
			FLIP = 2,
			NFLIP = 3);

		DEF_ENUM(InterpolationModel, InterpolationModel, InterpolationModel::FLIP, "");

	private:

		DArray<Real> m_gridMass;
		DArray<Matrix> m_C;			//Affine velocity of particle
		DArray<Real> m_J;			//Divergence of velocity

		DArray<Coord> m_pVelo_old;	//Old Particle Velocity 
		DArray<Coord> m_gVelo_old;	//Old Grid Velocity
		DArray<Coord> m_FlipVelocity; //FLIP Velocity


		Real E = 5;				//stiffness 
		Real particle_Density = 1.0f;
		Real particle_Volume = 0.0f;
		Real particle_Mass = 0.0f;

		Coord origin = Coord(0.0f);

		bool initial = true;

	private:
		std::shared_ptr<SummationDensity<TDataType>> m_summation;
	};

	IMPLEMENT_TCLASS(FlipFluidExplicitSolver, TDataType)

}