#pragma once
#include "ParticleSystem/ParticleFluid.h"
#include "Topology/TriangleSets.h"
#include "Topology/TriangleSet.h"
#include "Topology/LevelSet.h"

namespace  dyno
{
	template <typename T> class ParticleSystem;
	/*!
	*	\class	SemiImplicitIncompressibleSPHModel
	*	\brief	Semi-Analytical Solid Fluid Interaction
	*
	*	This class represents an implementation of "Semi-Analytical Modeling of Fluid-Structure Interaction for Smoothed Particle Hydrodynamics".
	*
	*/

	template<typename TDataType>
	class SemiAnalyticalParticleFluid : public ParticleFluid<TDataType>
	{
		DECLARE_TCLASS(SemiAnalyticalParticleFluid, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename Topology::Triangle Triangle;

		SemiAnalyticalParticleFluid();
		~SemiAnalyticalParticleFluid() override;

	public:
		DEF_INSTANCE_IN(TriangleSets<TDataType>, TriangleSets, "Boundary triangular surface");

	public:
		DEF_VAR(Real, SearchRadius, 0.02, "SearchRadius");
		DEF_VAR_STATE(Real, TimeStep_CFL, Real(0.001), "Time step size");
		DEF_VAR_STATE(Real, SimulationTime, Real(0.0), "SimulationTime");
		DEF_ARRAY_OUT(Real, Density, DeviceType::GPU, "Density");
		DEF_ARRAY_OUT(Real, Kappas, DeviceType::GPU, "Kappas");
		DEF_ARRAY_STATE(Coord, PreTriangleVertexMerge, DeviceType::GPU, "pretriangle_vertex");


	protected:
		void resetStates() override;

		void preUpdateStates() override;
		void postUpdateStates() override;

		bool validateInputs() override;

	private:
		std::vector<std::shared_ptr<Module>> mModules;
	};
}
