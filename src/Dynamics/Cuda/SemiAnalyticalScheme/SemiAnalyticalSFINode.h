#pragma once
#include "ParticleSystem/ParticleFluid.h"

#include "Topology/TriangleSet.h"


namespace  dyno
{
	template <typename T> class ParticleSystem;
	/*!
	*	\class	SemiAnalyticalSFINode
	*	\brief	Semi-Analytical Solid Fluid Interaction
	*
	*	This class defines all fields necessary to implement a one way coupling between particles and static boundary meshes.
	*
	*/

	template<typename TDataType>
	class SemiAnalyticalSFINode : public ParticleFluid<TDataType>
	{
		DECLARE_TCLASS(SemiAnalyticalSFINode, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;
		
		SemiAnalyticalSFINode();
		~SemiAnalyticalSFINode() override;

	public:
		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "Boundary triangular surface");
		
	public:
		DEF_VAR(Bool, Fast, false, "");
		DEF_VAR(Bool, SyncBoundary, false, "");
		
	protected:
		void resetStates() override;

		void preUpdateStates() override;
		void postUpdateStates() override;

		bool validateInputs() override;
	};
}