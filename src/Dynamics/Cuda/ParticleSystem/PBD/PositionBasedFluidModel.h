#pragma once
#include "Module/GroupModule.h"

#include "Collision/Attribute.h"

namespace dyno
{
	/*!
	*	\class	ParticleSystem
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/
	template<typename TDataType>
	class PositionBasedFluidModel : public GroupModule
	{
		DECLARE_TCLASS(PositionBasedFluidModel, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PositionBasedFluidModel();
		virtual ~PositionBasedFluidModel() {};

	public:
		DEF_VAR_IN(Real, SamplingDistance, "Sampling distance");
		DEF_VAR_IN(Real, SmoothingLength, "Smoothing length");

		DEF_VAR_IN(Real, TimeStep, "Time step size!");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");

		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "");
	};
}