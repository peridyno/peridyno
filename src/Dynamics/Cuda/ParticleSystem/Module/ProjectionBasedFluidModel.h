#pragma once
#include "Module/GroupModule.h"

#include "Collision/Attribute.h"

namespace dyno
{
	template<typename TDataType>
	class ProjectionBasedFluidModel : public GroupModule
	{
		DECLARE_TCLASS(ProjectionBasedFluidModel, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ProjectionBasedFluidModel();
		virtual ~ProjectionBasedFluidModel() {};

	public:
		DEF_VAR(Real, SmoothingLength, Real(0.006), "The smoothing length in SPH");
		DEF_VAR(Real, SamplingDistance, Real(0.005), "Particle samplilng distance");

		DEF_VAR_IN(Real, TimeStep, "Time step size!");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");
		DEF_ARRAY_IN(Coord, Force, DeviceType::GPU, "");

		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, Normal, DeviceType::GPU, "");
	};
}