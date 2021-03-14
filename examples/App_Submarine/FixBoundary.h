#pragma once
#include "Framework/ModuleCustom.h"
#include "ParticleSystem/Attribute.h"

namespace dyno {

	template<typename TDataType>
	class FixBoundary : public CustomModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		FixBoundary();
		~FixBoundary() override;

		void applyCustomBehavior() override;

	public:
		DEF_EMPTY_IN_ARRAY(Position, Coord, DeviceType::GPU, "");

		DEF_EMPTY_IN_ARRAY(Velocity, Coord, DeviceType::GPU, "");

		DEF_EMPTY_IN_ARRAY(VertexAttribute, Attribute, DeviceType::GPU, "");

	private:
	};

#ifdef PRECISION_FLOAT
template class FixBoundary<DataType3f>;
#else
template class FixBoundary<DataType3d>;
#endif

}
