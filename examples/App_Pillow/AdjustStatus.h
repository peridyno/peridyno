#pragma once
#include "Framework/ModuleCustom.h"
#include "ParticleSystem/Attribute.h"

namespace dyno {

	template<typename TDataType>
	class AdjustStatus : public CustomModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		AdjustStatus();
		~AdjustStatus() override;

		void applyCustomBehavior() override;

	public:
		DEF_EMPTY_IN_ARRAY(Position, Coord, DeviceType::GPU, "");

		DEF_EMPTY_IN_ARRAY(Velocity, Coord, DeviceType::GPU, "");

		DEF_EMPTY_IN_ARRAY(VertexAttribute, Attribute, DeviceType::GPU, "");

	private:
	};

#ifdef PRECISION_FLOAT
template class AdjustStatus<DataType3f>;
#else
template class AdjustStatus<DataType3d>;
#endif

}
