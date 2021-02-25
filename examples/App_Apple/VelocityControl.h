#pragma once
#include "Framework/ModuleCustom.h"
#include "Framework/FieldArray.h"
#include "ParticleSystem/Attribute.h"

#include <map>

namespace dyno {

	template<typename TDataType>
	class VelocityControl : public CustomModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VelocityControl();
		~VelocityControl() override;

		void applyCustomBehavior() override;

	public:
		DEF_EMPTY_IN_ARRAY(Position, Coord, DeviceType::GPU, "");

		DEF_EMPTY_IN_ARRAY(Velocity, Coord, DeviceType::GPU, "");

	private:
	};

#ifdef PRECISION_FLOAT
template class VelocityControl<DataType3f>;
#else
template class VelocityControl<DataType3d>;
#endif

}
