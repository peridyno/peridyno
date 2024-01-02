#pragma once
#include <Module/CustomModule.h>
#include <Collision/Attribute.h>

#include <map>

namespace dyno {
	
	template<typename TDataType>
	class ManualControl : public CustomModule
	{
		DECLARE_TCLASS(ManualControl, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		ManualControl();
		~ManualControl() override;

		void begin();
		void applyCustomBehavior();

	public:
	
		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");
	
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");

		DEF_VAR_IN(uint, FrameNumber, "Frame number");

		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "");
	
	protected:
		void updateImpl() override { this->begin(); this->applyCustomBehavior(); };
	};

}
