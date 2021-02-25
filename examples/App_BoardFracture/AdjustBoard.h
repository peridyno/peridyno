#pragma once
#include "Framework/ModuleCustom.h"
#include "Framework/FieldArray.h"
#include "ParticleSystem/Attribute.h"

#include <map>

namespace dyno {

	template<typename TDataType>
	class AdjustBoard : public CustomModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		AdjustBoard();
		~AdjustBoard() override;

		void applyCustomBehavior() override;

	public:
		DEF_EMPTY_IN_ARRAY(Position, Coord, DeviceType::GPU, "");

		DEF_EMPTY_IN_ARRAY(Velocity, Coord, DeviceType::GPU, "");

		DEF_EMPTY_IN_ARRAY(VertexAttribute, Attribute, DeviceType::GPU, "");

	private:
	};

#ifdef PRECISION_FLOAT
template class AdjustBoard<DataType3f>;
#else
template class AdjustBoard<DataType3d>;
#endif

}
