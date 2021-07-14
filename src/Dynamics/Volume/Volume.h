#pragma once
#include "Node.h"

#include "Topology/SignedDistanceField.h"

namespace dyno {

	template<typename TDataType>
	class Volume : public Node
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Volume();
		~Volume() override;

		virtual void updateVolume() = 0;
	public:
		DEF_INSTANCE_STATE(SignedDistanceField<TDataType>, SDF, "");
	};
}
