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

		virtual void updateVolume() {};

	public:
		DEF_INSTANCE_STATE(SignedDistanceField<TDataType>, SDF, "");

		DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");
	};
}
