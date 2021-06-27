#pragma once
#include "Framework/Node.h"

#include "Topology/DistanceField3D.h"

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
	};
}
