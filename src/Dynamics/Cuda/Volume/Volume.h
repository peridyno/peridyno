#pragma once
#include "Node.h"

#include "Topology/LevelSet.h"

namespace dyno {

	template<typename TDataType>
	class Volume : public Node
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Volume();
		~Volume() override;

		std::string getNodeType() override;

		virtual void updateVolume() {};

	public:
		DEF_INSTANCE_STATE(LevelSet<TDataType>, LevelSet, "");
	};
}
