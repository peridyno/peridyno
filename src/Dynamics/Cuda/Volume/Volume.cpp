#include "Volume.h"
#include "Topology/LevelSet.h"

namespace dyno
{
	template<typename TDataType>
	Volume<TDataType>::Volume()
		: Node()
	{
 		this->stateLevelSet()->setDataPtr(std::make_shared<LevelSet<TDataType>>());
	}

	template<typename TDataType>
	Volume<TDataType>::~Volume()
	{
	}

	template<typename TDataType>
	std::string Volume<TDataType>::getNodeType()
	{
		return "Volume";
	}

	DEFINE_CLASS(Volume);
}