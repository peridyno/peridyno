#include "Volume.h"
#include "Topology/SignedDistanceField.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(Volume, TDataType)

	template<typename TDataType>
	Volume<TDataType>::Volume()
		: Node()
	{
		auto sdf = std::make_shared<SignedDistanceField<TDataType>>();
		this->setTopologyModule(TypeInfo::cast<TopologyModule>(sdf));
	}

	template<typename TDataType>
	Volume<TDataType>::~Volume()
	{
	}
}