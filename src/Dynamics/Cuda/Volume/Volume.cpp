#include "Volume.h"
#include "Topology/SignedDistanceField.h"

namespace dyno
{
	template<typename TDataType>
	Volume<TDataType>::Volume()
		: Node()
	{
 		auto sdf = std::make_shared<SignedDistanceField<TDataType>>();
// 		this->setTopologyModule(TypeInfo::cast<TopologyModule>(sdf));
		this->stateTopology()->setDataPtr(TypeInfo::cast<TopologyModule>(sdf));

	}

	template<typename TDataType>
	Volume<TDataType>::~Volume()
	{
	}

	DEFINE_CLASS(Volume);
}