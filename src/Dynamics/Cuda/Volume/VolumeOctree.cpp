#include "VolumeOctree.h"

#include "Module/AdaptiveVolumeToTriangleSet.h"

#include "GLSurfaceVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	VolumeOctree<TDataType>::VolumeOctree()
		: Node()
	{
		auto mapper = std::make_shared<AdaptiveVolumeToTriangleSet<TDataType>>();
		this->stateSDFTopology()->connect(mapper->ioVolume());
		this->graphicsPipeline()->pushModule(mapper);

		auto renderer = std::make_shared<GLSurfaceVisualModule>();
		mapper->outTriangleSet()->connect(renderer->inTriangleSet());
		this->graphicsPipeline()->pushModule(renderer);
	}

	template<typename TDataType>
	VolumeOctree<TDataType>::~VolumeOctree()
	{
	}

	template<typename TDataType>
	std::string VolumeOctree<TDataType>::getNodeType()
	{
		return "Adaptive Volume";
	}

	DEFINE_CLASS(VolumeOctree);
}