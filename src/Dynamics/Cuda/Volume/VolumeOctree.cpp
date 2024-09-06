#include "VolumeOctree.h"

#include "Module/VolumeToTriangleSet.h"

#include "GLSurfaceVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	VolumeOctree<TDataType>::VolumeOctree()
		: Node()
	{
		auto mapper = std::make_shared<VolumeToTriangleSet<TDataType>>();
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

	DEFINE_CLASS(VolumeOctree);
}