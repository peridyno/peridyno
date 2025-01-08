#include "Volume.h"

#include "Topology/LevelSet.h"

#include "Module/VolumeToTriangleSet.h"

//Rendering
#include "GLSurfaceVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	Volume<TDataType>::Volume()
		: Node()
	{
		this->setAutoHidden(true);

		auto mapper = std::make_shared<VolumeToTriangleSet<TDataType>>();
		this->stateLevelSet()->connect(mapper->ioVolume());
		this->graphicsPipeline()->pushModule(mapper);

		auto renderer = std::make_shared<GLSurfaceVisualModule>();
		mapper->outTriangleSet()->connect(renderer->inTriangleSet());
		this->graphicsPipeline()->pushModule(renderer);
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