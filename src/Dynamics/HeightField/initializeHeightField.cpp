#include "initializeHeightField.h"

#include "Ocean.h"
#include "CapillaryWave.h"
#include "OceanPatch.h"

#include "NodeFactory.h"

namespace dyno
{
	HeightFieldInitializer::HeightFieldInitializer()
	{
		
		TypeInfo::New<OceanPatch<DataType3f>>();
		//TypeInfo::New<Ocean<DataType3f>>();
		TypeInfo::New<CapillaryWave<DataType3f>>();

		initializeNodeCreators();
	}

	void HeightFieldInitializer::initializeNodeCreators()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto group = factory->addGroup(
			"Ocean",
			"Ocean",
			"ToolBarIco/HeightField/HeightField.png");

		group->addAction(
			"Ocean Patch",
			"ToolBarIco/HeightField/OceanPatch.png",
			[=]()->std::shared_ptr<Node> { return std::make_shared<OceanPatch<DataType3f>>(); });

		group->addAction(
			"Ocean",
			"ToolBarIco/HeightField/Ocean.png",
			[=]()->std::shared_ptr<Node> { return std::make_shared<Ocean<DataType3f>>(); });

		group->addAction(
			"CapillaryWave",
			"ToolBarIco/HeightField/CapillaryWave.png",
			[=]()->std::shared_ptr<Node> { return std::make_shared<CapillaryWave<DataType3f>>(); });
	}

}