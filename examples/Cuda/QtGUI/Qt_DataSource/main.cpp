#include <QtApp.h>
#include <GLRenderEngine.h>

#include "Auxiliary/DataSource.h"
#include "Auxiliary/DebugInfo.h"

using namespace dyno;

/**
 * @brief This example demonstrates how to use data sources inside PeriDyno.
 */
class Source : public Node
{
	DECLARE_CLASS(Source);
public:
	Source() {
		auto floatSource = std::make_shared<FloatSource>();

		auto printFloat = std::make_shared<PrintFloat>();

		floatSource->outFloat()->connect(printFloat->inFloat());

		this->animationPipeline()->pushModule(floatSource);
		this->animationPipeline()->pushModule(printFloat);
	};

	~Source() override {};
};

IMPLEMENT_CLASS(Source);


int main(int, char**)
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	//Create a sphere
	auto src = scn->addNode(std::make_shared<Source>());
	

	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}
