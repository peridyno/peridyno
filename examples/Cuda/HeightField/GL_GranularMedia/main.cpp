#include <GlfwApp.h>

#include <SceneGraph.h>

#include <HeightField/GranularMedia.h>

#include "Mapping/HeightFieldToTriangleSet.h"

#include <GLSurfaceVisualModule.h>

#include <HeightField/SurfaceParticleTracking.h>
#include "Module/ComputeModule.h"

using namespace std;
using namespace dyno;

class InializeGranularMedia : public ComputeModule
{
public:
	InializeGranularMedia() {};

public:
	DEF_INSTANCE_IN(HeightField<DataType3f>, InitialHeightField, "");

	DEF_ARRAY2D_IN(Vec4f, Grid, DeviceType::GPU, "");

	DEF_ARRAY2D_IN(Vec4f, GridNext, DeviceType::GPU, "");

protected:
	void compute()
	{
		auto grid = this->inGrid()->getDataPtr();
		auto gridNext = this->inGridNext()->getDataPtr();

		auto exW = grid->nx();
		auto exH = grid->ny();

		CArray2D<Vec4f> initializer(exW, exH);

		for (uint i = 0; i < exW; i++)
		{
			for (uint j = 0; j < exH; j++)
			{
				if (abs(i - exW / 2) < 10 && abs(j - exW / 2) < 10)
				{
					initializer(i, j) = Vec4f(5, 0, 0, 0);
				}
				else
					initializer(i, j) = Vec4f(0, 0, 0, 0);
			}
		}

		this->inGrid()->assign(initializer);
		this->inGridNext()->assign(initializer);

		initializer.clear();
	}
};

std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto sand = scn->addNode(std::make_shared<GranularMedia<DataType3f>>());
	sand->varOrigin()->setValue(Vec3f(-32.0f, 0.0f, -32.0f));

	auto initializer = std::make_shared<InializeGranularMedia>();
	sand->stateGrid()->connect(initializer->inGrid());
	sand->stateGridNext()->connect(initializer->inGridNext());
	sand->stateHeightField()->connect(initializer->inInitialHeightField());
	sand->resetPipeline()->pushModule(initializer);


// 	auto mapper = std::make_shared<HeightFieldToTriangleSet<DataType3f>>();
// 	sand->stateHeightField()->connect(mapper->inHeightField());
// 	sand->graphicsPipeline()->pushModule(mapper);
// 
// 	auto sRender = std::make_shared<GLSurfaceVisualModule>();
// 	sRender->setColor(Color(0.8, 0.8, 0.8));
// 	sRender->varUseVertexNormal()->setValue(true);
// 	mapper->outTriangleSet()->connect(sRender->inTriangleSet());
// 	sand->graphicsPipeline()->pushModule(sRender);

	auto tracking = scn->addNode(std::make_shared<SurfaceParticleTracking<DataType3f>>());
	sand->connect(tracking->importGranularMedia());

	return scn;
}

int main()
{
	GlfwApp app;
	app.initialize(1024, 768);

	app.setSceneGraph(createScene());
	app.renderWindow()->getCamera()->setUnitScale(52);

	app.mainLoop();

	return 0;
}