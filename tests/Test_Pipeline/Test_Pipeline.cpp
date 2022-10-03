#include "gtest/gtest.h"

#include "SceneGraph.h"
#include "SceneGraphFactory.h"
#include "Module/Pipeline.h"

using namespace dyno;

TEST(Pipeline, connect)
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	SceneGraphFactory::instance()->pushScene(scn);
	SceneGraphFactory::instance()->popScene();
}
