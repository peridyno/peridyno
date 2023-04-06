#include "initializeObjIO.h"

#include "NodeFactory.h"

#include "ObjLoader.h"
#include "GLWireframeVisualModule.h"
#include "GLSurfaceVisualModule.h"
#include "GLPointVisualModule.h"

#include "OBJexporter.h"
#include "ObjPointLoader.h"
#include "PLYexporter.h"

namespace dyno 
{
	std::atomic<OBJInitializer*> OBJInitializer::gInstance;
	std::mutex OBJInitializer::gMutex;

	PluginEntry* OBJInitializer::instance()
	{
		OBJInitializer* ins = gInstance.load(std::memory_order_acquire);
		if (!ins) {
			std::lock_guard<std::mutex> tLock(gMutex);
			ins = gInstance.load(std::memory_order_relaxed);
			if (!ins) {
				ins = new OBJInitializer();
				ins->setName("ObjIO");
				ins->setVersion("1.0");
				ins->setDescription("A ObjIO library");

				gInstance.store(ins, std::memory_order_release);
			}
		}

		return ins;
	}

	void OBJInitializer::initializeActions()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto page = factory->addPage(
			"Modeling", 
			"ToolBarIco/Modeling/Modeling.png");

		auto group = page->addGroup("Modeling");

		group->addAction(
			"Import OBJ",
			"ToolBarIco/Modeling/TriangularMesh.png",
			[=]()->std::shared_ptr<Node> { 
				auto node = std::make_shared<ObjMesh<DataType3f>>();

				//auto pointrender = std::make_shared<GLPointVisualModule>();
				//pointrender->setVisible(true);
				//pointrender->setColor(Vec3f(1, 0, 0));
				//node->stateTopology()->connect(pointrender->inPointSet());
				//node->graphicsPipeline()->pushModule(pointrender);

				//auto wirerender = std::make_shared<GLWireframeVisualModule>();
				//wirerender->setVisible(true);
				//wirerender->setColor(Vec3f(0, 1, 0));
				//node->stateTopology()->connect(wirerender->inEdgeSet());
				//node->graphicsPipeline()->pushModule(wirerender);

				return node; 
			});

		group->addAction(
			"OBJ Point Loader",
			"ToolBarIco/Modeling/ObjPointLoader.png",
			[=]()->std::shared_ptr<Node> {
				auto node = std::make_shared<ObjPoint<DataType3f>>();

				return node;
			});

		group->addAction(
			"Export OBJ",
			"ToolBarIco/Modeling/OBJExport_v3.png",
			[=]()->std::shared_ptr<Node> {
				auto node = std::make_shared<OBJExporter<DataType3f>>();

				auto module = std::make_shared<GLSurfaceVisualModule>();
				module->setColor(Vec3f(0.8, 0.52, 0.25));
				module->setVisible(true);
				//node->stateTopology()->connect(module->inTriangleSet());
				node->graphicsPipeline()->pushModule(module);

				return node;
			});


		group->addAction(
			"Ply Exporter",
			"ToolBarIco/Modeling/OBJExport_v3.png",
			[=]()->std::shared_ptr<Node> {
				auto node = std::make_shared<PlyExporter<DataType3f>>();

				auto module = std::make_shared<GLSurfaceVisualModule>();
				module->setColor(Vec3f(0.8, 0.52, 0.25));
				module->setVisible(true);
				node->graphicsPipeline()->pushModule(module);

				return node;
			});

	}
}

dyno::PluginEntry* ObjIO::initStaticPlugin()
{
	if (dyno::OBJInitializer::instance()->initialize())
		return dyno::OBJInitializer::instance();

	return nullptr;
}

PERIDYNO_API dyno::PluginEntry* ObjIO::initDynoPlugin()
{
	if (dyno::OBJInitializer::instance()->initialize())
		return dyno::OBJInitializer::instance();

	return nullptr;
}

