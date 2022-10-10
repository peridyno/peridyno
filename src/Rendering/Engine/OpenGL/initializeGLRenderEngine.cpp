#include "initializeGLRenderEngine.h"

#include "NodeFactory.h"

#include "Node/GLPointVisualNode.h"
#include "Node/GLSurfaceVisualNode.h"

#include "ColorMapping.h"

namespace dyno
{
	GLRenderEngineInitializer::GLRenderEngineInitializer()
	{
		this->initialize();
	}

	void GLRenderEngineInitializer::initializeNodeCreators()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto page = factory->addPage(
			"Rendering",
			"ToolBarIco/Node/Display.png");

		auto group = page->addGroup("Rendering");

		group->addAction(
			"Particle Renderer",
			"ToolBarIco/RigidBody/GhostParticles.png",
			[=]()->std::shared_ptr<Node> { return std::make_shared<GLPointVisualNode<DataType3f>>(); });

		group->addAction(
			"Surface Renderer",
			"ToolBarIco/Rendering/SurfaceRender_v2.png",
			[=]()->std::shared_ptr<Node> { return std::make_shared<GLSurfaceVisualNode<DataType3f>>(); });
	}

}