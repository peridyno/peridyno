#include "initializeGLRenderEngine.h"

#include "NodeFactory.h"

#include "Node/GLPointVisualNode.h"
#include "Node/GLSurfaceVisualNode.h"

#include "ColorMapping.h"

namespace dyno
{
	GLRenderEngineInitializer::GLRenderEngineInitializer()
	{
		TypeInfo::New<ColorMapping<DataType3f>>();

		initializeNodeCreators();
	}

	void GLRenderEngineInitializer::initializeNodeCreators()
	{
		NodeFactory* factory = NodeFactory::instance();

		auto group = factory->addGroup(
			"Rendering",
			"Rendering",
			"ToolBarIco/HeightField/HeightField.png");

		group->addAction(
			"Particle Renderer",
			"ToolBarIco/RigidBody/GhostParticles.png",
			[=]()->std::shared_ptr<Node> { return std::make_shared<GLPointVisualNode<DataType3f>>(); });

		group->addAction(
			"Surface Renderer",
			"ToolBarIco/HeightField/HeightField.png",
			[=]()->std::shared_ptr<Node> { return std::make_shared<GLSurfaceVisualNode<DataType3f>>(); });
	}

}