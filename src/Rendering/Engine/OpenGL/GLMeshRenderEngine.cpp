#include "GLMeshRenderEngine.h"

// Visual modules for rendering
#include "Backend/Cuda/Module/GLSurfaceVisualModule.h"

//
#include "GLRenderHelper.h"
#include "GLVisualModule.h"

#include "ShadowMap.h"
#include "SSAO.h"
#include "FXAA.h"
#include "Envmap.h"

// dyno
#include "SceneGraph.h"
#include "Action.h"

// GLM

#include <OrbitCamera.h>
#include <TrackballCamera.h>
#include <unordered_set>
#include <memory>

#include "screen.vert.h"
#include "blend.frag.h"
#include "postprocess.frag.h"
#include "surface.frag.h"
#include "Module/GLPhotorealisticRender.h"
#include "Topology/TriangleSet.h"
#include "Topology/TextureMesh.h"
#include "Topology/DiscreteElements.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"
#include "Mapping/DiscreteElementsToTriangleSet.h"

namespace dyno
{
	GLMeshRenderEngine::GLMeshRenderEngine()
	{
	}

	GLMeshRenderEngine::~GLMeshRenderEngine()
	{
		this->terminate();
	}

	void GLMeshRenderEngine::addField(FBase* field)
	{
		if (!renderSceneGraph)
			renderSceneGraph = std::make_shared<SceneGraph>();

		std::shared_ptr<Node> fieldWrapper = std::make_shared<Node>();
		fieldWrapper->addField(field);

		if (auto triSet = dynamic_cast<FInstance<TriangleSet<DataType3f>>*>(field)) 
		{
			auto triModule = std::make_shared<GLSurfaceVisualModule>();
			triSet->connect(triModule->inTriangleSet());
			fieldWrapper->graphicsPipeline()->pushModule(triModule);
		}
		else if (auto edgeSet = dynamic_cast<FInstance<EdgeSet<DataType3f>>*>(field)) 
		{
			auto edgeModule = std::make_shared<GLWireframeVisualModule>();
			edgeSet->connect(edgeModule->inEdgeSet());
			fieldWrapper->graphicsPipeline()->pushModule(edgeModule);
		}
		else if (auto ptSet = dynamic_cast<FInstance<PointSet<DataType3f>>*>(field)) 
		{
			auto ptModule = std::make_shared<GLPointVisualModule>();
			ptSet->connect(ptModule->inPointSet());
			fieldWrapper->graphicsPipeline()->pushModule(ptModule);
		}
		else if (auto texMesh = dynamic_cast<FInstance<TextureMesh>*>(field)) 
		{
			auto photoModule = std::make_shared<GLPhotorealisticRender>();
			texMesh->connect(photoModule->inTextureMesh());
			fieldWrapper->graphicsPipeline()->pushModule(photoModule);
		}
		else if (auto basicElement = dynamic_cast<FInstance<DiscreteElements<DataType3f>>*>(field)) 
		{
			auto element2Tri = std::make_shared<DiscreteElementsToTriangleSet<DataType3f>>();
			basicElement->connect(element2Tri->inDiscreteElements());
			fieldWrapper->graphicsPipeline()->pushModule(element2Tri);
			auto triModule = std::make_shared<GLSurfaceVisualModule>();
			element2Tri->outTriangleSet()->connect(triModule->inTriangleSet());
			fieldWrapper->graphicsPipeline()->pushModule(triModule);
		}

		renderSceneGraph->addNode(fieldWrapper);

	}


	std::string GLMeshRenderEngine::name() const
	{
		return std::string("GL Mesh Render Engine");
	}

}
