#include "ExtractShape.h"
#include "GLPhotorealisticRender.h"
#include "Mapping/TextureMeshToTriangleSet.h"
#include "GLWireframeVisualModule.h"
#include "GLSurfaceVisualModule.h"

namespace dyno 
{
	
	template<typename TDataType>
	ExtractShape<TDataType>::ExtractShape()
	{
		auto inTexMesh = this->inInTextureMesh();
		auto convertModule = std::make_shared<TextureMeshToTriangleSet<DataType3f>>();
		inTexMesh->connect(convertModule->inTextureMesh());

		//auto wireRender = std::make_shared<GLWireframeVisualModule>();
		//convertModule->outTriangleSet()->connect(wireRender->inEdgeSet());
		//wireRender->setColor(Color::Black());
		//wireRender->setVisible(false);

		auto surfaceRender = std::make_shared<GLSurfaceVisualModule>();
		convertModule->outTriangleSet()->connect(surfaceRender->inTriangleSet());
		surfaceRender->setAlpha(0.1);
		surfaceRender->setColor(Color::LightGray());

		//this->graphicsPipeline()->pushModule(wireRender);

		this->graphicsPipeline()->pushModule(convertModule);
		this->graphicsPipeline()->pushModule(surfaceRender);

		this->stateTarget()->setDataPtr(std::make_shared<TextureMesh>());

		auto render = std::make_shared<GLPhotorealisticRender>();
		this->stateTarget()->connect(render->inTextureMesh());

		this->graphicsPipeline()->pushModule(render);

	}

	template<typename TDataType>
	ExtractShape<TDataType>::~ExtractShape()
	{
	}

	
	template<typename TDataType>
	void ExtractShape<TDataType>::resetStates()
	{
		auto inTexMesh = this->inInTextureMesh()->getDataPtr();
		auto out = this->stateTarget()->getDataPtr();

		out->vertices().assign(inTexMesh->vertices());
		out->normals().assign(inTexMesh->normals());
		out->texCoords().assign(inTexMesh->texCoords());
		out->shapeIds().assign(inTexMesh->shapeIds());

		auto inMaterial = inTexMesh->materials();
		std::vector<std::shared_ptr<Material>>outMaterials;

		for (auto it : inMaterial)
			outMaterials.push_back(it);


		auto inShapes = inTexMesh->shapes();

		auto extractId = this->varShapeId()->getValue();
		std::vector<std::shared_ptr<Shape>> outShapes;
		
		std::sort(extractId.begin(), extractId.end());
		extractId.erase(std::unique(extractId.begin(), extractId.end()), extractId.end());

		for (auto id : extractId)
		{
			if (id < inShapes.size()) 
			{
				auto element = std::make_shared<Shape>();
				auto it = inShapes[id];
				element->vertexIndex.assign(it->vertexIndex);
				element->normalIndex.assign(it->normalIndex);
				element->texCoordIndex.assign(it->texCoordIndex);
				element->boundingBox = it->boundingBox;
				element->boundingTransform = it->boundingTransform;
				element->material = it->material;

				outShapes.push_back(element);
			}

		}

		out->materials() = outMaterials;
		out->shapes() = outShapes;


	}

	DEFINE_CLASS(ExtractShape);

}