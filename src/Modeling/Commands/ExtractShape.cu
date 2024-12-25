#include "ExtractShape.h"
#include "GLPhotorealisticRender.h"
#include "Mapping/TextureMeshToTriangleSet.h"
#include "GLWireframeVisualModule.h"
#include "GLSurfaceVisualModule.h"

namespace dyno 
{
	
	template< typename uint >
	__global__ void GetShapeVerticesRange(
		DArray<uint> shapeIDs,
		int* min,
		int* max,
		int i
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= shapeIDs.size()) return;

		if (shapeIDs[pId] == i)
		{
			atomicMax(max, pId);
			atomicMin(min, pId);
		}
	}

	template< typename Vec3f , typename Vec2i>
	__global__ void extractShapeVertices(
		DArray<Vec3f> inVertices,
		DArray<Vec3f> vertices,
		DArray<Vec2i> idRange,
		int id,
		int offset
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= inVertices.size()) return;


		Vec2i range = idRange[id];

		if (pId <= range[1] && pId >= range[0])
		{
			vertices[pId - range[0] + offset] = inVertices[pId];
		}
	}

	template< typename Vec2f, typename Vec2i>
	__global__ void extractShapeVec2f(
		DArray<Vec2f> inTexcoords,
		DArray<Vec2f> texcoords,
		DArray<Vec2i> idRange,
		int id,
		int offset
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= inTexcoords.size()) return;


		Vec2i range = idRange[id];

		if (pId <= range[1] && pId >= range[0])
		{
			texcoords[pId - range[0] + offset] = inTexcoords[pId];
		}
	}

	template< typename uint, typename Vec2i>
	__global__ void extractShapeIds(
		DArray<uint> inShapeIds,
		DArray<uint> shapeIds,
		DArray<Vec2i> idRange,
		int id,
		int offset,
		int newId
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= inShapeIds.size()) return;


		Vec2i range = idRange[id];

		if (pId <= range[1] && pId >= range[0])
		{
			shapeIds[pId - range[0] + offset] = newId;
		}
	}

	template< typename Triangle >
	__global__ void extractShapeTriangles(
		DArray<Triangle> inTriangle,
		DArray<Triangle> triangle,
		int offset,
		int verticesOffset
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= inTriangle.size()) return;


		triangle[pId][0] = inTriangle[pId][0] - offset + verticesOffset;
		triangle[pId][1] = inTriangle[pId][1] - offset + verticesOffset;
		triangle[pId][2] = inTriangle[pId][2] - offset + verticesOffset;


	}


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

		this->stateResult()->setDataPtr(std::make_shared<TextureMesh>());
		this->stateResult()->promoteOuput();

		auto render = std::make_shared<GLPhotorealisticRender>();
		this->stateResult()->connect(render->inTextureMesh());

		this->graphicsPipeline()->pushModule(render);
		this->setForceUpdate(false);
	}

	template<typename TDataType>
	ExtractShape<TDataType>::~ExtractShape()
	{
	}

	
	template<typename TDataType>
	void ExtractShape<TDataType>::resetStates()
	{
		auto extractId = this->varShapeId()->getValue();
		//std::sort(extractId.begin(), extractId.end());
		//extractId.erase(std::unique(extractId.begin(), extractId.end()), extractId.end());

		auto inTexMesh = this->inInTextureMesh()->constDataPtr();

		for (size_t i = 0; i < extractId.size(); i++)
		{
			if (extractId[i] >= inTexMesh->shapes().size())
				return;
		}

		auto out = this->stateResult()->getDataPtr();

		std::vector<Vec2i> shapeVerticesRange;
		std::vector<int> shapeVerticesSize;
		shapeVerticesRange.resize(inTexMesh->shapes().size());
		shapeVerticesSize.resize(inTexMesh->shapes().size());

		//get Shape_VerticesRange
		for (int i = 0; i < inTexMesh->shapes().size(); i++)
		{
			int max = INT_MIN;
			int min = INT_MAX;

			int* d_max;
			int* d_min;

			cudaMalloc(&d_max, sizeof(int));
			cudaMalloc(&d_min, sizeof(int));

			cudaMemcpy(d_max, &max, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_min, &min, sizeof(int), cudaMemcpyHostToDevice);

			cuExecute(inTexMesh->shapeIds().size(),
				GetShapeVerticesRange,
				inTexMesh->shapeIds(),
				d_min,
				d_max,
				i
			);

			cudaMemcpy(&max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&min, d_min, sizeof(int), cudaMemcpyDeviceToHost);

			shapeVerticesRange[i] = Vec2i(min,max);

		}

		//get Shape_VerticesSize
		for (size_t i = 0; i < shapeVerticesRange.size(); i++)
		{
			shapeVerticesSize[i] = shapeVerticesRange[i][1] - shapeVerticesRange[i][0] + 1;
		}
		
		uint count = 0;

		for (auto it : extractId)
			count += shapeVerticesSize[it];			

		DArray<Vec3f> d_extractVertices;
		d_extractVertices.resize(count);

		DArray<Vec3f> d_extractNormals;
		d_extractNormals.resize(count);

		DArray<Vec2f> d_extractTexCoords;
		d_extractTexCoords.resize(count);

		DArray<uint>d_extractShapeIds;
		d_extractShapeIds.resize(count);

		DArray<Vec2i> d_range;
		d_range.assign(shapeVerticesRange);

		DArray<int> d_size;
		d_size.assign(shapeVerticesSize);

		int verticesOffset = 0;

		auto inMaterial = inTexMesh->materials();
		std::vector<std::shared_ptr<Material>>outMaterials;

		for (auto it : inMaterial)
			outMaterials.push_back(it);

		auto inShapes = inTexMesh->shapes();

		std::vector<std::shared_ptr<Shape>> outShapes;


		auto userTransform = this->varShapeTransform()->getValue();

		for (size_t i = 0; i < extractId.size(); i++)
		{
			auto id = extractId[i];

			cuExecute(inTexMesh->vertices().size(),
				extractShapeVertices,
				inTexMesh->vertices(),
				d_extractVertices,
				d_range,
				id,
				verticesOffset
			);

			cuExecute(inTexMesh->vertices().size(),
				extractShapeVertices,
				inTexMesh->normals(),
				d_extractNormals,
				d_range,
				id,
				verticesOffset
			);

			cuExecute(inTexMesh->vertices().size(),
				extractShapeVec2f,
				inTexMesh->texCoords(),
				d_extractTexCoords,
				d_range,
				id,
				verticesOffset
			);

			cuExecute(inTexMesh->vertices().size(),
				extractShapeIds,
				inTexMesh->shapeIds(),
				d_extractShapeIds,
				d_range,
				id,
				verticesOffset,
				i
			);

			int triangleIndexOffset = shapeVerticesRange[id][0];

			DArray<Triangle> d_triangle;
			d_triangle.resize(inTexMesh->shapes()[id]->vertexIndex.size());

			cuExecute(inTexMesh->shapes()[id]->vertexIndex.size(),
				extractShapeTriangles,
				inTexMesh->shapes()[id]->vertexIndex,
				d_triangle,
				triangleIndexOffset,
				verticesOffset
			);

			verticesOffset += shapeVerticesSize[id];

			if (id < inShapes.size())
			{
				auto element = std::make_shared<Shape>();
				auto currentShape = inShapes[id];
				element->vertexIndex.assign(d_triangle);
				element->normalIndex.assign(d_triangle);
				element->texCoordIndex.assign(d_triangle);
				element->boundingBox = currentShape->boundingBox;

				if (i < userTransform.size()) 
				{
					auto currentT = currentShape->boundingTransform;
					auto userT = userTransform[i];
					if (this->varOffset()->getValue()) 
					{
						element->boundingTransform.translation() = currentT.translation() + userT.translation();
						element->boundingTransform.scale() = currentT.scale() * userT.scale();
						element->boundingTransform.rotation() = userT.rotation() * currentT.rotation();
					}
					else 
					{
						element->boundingTransform.translation() = userT.translation();
						element->boundingTransform.scale() = userT.scale();
						element->boundingTransform.rotation() = currentT.rotation();
					}
				}
				else
					element->boundingTransform = currentShape->boundingTransform;

				element->material = currentShape->material;

				outShapes.push_back(element);

				d_triangle.clear();
			}

		}

		out->vertices().assign(d_extractVertices);
		out->normals().assign(d_extractNormals);
		out->texCoords().assign(d_extractTexCoords);
		out->shapeIds().assign(d_extractShapeIds);

		out->materials() = outMaterials;
		out->shapes() = outShapes;

		d_extractVertices.clear();
		d_extractNormals.clear();
		d_extractTexCoords.clear();
		d_extractShapeIds.clear();
		d_range.clear();
		d_size.clear();

	}

	DEFINE_CLASS(ExtractShape);

}