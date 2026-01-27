#include "TextureMeshMerge.h"
#include "GLPhotorealisticRender.h"

namespace dyno 
{
	template<typename Vec3f>
	__global__ void mergeVec3f(
		DArray<Vec3f> v0,
		DArray<Vec3f> v1,
		DArray<Vec3f> target,
		int sizeV0
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= target.size()) return;

		if (pId < sizeV0) 
			target[pId] = v0[pId];
		else 
			target[pId] = v1[pId - sizeV0];
	}

	template<typename Vec2f>
	__global__ void mergeVec2f(
		DArray<Vec2f> v0,
		DArray<Vec2f> v1,
		DArray<Vec2f> target,
		int sizeV0
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= target.size()) return;

		if (pId < sizeV0)
			target[pId] = v0[pId];
		else
			target[pId] = v1[pId - sizeV0];
	}

	__global__ void mergeUint(
		DArray<uint> v0,
		DArray<uint> v1,
		DArray<uint> target,
		int sizeV0
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= target.size()) return;

		if (pId < sizeV0)
			target[pId] = v0[pId];
		else
			target[pId] = v1[pId - sizeV0];
	}

	__global__ void mergeShapeId(
		DArray<uint> v0,
		DArray<uint> v1,
		DArray<uint> target,
		int sizeV0,
		int shapeSize
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= target.size()) return;

		if (pId < sizeV0)
			target[pId] = v0[pId];
		else
			target[pId] = v1[pId - sizeV0] + shapeSize;
	}

	template<typename Triangle>
	__global__ void updateVertexIndex(
		DArray<Triangle> v,
		DArray<Triangle> target,
		int offset
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= target.size()) return;

		target[pId] = Triangle(v[pId][0] + offset, v[pId][1] + offset, v[pId][2] + offset);
	}

	template<typename TDataType>
	TextureMeshMerge<TDataType>::TextureMeshMerge()
	{
		this->stateTextureMesh()->setDataPtr(std::make_shared<TextureMesh>());
		
		auto render = std::make_shared<GLPhotorealisticRender>();
		this->stateTextureMesh()->connect(render->inTextureMesh());
		this->stateTextureMesh()->promoteOuput();

		this->graphicsPipeline()->pushModule(render);
		this->setForceUpdate(false);
	}

	template<typename TDataType>
	TextureMeshMerge<TDataType>::~TextureMeshMerge()
	{
	}

	template<typename TDataType>
	void TextureMeshMerge<TDataType>::merge(const std::shared_ptr<TextureMesh>& texMesh01, const std::shared_ptr<TextureMesh>& texMesh02, std::shared_ptr<TextureMesh>& out)
	{
		auto vertices01 = texMesh01->geometry()->vertices();
		auto vertices02 = texMesh02->geometry()->vertices();
	
		out->geometry()->vertices().resize(vertices01.size() + vertices02.size());

		cuExecute(out->geometry()->vertices().size(),
			mergeVec3f,
			vertices01,
			vertices02,
			out->geometry()->vertices(),
			vertices01.size()
		);

		auto normals01 = texMesh01->geometry()->normals();
		auto normals02 = texMesh02->geometry()->normals();
		out->geometry()->normals().resize(normals01.size() + normals02.size());

		cuExecute(out->geometry()->normals().size(),
			mergeVec3f,
			normals01,
			normals02,
			out->geometry()->normals(),
			normals01.size()
		);

		auto texCoords01 = texMesh01->geometry()->texCoords();
		auto texCoords02 = texMesh02->geometry()->texCoords();
		out->geometry()->texCoords().resize(texCoords01.size() + texCoords02.size());

		cuExecute(out->geometry()->texCoords().size(),
			mergeVec2f,
			texCoords01,
			texCoords02,
			out->geometry()->texCoords(),
			texCoords01.size()
		);

		auto shapeIds01 = texMesh01->geometry()->shapeIds();
		auto shapeIds02 = texMesh02->geometry()->shapeIds();
		out->geometry()->shapeIds().resize(shapeIds01.size() + shapeIds02.size());

		cuExecute(out->geometry()->texCoords().size(),
			mergeShapeId,
			shapeIds01,
			shapeIds02,
			out->geometry()->shapeIds(),
			shapeIds01.size(),
			texMesh01->shapes().size()
		);


		auto shapes01 = texMesh01->shapes();
		auto shapes02 = texMesh02->shapes();

		std::vector<std::shared_ptr<Shape>> outShapes;

		for (auto it : shapes01)
		{
			auto element = std::make_shared<Shape>();
			element->vertexIndex.assign(it->vertexIndex);
			element->normalIndex.assign(it->normalIndex);
			element->texCoordIndex.assign(it->texCoordIndex);
			element->boundingBox = it->boundingBox;
			element->boundingTransform = it->boundingTransform;
			element->material = it->material;

			outShapes.push_back(element);
		}


		for (auto it : shapes02)
		{
			auto element = std::make_shared<Shape>();
			element->vertexIndex.assign(it->vertexIndex);
			element->normalIndex.assign(it->normalIndex);
			element->texCoordIndex.assign(it->texCoordIndex);
			element->boundingBox = it->boundingBox;
			element->boundingTransform = it->boundingTransform;
			element->material = it->material;

			outShapes.push_back(element);

			cuExecute(it->vertexIndex.size(),
				updateVertexIndex,
				it->vertexIndex,
				element->vertexIndex,
				vertices01.size()
			);

			cuExecute(it->normalIndex.size(),
				updateVertexIndex,
				it->normalIndex,
				element->normalIndex,
				normals01.size()
			);

			cuExecute(it->texCoordIndex.size(),
				updateVertexIndex,
				it->texCoordIndex,
				element->texCoordIndex,
				texCoords01.size()
			);

		}

		out->shapes() = outShapes;

	}

	template<typename TDataType>
	void TextureMeshMerge<TDataType>::resetStates()
	{
		auto texMesh01 = this->inFirst()->constDataPtr();
		auto texMesh02 = this->inSecond()->constDataPtr();

		auto out = this->stateTextureMesh()->getDataPtr();
		
		this->merge(texMesh01,texMesh02,out);
	}

	DEFINE_CLASS(TextureMeshMerge);

}