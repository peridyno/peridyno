#include "TextureMeshToTriangleSet.h"

namespace dyno
{
	IMPLEMENT_TCLASS(TextureMeshToTriangleSet, TDataType)

		template<typename TDataType>
	TextureMeshToTriangleSet<TDataType>::TextureMeshToTriangleSet()
		: TopologyMapping()
	{
		this->inTransform()->tagOptional(true);
	}

	template<typename TDataType>
	TextureMeshToTriangleSet<TDataType>::~TextureMeshToTriangleSet()
	{

	}

	template <typename Transform>
	__global__ void equalSize(
		DArrayList<Transform> globalTransform,
		int* size
	)
	{
		if (threadIdx.x == 0 && blockIdx.x == 0)
		{
			int count = globalTransform.size(); 
			if (count == 0)
			{
				*size = -1; 
				return;
			}

			int firstSize = globalTransform[0].size();
			if (firstSize == 0)
			{
				*size = -1;
				return;
			}
			for (int i = 1; i < count; i++)
			{
				if (globalTransform[i].size() != firstSize)
				{
					*size = -1; 

					return;
				}
			}

			*size = firstSize;
		}
		else
			return;
	}

	template <typename Coord, typename Transform>
	__global__ void TM2TS_TransformVertices(
		DArray<Coord> vertices,
		DArray<uint> shapeIds,
		DArray<Transform> localTransform,
		DArrayList<Transform> globalTransform)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vertices.size()) return;

		uint shapeId = shapeIds[tId];
		Coord v = vertices[tId];

		Transform locT = localTransform[shapeId];
		//TODO: This is a temporary code
		Transform globalT = globalTransform[shapeId][0];

		Coord s = globalT.scale();

		vertices[tId] = globalT.rotation() * Coord(v.x * s.x, v.y * s.y, v.z * s.z) + globalT.translation();
	}

	template <typename Coord, typename Transform>
	__global__ void TM2TS_TransformVerticesByInstanceID(
		DArray<Coord> sourceVertices,
		DArray<Coord> vertices,
		DArray<uint> shapeIds,
		DArray<Transform> localTransform,
		DArrayList<Transform> globalTransform,
		uint instanceID
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vertices.size()) return;

		uint shapeId = shapeIds[tId];
		Coord v = sourceVertices[tId];

		Transform locT = localTransform[shapeId];
		//TODO: This is a temporary code


		Transform globalT = globalTransform[shapeId][instanceID];

		Coord s = globalT.scale();

		vertices[tId + instanceID * sourceVertices.size()] = globalT.rotation() * Coord(v.x * s.x, v.y * s.y, v.z * s.z) + globalT.translation();
	}

	template< typename Triangle >
	__global__ void TM2TS_ConstructIndices(
		DArray<Triangle> sourceTriangle,
		DArray<Triangle> triangle,
		int instanceID,
		int verticesNum
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= sourceTriangle.size()) return;


		triangle[tId + sourceTriangle.size() * instanceID][0] = sourceTriangle[tId][0] + verticesNum * instanceID;
		triangle[tId + sourceTriangle.size() * instanceID][1] = sourceTriangle[tId][1] + verticesNum * instanceID;
		triangle[tId + sourceTriangle.size() * instanceID][2] = sourceTriangle[tId][2] + verticesNum * instanceID;


	}


	template <typename Coord, typename Transform>
	__global__ void TransformVertices(
		DArray<Coord> vertices,
		DArray<uint> shapeIds,
		DArray<Transform> localTransform
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vertices.size()) return;

		uint shapeId = shapeIds[tId];
		Coord v = vertices[tId];

		Transform locT = localTransform[shapeId];
		Coord s = locT.scale();

		vertices[tId] = locT.rotation() * Coord(v.x * s.x, v.y * s.y, v.z * s.z) + locT.translation();	

	}


	template<typename TDataType>
	bool TextureMeshToTriangleSet<TDataType>::apply()
	{
		if (this->outTriangleSet()->isEmpty()) {
			this->outTriangleSet()->allocate();
		}

		auto mesh = this->inTextureMesh()->constDataPtr();

		uint indexNum = 0;
		for (uint i = 0; i < mesh->shapes().size(); i++) {
			indexNum += mesh->shapes()[i]->vertexIndex.size();
		}

		auto ts = this->outTriangleSet()->getDataPtr();

		auto& vertices = ts->getPoints();
		auto& indices = ts->triangleIndices();

		vertices.assign(mesh->geometry()->vertices());

		if (indices.size() != indexNum)
		{
			indices.resize(indexNum);
		}

		uint offset = 0;
		for (uint i = 0; i < mesh->shapes().size(); i++) {
			uint num = mesh->shapes()[i]->vertexIndex.size();
			indices.assign(mesh->shapes()[i]->vertexIndex, num, offset, 0);

			offset += num;
		}
		DArray<TopologyModule::Triangle> srcIndices;
		srcIndices.assign(indices);

		if (!this->inTransform()->isEmpty())
		{
			uint N = mesh->shapes().size();
			CArray<Transform> hostT(N);
			DArray<Transform> devT(N);


			for (uint i = 0; i < N; i++)
			{
				hostT[i] = mesh->shapes()[i]->boundingTransform;
			}

			devT.assign(hostT);
			
			bool transformID0 = false;

			int sizeEqual = -1;
			int* d_sizeEqual;
			cudaMalloc(&d_sizeEqual, sizeof(int));
			cuExecute(1,
				equalSize,
				this->inTransform()->constData(),
				d_sizeEqual
			);
			cudaMemcpy(&sizeEqual, d_sizeEqual, sizeof(int), cudaMemcpyDeviceToHost);

			if (transformID0)
			{
				cuExecute(vertices.size(),
					TM2TS_TransformVertices,
					vertices,
					mesh->geometry()->shapeIds(),
					devT,
					this->inTransform()->constData());
			}
			else if (sizeEqual > 0)
			{

				vertices.resize(mesh->geometry()->vertices().size() * sizeEqual);
				indices.resize(indexNum * sizeEqual);


				for (size_t instanceID = 0; instanceID < sizeEqual; instanceID++)
				{
					cuExecute(mesh->geometry()->vertices().size(),
						TM2TS_TransformVerticesByInstanceID,
						mesh->geometry()->vertices(),
						vertices,
						mesh->geometry()->shapeIds(),
						devT,
						this->inTransform()->constData(),
						instanceID
					);

					cuExecute(indices.size(),
						TM2TS_ConstructIndices,
						srcIndices,
						indices,
						instanceID,
						mesh->geometry()->vertices().size()
					);

				}

			}




			hostT.clear();
			devT.clear();
		}
		else
		{
			uint N = mesh->shapes().size();
			CArray<Transform> hostT(N);
			DArray<Transform> devT(N);

			for (uint i = 0; i < N; i++)
			{
				hostT[i] = mesh->shapes()[i]->boundingTransform;
			}

			devT.assign(hostT);

			cuExecute(vertices.size(),
				TransformVertices,
				vertices,
				mesh->geometry()->shapeIds(),
				devT
			);

			hostT.clear();
			devT.clear();
		}

		ts->update();

		return true;
	}

	DEFINE_CLASS(TextureMeshToTriangleSet);


	IMPLEMENT_TCLASS(TextureMeshToTriangleSetNode, TDataType);

	template<typename TDataType>
	TextureMeshToTriangleSetNode<TDataType>::TextureMeshToTriangleSetNode()
		: Node()
	{
		this->setForceUpdate(false);

		auto mTM2TS = std::make_shared<TextureMeshToTriangleSet<TDataType>>();

		this->inTextureMesh()->connect(mTM2TS->inTextureMesh());
		mTM2TS->outTriangleSet()->connect(this->outTriangleSet());
		this->animationPipeline()->pushModule(mTM2TS);
	}

	template<typename TDataType>
	void TextureMeshToTriangleSetNode<TDataType>::resetStates()
	{
		Node::resetStates();
		this->animationPipeline()->update();
	}

	template<typename TDataType>
	void TextureMeshToTriangleSetNode<TDataType>::updateStates()
	{
		Node::updateStates();

	}

	DEFINE_CLASS(TextureMeshToTriangleSetNode);
}