#include "ExtractTriangleSets.h"
#include "Topology/TriangleSets.h"
#include "GLSurfaceVisualModule.h"
#include <thrust/sort.h>

namespace dyno
{
	template<typename TDataType>
	ExtractTriangleSets<TDataType>::ExtractTriangleSets()
		: ParametricModel<TDataType>()
	{
		this->stateTriangleSets()->setDataPtr(std::make_shared<TriangleSets<TDataType>>());

		auto inSurfaceRender = std::make_shared<GLSurfaceVisualModule>();
		this->inTriangleSets()->connect(inSurfaceRender->inTriangleSet());
		inSurfaceRender->setAlpha(0.1);
		inSurfaceRender->setColor(Color::LightGray());

		this->graphicsPipeline()->pushModule(inSurfaceRender);

		auto glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Color(0.8f, 0.52f, 0.25f));
		glModule->setVisible(true);
		this->stateTriangleSets()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);

	}

	template<typename TDataType>
	void ExtractTriangleSets<TDataType>::resetStates()
	{
		if (this->inTriangleSets()->isEmpty())
			return;
		this->stateTriangleSets()->getDataPtr()->clear();
		auto tsArray = this->Extract(this->inTriangleSets()->getDataPtr(), this->stateTriangleSets()->getDataPtr(), this->varID()->getValue());

		for (size_t i = 0; i < this->varShapeTransform()->getValue().size(); i++)
		{
			if (i >= tsArray.size())
				break;

			Transform3f trans = this->varShapeTransform()->getValue()[i];

			std::shared_ptr<TriangleSet<TDataType>> triSet = tsArray[i];

			Vec3f lo;
			Vec3f hi;
			//void requestBoundingBox(Coord& lo, Coord& hi);
			triSet->requestBoundingBox(lo, hi);
			Vec3f Center = (lo + hi) / 2;

			triSet->translate(-Center);
			triSet->scale(trans.scale());
			triSet->rotate(Quat<Real>(trans.rotation()));
			triSet->translate(trans.translation() + Center);

		}

		this->stateTriangleSets()->getDataPtr()->load(tsArray);
		for (size_t i = 0; i < tsArray.size(); i++)
		{
			tsArray[i]->clear();
		}
	}


	template<typename uint>
	__global__ void MarkTriangle(
		DArray<uint> shapeIds,
		DArray<int> mark,
		int groupId
		)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= shapeIds.size()) return;

		if (shapeIds[tId] == groupId)
			mark[tId] = 1;
		else 
			mark[tId] = 0;
	}

	template<typename Triangle, typename uint>
	__global__ void MarkPoints(
		DArray<Triangle> Triangles,
		DArray<uint> shapeIds,
		DArray<int> markPt,
		int groupId
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= shapeIds.size()) return;

		if (shapeIds[tId] == groupId)
		{
			int v0 = Triangles[tId][0];
			int v1 = Triangles[tId][1];
			int v2 = Triangles[tId][2];

			markPt[v0] = 1;
			markPt[v1] = 1;
			markPt[v2] = 1;
		}
	}

	__global__ void BuildNew2OldPtIndex(
		DArray<int> markPt,
		DArray<int> scanMarkPt,
		DArray<int> New2OldPtIndex,
		DArray<int> Old2NewPtIndex
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= markPt.size()) return;

		 if (markPt[tId] == 1)
		 {
			 New2OldPtIndex[scanMarkPt[tId]] = tId;
			 Old2NewPtIndex[tId] = scanMarkPt[tId];
		 }
	}

	template<typename Vec3f>
	__global__ void UpdateNewPoints(
		DArray<Vec3f> newPoints,
		DArray<Vec3f> oldPoints,
		DArray<int> New2OldPtIndex	
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= newPoints.size()) return;

		newPoints[tId] = oldPoints[New2OldPtIndex[tId]];
	}

	template< typename Triangle>
	__global__ void UpdateNewTriangles(
		DArray<Triangle> newTriangles,
		DArray<Triangle> oldTriangles,
		DArray<int> old2NewPtIndex,
		DArray<int> New2OldTriIndex
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= newTriangles.size()) return;

		Triangle old = oldTriangles[New2OldTriIndex[tId]];

		newTriangles[tId][0] = old2NewPtIndex[old[0]];
		newTriangles[tId][1] = old2NewPtIndex[old[1]];
		newTriangles[tId][2] = old2NewPtIndex[old[2]];
	}

	__global__ void BuildNew2OldTriIndex(
		DArray<int> markTri,
		DArray<int> scanMarkTri,
		DArray<int> New2OldTriIndex
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= markTri.size()) return;

		if (markTri[tId] == 1)
		{
			New2OldTriIndex[scanMarkTri[tId]] = tId;
		}
	}


	template<typename TDataType>
	std::vector<std::shared_ptr<TriangleSet<TDataType>>> ExtractTriangleSets<TDataType>::Extract(std::shared_ptr<TriangleSets<TDataType>> triSets, std::shared_ptr<TriangleSets<TDataType>> outTriSet, std::vector<int> triSetId)
	{
		
		std::vector<std::shared_ptr<TriangleSet<TDataType>>> tsArray;

		for (auto groupId : triSetId)
		{
			if (groupId >= triSets->shapeSize())
			{
				std::cout<<"ExtractTriangleSets : Out of Id\n";
				continue;
			}
			
			auto currentTriSet = std::make_shared<TriangleSet<TDataType>>();
			tsArray.push_back(currentTriSet);

			auto shapeIds = triSets->shapeIds();

			auto oldTriangles = triSets->triangleIndices();
			auto oldPoints = triSets->getPoints();
			DArray<int> markPt(oldPoints.size());

			cudaMemset(markPt.begin(), 0, markPt.size() * sizeof(int));

			cuExecute(shapeIds.size(),
				MarkPoints,
				oldTriangles,
				shapeIds,
				markPt,
				groupId
			);

			int newPointsSize = thrust::reduce(thrust::device, markPt.begin(), markPt.begin() + markPt.size());

			DArray<int> scanMarkPt;
			scanMarkPt.assign(markPt);
			thrust::exclusive_scan(thrust::device, scanMarkPt.begin(), scanMarkPt.begin() + scanMarkPt.size(), scanMarkPt.begin());

			DArray<int> New2OldPtIndex(newPointsSize);
			DArray<int> old2NewPtIndex(oldPoints.size());

			cuExecute(markPt.size(),
				BuildNew2OldPtIndex,
				markPt,
				scanMarkPt,
				New2OldPtIndex,
				old2NewPtIndex
			);

			DArray<Vec3f> NewPts(newPointsSize);

			cuExecute(NewPts.size(),
				UpdateNewPoints,
				NewPts,
				oldPoints,
				New2OldPtIndex
			);

			DArray<int> markTri(shapeIds.size());

			cudaMemset(markTri.begin(), 0, markTri.size() * sizeof(int));

			cuExecute(shapeIds.size(),
				MarkTriangle,
				shapeIds,
				markTri,
				groupId
			);

			int newTriSize = thrust::reduce(thrust::device, markTri.begin(), markTri.begin() + markTri.size());

			DArray<int> scanMarkTri;
			scanMarkTri.assign(markTri);

			thrust::exclusive_scan(thrust::device, scanMarkTri.begin(), scanMarkTri.begin() + scanMarkTri.size(), scanMarkTri.begin());

			DArray<TopologyModule::Triangle> NewTriangles(newTriSize);

			DArray<int> New2OldTriIndex(newTriSize);

			cuExecute(markTri.size(),
				BuildNew2OldTriIndex,
				markTri,
				scanMarkTri,
				New2OldTriIndex
			);

			cuExecute(markTri.size(),
				UpdateNewTriangles,
				NewTriangles,
				oldTriangles,
				old2NewPtIndex,
				New2OldTriIndex
			);

			currentTriSet->setPoints(NewPts);
			currentTriSet->setTriangles(NewTriangles);
			currentTriSet->update();

			markPt.clear();
			scanMarkPt.clear();
			New2OldPtIndex.clear();
			old2NewPtIndex.clear();
			NewPts.clear();
			markTri.clear();
			scanMarkTri.clear();
			NewTriangles.clear();
			New2OldTriIndex.clear();
		}

		return tsArray;
	}

	DEFINE_CLASS(ExtractTriangleSets);
}