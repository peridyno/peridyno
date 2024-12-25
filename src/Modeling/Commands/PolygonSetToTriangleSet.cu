#pragma once
#include "PolygonSetToTriangleSet.h"
#include "cuda_runtime.h" 
#include <thrust/sort.h>
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "EarClipper.h"

namespace dyno
{
	__global__ void extractPolyIndices(
		DArray<uint> input,
		DArray<uint> output,
		int* arrayIndex
	) 
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= input.size()) return;

		if (input[tId] == 1) {
			int index = atomicAdd(arrayIndex, 1);
			output[index] = tId;
		}
	}

	__global__ void PolygonEdgeCounting(
		DArray<uint> counter3,
		DArray<uint> counter4,
		DArray<uint> counter5,
		DArrayList<uint> polygonIndices)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= polygonIndices.size()) return;

		counter3[tId] = polygonIndices[tId].size() == 3 ? 1 : 0;
		counter4[tId] = polygonIndices[tId].size() == 4 ? 1 : 0;
		counter5[tId] = polygonIndices[tId].size() >= 5 ? 1 : 0;
	}

	__global__ void PolygonSet_CountPoly2TriangleNumber(
		DArray<uint> counter,
		DArrayList<uint> polygonIndices)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= counter.size()) return;

		counter[tId] = polygonIndices[tId].size() - 2;
	}

		template<typename Triangle>
	__global__ void Edge4_ExtractTriangleIndices(
		DArray<Triangle> triangles,
		DArrayList<uint> polygonIndices,
		DArray<uint> radix)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= radix.size()) return;

		uint offset = radix[tId];

		auto& index = polygonIndices[tId];

		if (index.size() == 3)
		{
			uint v0 = index[0];
			uint v1 = index[1];
			uint v2 = index[2];
			triangles[offset] = Triangle(v0, v1, v2);
		}
	}

	template<typename Triangle>
	__global__ void CopyTriangles(
		DArray<Triangle> outTriangles,
		DArrayList<uint> polygonIndices,
		DArray<uint> triIndex,
		DArrayList<uint> poly2tri
		)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triIndex.size()) return;

		auto& index = polygonIndices[tId];
		if (index.size() == 3) 
		{
			uint offset = triIndex[tId];
			outTriangles[offset] = Triangle(index[0], index[1], index[2]);
			poly2tri[tId][0] = offset;
		}

	}

	template<typename Triangle>
	__global__ void Quads2Triangles(
		DArray<Triangle> outTriangles,
		DArrayList<uint> polygonIndices,
		DArray<Vec3f> pos,
		DArray<uint> quadIndex,
		DArrayList<uint> poly2tri,
		int offset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= quadIndex.size()) return;

		auto& index = polygonIndices[tId];
		printf("poly list :%d\n",index.size());
		if (index.size() == 4) 
		{
			uint id = 2 * quadIndex[tId] + offset;

			uint v0 = index[0];
			uint v1 = index[1];
			uint v2 = index[2];
			uint v3 = index[3];
			//判断点是否在外侧

			Vec3f A = pos[v0];
			Vec3f B = pos[v1];
			Vec3f C = pos[v2];
			Vec3f P = pos[v3];

			Vec3f AB = B - A;
			Vec3f BC = C - B;
			Vec3f CA = A - C;

			Vec3f AP = P - A;
			Vec3f BP = P - B;
			Vec3f CP = P - C;

			// 计算叉积
			Vec3f cross1 = AB.cross(AP).normalize();
			Vec3f cross2 = BC.cross(BP).normalize();
			Vec3f cross3 = CA.cross(CP).normalize();

			// 判断所有叉积的符号
			float dot1 = cross1.dot(cross2);
			float dot2 = cross1.dot(cross3);
			float dot3 = cross2.dot(cross3);

			if ((dot1 >= 0 && dot2 >= 0 && dot3 >= 0) || (dot1 <= 0 && dot2 <= 0 && dot3 <= 0))
			{
				outTriangles[id] = Triangle(v0, v1, v3);
				outTriangles[id + 1] = Triangle(v1, v2, v3);
			}
			else
			{
				outTriangles[id] = Triangle(v0, v1, v2);
				outTriangles[id + 1] = Triangle(v0, v2, v3);
			}

			poly2tri[tId][0] = id;
			poly2tri[tId][1] = id + 1;
		}
		

	}

	template<typename Triangle>
	__global__ void CopyTriangles2Triangles(
		DArray<Triangle> outTriangles,
		DArray<Triangle> inTriangles,
		uint offset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= inTriangles.size()) return;

		outTriangles[tId + offset] = inTriangles[tId];
	}


	__global__ void CounterLarger5Edges(
		DArray<uint> counter
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= counter.size()) return;

		if (counter[tId] <= 2) 
			counter[tId] = 0;
	}


	__global__ void updatepoly2triangleId_moreThan5Edge(
		DArrayList<uint> poly2tri,
		DArrayList<uint> polygonIndex,
		DArray<uint> scanResult,
		DArray<uint> edgeNum,
		uint offset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= polygonIndex.size()) return;

		uint count = edgeNum[tId];
		uint start = scanResult[tId];

		if (polygonIndex[tId].size() >= 5) 
		{
			//printf("tId- d%   start: %d   i: %d   count: %d\n", tId, start, i, count);

			for (uint i = 0; i < count; i++)
			{
				auto& list = poly2tri[tId];
				list[i] = offset + start + i;
				printf("tId- %d , start: %d, i: %d, result: %d , listsize: %d\n",tId, start,i,poly2tri[tId][i], list.size());

			}
		}

	}

	template<typename TDataType>
	void PolygonSetToTriangleSetModule<TDataType>:: convert(std::shared_ptr<PolygonSet<TDataType>> polygonset, std::shared_ptr<TriangleSet<TDataType>> triset,DArrayList<uint>& poly2tri)
	{
		triset->clear();

		auto& polylist = polygonset->polygonIndices();

		auto lists = polylist.lists();


		DArray<uint> edge3(lists.size());
		DArray<uint> edge4(lists.size());
		DArray<uint> edge5(lists.size());

		uint polyNum = polygonset->polygonIndices().size();

		DArray<uint> poly2triCounter(polyNum);
		cuExecute(polyNum,
			PolygonSet_CountPoly2TriangleNumber,
			poly2triCounter,
			polygonset->polygonIndices());

		poly2tri.resize(poly2triCounter);

		int tNum = thrust::reduce(thrust::device, poly2triCounter.begin(), poly2triCounter.begin() + poly2triCounter.size());
		if (tNum < 1)
			return;

		auto& triangles = triset->getTriangles();
		triangles.resize(tNum);

		cuExecute(polyNum,
			PolygonEdgeCounting,
			edge3,
			edge4,
			edge5,
			polygonset->polygonIndices()
		);

		uint total_num3 = thrust::reduce(thrust::device, edge3.begin(), edge3.begin() + edge3.size());
		thrust::exclusive_scan(thrust::device, edge3.begin(), edge3.begin() + edge3.size(), edge3.begin());

		cuExecute(polyNum,
			CopyTriangles,
			triangles,
			polygonset->polygonIndices(),
			edge3,
			poly2tri
		);

		uint total_num4 = thrust::reduce(thrust::device, edge4.begin(), edge4.begin() + edge4.size());
		thrust::exclusive_scan(thrust::device, edge4.begin(), edge4.begin() + edge4.size(), edge4.begin());


		//poly2triCounter

		cuExecute(polyNum,
			Quads2Triangles,
			triangles,
			polygonset->polygonIndices(),
			polygonset->getPoints(),
			edge4,
			poly2tri,
			total_num3
		);

		uint total_num5 = thrust::reduce(thrust::device, edge5.begin(), edge5.begin() + edge5.size());


		CArray<uint> c_p2t;
		c_p2t.assign(poly2triCounter);
		for (size_t i = 0; i < c_p2t.size(); i++)
		{
			printf("poly2triCounter: %d\n", c_p2t[i]);
		}

		cuExecute(polyNum,
			CounterLarger5Edges,
			poly2triCounter
		);
		DArray<uint> startId;
		startId.assign(poly2triCounter);

		printf("************\n");
		c_p2t.assign(poly2triCounter);
		for (size_t i = 0; i < c_p2t.size(); i++)
		{
			printf("poly2triCounter: %d\n", c_p2t[i]);
		}

		thrust::exclusive_scan(thrust::device, startId.begin(), startId.begin() + startId.size(), startId.begin());

		printf("*******SCAN*****\n");
		c_p2t.assign(startId);
		for (size_t i = 0; i < c_p2t.size(); i++)
		{
			printf("startId: %d\n", c_p2t[i]);
		}

		cuExecute(polyNum,
			updatepoly2triangleId_moreThan5Edge,
			poly2tri,
			polygonset->polygonIndices(),
			startId,
			poly2triCounter,
			total_num3 + total_num4
		);



		int* arrayIndex;
		cudaMalloc((void**)&arrayIndex, sizeof(int));
		cudaMemset(arrayIndex, 0, sizeof(int));

		DArray<uint> d_v5_faceIds(total_num5);

		cuExecute(edge5.size(),
			extractPolyIndices,
			edge5,
			d_v5_faceIds,
			arrayIndex
		);

		CArray<uint> c_v5_faceIds;
		c_v5_faceIds.assign(d_v5_faceIds);

		EarClipper<DataType3f> earClip;
		
		CArrayList<uint> c_polygons;
		c_polygons.assign(polylist);
		CArray<Vec3f> c_points;
		c_points.assign(polygonset->getPoints());

		int poly2triOffset = 0;
		for (size_t i = 0; i < c_v5_faceIds.size(); i++)
		{

			auto ptIndexList = c_polygons[c_v5_faceIds[i]];
			std::vector<Vec3f> pts;
			for (size_t j = 0; j < ptIndexList.size(); j++)
			{
				auto pt = ptIndexList[j];
				pts.push_back(c_points[pt]);
			}

			std::vector<TopologyModule::Triangle> c_earTriangles;
			earClip.polyClip(pts, c_earTriangles);//
			if (c_earTriangles.size() != (pts.size() - 2)) 
			{
				printf("EarClip Error!!\n");
				std::vector<TopologyModule::Triangle> c_earTest;
				earClip.polyClip(pts, c_earTest);//
			}

			for (size_t k = 0; k < c_earTriangles.size(); k++)
			{
				c_earTriangles[k][0] = ptIndexList[c_earTriangles[k][0]];
				c_earTriangles[k][1] = ptIndexList[c_earTriangles[k][1]];
				c_earTriangles[k][2] = ptIndexList[c_earTriangles[k][2]];
			}

			int aa = 1;

			DArray<TopologyModule::Triangle> d_earTriangles;
			d_earTriangles.assign(c_earTriangles);

			cuExecute(d_earTriangles.size(),
				CopyTriangles2Triangles,
				triangles,
				d_earTriangles,
				total_num3 + total_num4*2 + poly2triOffset
			);
			d_earTriangles.clear();

			poly2triOffset += (ptIndexList.size() - 2);
		}

		triset->setPoints(polygonset->getPoints());
		triset->update();

		edge3.clear();
		edge4.clear();
		edge5.clear();
		poly2triCounter.clear();

		d_v5_faceIds.clear();

		CArrayList<uint> c_poly2tri;
		c_poly2tri.assign(poly2tri);
		for (size_t i = 0; i < c_poly2tri.size(); i++)
		{
			printf("%d: ",i);
			auto& list = c_poly2tri[i];
			printf("size %d: ", list.size());

			for (size_t j = 0; j < list.size(); j++)
			{
				printf("%d - ",list[j]);
			}
			printf("\n");
		}

	};
	DEFINE_CLASS(PolygonSetToTriangleSetModule);

	template<typename TDataType>
	PolygonSetToTriangleSetNode<TDataType>::PolygonSetToTriangleSetNode()
	{
		mPolygonSetToTriangleSetMoudle = std::make_shared<PolygonSetToTriangleSetModule<DataType3f>>();
		this->inPolygonSet()->connect(mPolygonSetToTriangleSetMoudle->inPolygonSet());

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<DataType3f>>());

		this->stateTriangleSet()->promoteOuput();

		auto surfaceRender = std::make_shared<GLSurfaceVisualModule>();
		auto wireRender = std::make_shared<GLWireframeVisualModule>();
		this->stateTriangleSet()->connect(surfaceRender->inTriangleSet());
		this->stateTriangleSet()->connect(wireRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(surfaceRender);
		this->graphicsPipeline()->pushModule(wireRender);

	};
	DEFINE_CLASS(PolygonSetToTriangleSetNode);

}