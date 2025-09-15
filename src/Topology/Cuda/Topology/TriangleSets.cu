#include "TriangleSets.h"

namespace dyno
{
	template<typename TDataType>
	TriangleSets<TDataType>::TriangleSets()
		: TriangleSet<TDataType>()
	{
	}

	template<typename TDataType>
	TriangleSets<TDataType>::~TriangleSets()
	{
		mShapeIds.clear();
	}

	template<typename Triangle>
	__global__ void TSS_UpdateIndexAndShapeIds(
		DArray<Triangle> indices,
		DArray<uint> shapeIds,
		uint indexSize,
		uint shapeId,
		uint vertexOffset,
		uint indexOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= indexSize) return;

		Triangle t = indices[indexOffset + tId];
		t[0] += vertexOffset;
		t[1] += vertexOffset;
		t[2] += vertexOffset;

		indices[indexOffset + tId] = t;
		shapeIds[indexOffset + tId] = shapeId;
	}

	template<typename TDataType>
	void TriangleSets<TDataType>::load(std::vector<std::shared_ptr<TriangleSet<TDataType>>>& tsArray)
	{
		mShapeSize = tsArray.size();
		for (auto ts : tsArray)
		{
			if (!ts)
				return;
		}
		uint vNum = 0;
		uint tNum = 0;
		for (auto ts : tsArray)
		{
			vNum += ts->getPoints().size();
			tNum += ts->triangleIndices().size();
		}

		auto& vertices = this->getPoints();
		auto& indices = this->triangleIndices();

		vertices.resize(vNum);
		indices.resize(tNum);
		mShapeIds.resize(tNum);

		uint vOffset = 0;
		uint tOffset = 0;
		for (auto ts : tsArray)
		{
			auto& vSrc = ts->getPoints();
			auto& tSrc = ts->triangleIndices();
			vertices.assign(vSrc, vSrc.size(), vOffset, 0);
			indices.assign(tSrc, tSrc.size(), tOffset, 0);

			vOffset += vSrc.size();
			tOffset += tSrc.size();
		}

		vOffset = 0;
		tOffset = 0;
		uint shapeId = 0;
		for (auto ts : tsArray)
		{
			auto& vSrc = ts->getPoints();
			auto& tSrc = ts->triangleIndices();

			uint num = tSrc.size();
			cuExecute(num,
				TSS_UpdateIndexAndShapeIds,
				indices,
				mShapeIds,
				num,
				shapeId,
				vOffset,
				tOffset);

			vOffset += vSrc.size();
			tOffset += tSrc.size();
			shapeId++;
		}

		this->update();
	}

	template<typename Triangle>
	__global__ void TS_UpdateAppendIndex(
		DArray<Triangle> indices,
		DArray<uint> shapeId,
		uint vertexOffset,
		uint indexOffset,
		uint appendIndex)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= indices.size() - indexOffset) return;

		Triangle t = indices[indexOffset + tId];
		t[0] += vertexOffset;
		t[1] += vertexOffset;
		t[2] += vertexOffset;

		indices[indexOffset + tId] = t;
		shapeId[indexOffset + tId] = appendIndex;

	}


	template<typename TDataType>
	void TriangleSets<TDataType>::appendShape(std::vector<Vec3f>& vertices, std::vector<TopologyModule::Triangle>& triangles)
	{
		auto& pts = this->getPoints();
		auto& tris = this->triangleIndices();
		DArray<Vec3f> tempPts;
		DArray<TopologyModule::Triangle> tempTris;
		DArray<uint> tempId;

		tempPts.assign(pts);
		tempTris.assign(tris);
		tempId.assign(mShapeIds);

		int vNum0 = pts.size();
		int tNum0 = tris.size();

		int tNum1 = triangles.size();

		CArray<Vec3f> cvtsOld;
		cvtsOld.assign(tempPts);

		pts.resize(pts.size() + vertices.size());
		tris.resize(tris.size() + triangles.size());
		mShapeIds.resize(tris.size());

		pts.assign(tempPts, vNum0, 0, 0);
		tris.assign(tempTris, tNum0, 0, 0);
		mShapeIds.assign(tempId, tNum0, 0, 0);

		pts.assign(vertices, vertices.size(), vNum0, 0);
		tris.assign(triangles, triangles.size(), tNum0, 0);

		cuExecute(tNum1,
			TS_UpdateAppendIndex,
			tris,
			mShapeIds,
			vNum0,
			tNum0,
			mShapeSize);

		mShapeSize++;

		this->update();
	}

	template<typename TDataType>
	void TriangleSets<TDataType>::appendShape(std::vector<Vec3f>& vertices, CArray<TopologyModule::Triangle>& triangles)
	{
		auto& pts = this->getPoints();
		auto& tris = this->triangleIndices();
		DArray<Vec3f> tempPts;
		DArray<TopologyModule::Triangle> tempTris;
		DArray<uint> tempId;

		tempPts.assign(pts);
		tempTris.assign(tris);
		tempId.assign(mShapeIds);

		int vNum0 = pts.size();
		int tNum0 = tris.size();

		int tNum1 = triangles.size();

		CArray<Vec3f> cvtsOld;
		cvtsOld.assign(tempPts);

		pts.resize(pts.size() + vertices.size());
		tris.resize(tris.size() + triangles.size());
		mShapeIds.resize(tris.size());

		pts.assign(tempPts, vNum0, 0, 0);
		tris.assign(tempTris, tNum0, 0, 0);
		mShapeIds.assign(tempId, tNum0, 0, 0);

		pts.assign(vertices, vertices.size(), vNum0, 0);
		tris.assign(triangles, triangles.size(), tNum0, 0);

		cuExecute(tNum1,
			TS_UpdateAppendIndex,
			tris,
			mShapeIds,
			vNum0,
			tNum0,
			mShapeSize);

		mShapeSize++;

		this->update();
	}

	template<typename TDataType>
	void TriangleSets<TDataType>::appendShape(DArray<Vec3f>& vertices, DArray<TopologyModule::Triangle>& triangles)
	{
		auto& pts = this->getPoints();
		auto& tris = this->triangleIndices();
		DArray<Vec3f> tempPts;
		DArray<TopologyModule::Triangle> tempTris;
		DArray<uint> tempId;

		tempPts.assign(pts);
		tempTris.assign(tris);
		tempId.assign(mShapeIds);

		int vNum0 = pts.size();
		int tNum0 = tris.size();

		int tNum1 = triangles.size();

		CArray<Vec3f> cvtsOld;
		cvtsOld.assign(tempPts);

		pts.resize(pts.size() + vertices.size());
		tris.resize(tris.size() + triangles.size());
		mShapeIds.resize(tris.size());

		pts.assign(tempPts, vNum0, 0, 0);
		tris.assign(tempTris, tNum0, 0, 0);
		mShapeIds.assign(tempId, tNum0, 0, 0);

		pts.assign(vertices, vertices.size(), vNum0, 0);
		tris.assign(triangles, triangles.size(), tNum0, 0);

		cuExecute(tNum1,
			TS_UpdateAppendIndex,
			tris,
			mShapeIds,
			vNum0,
			tNum0,
			mShapeSize);

		mShapeSize++;

		this->update();
	}

	DEFINE_CLASS(TriangleSets);
}