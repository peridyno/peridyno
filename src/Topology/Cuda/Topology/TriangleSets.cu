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
			tNum += ts->getTriangles().size();
		}

		auto& vertices = this->getPoints();
		auto& indices = this->getTriangles();
		
		vertices.resize(vNum);
		indices.resize(tNum);
		mShapeIds.resize(tNum);

		uint vOffset = 0;
		uint tOffset = 0;
		for (auto ts : tsArray)
		{
			auto& vSrc = ts->getPoints();
			auto& tSrc = ts->getTriangles();
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
			auto& tSrc = ts->getTriangles();

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

	DEFINE_CLASS(TriangleSets);
}