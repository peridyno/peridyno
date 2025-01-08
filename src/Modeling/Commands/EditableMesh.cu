#pragma once
#include "EditableMesh.h"
#include "cuda_runtime.h" 
#include <thrust/sort.h>
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

namespace dyno
{
	//__global__ void extractPolyIndices(
	//	DArray<uint> input,
	//	DArray<uint> output,
	//	int* arrayIndex
	//) 
	//{
	//	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	//	if (tId >= input.size()) return;

	//	if (input[tId] == 1) {
	//		int index = atomicAdd(arrayIndex, 1);
	//		printf("output index = %d\n",index);
	//		output[index] = tId;
	//	}
	//}

	template<typename Triangle>
	__global__ void updateNormal(
		DArray<Vec3f> points,
		DArray<Triangle> triangles,
		DArrayList<uint> polygonIndex)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= polygonIndex.size()) return;


	}

	template<typename TDataType>
	void EditableMesh<TDataType>::resetStates()
	{

	};


	DEFINE_CLASS(EditableMesh);

	

}