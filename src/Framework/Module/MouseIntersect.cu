#include "MouseIntersect.h"

namespace dyno
{
	template<typename TDataType>
	MouseIntersect<TDataType>::MouseIntersect()
		: Node()
	{

	}

	template<typename TDataType>
	MouseIntersect<TDataType>::~MouseIntersect()
	{

	}

	template<typename TDataType>
	void MouseIntersect<TDataType>::resetStates()
	{
	}

	template<typename TDataType>
	void MouseIntersect<TDataType>::updateStates()
	{

	}

	template<typename TDataType>
	void MouseIntersect<TDataType>::calcIntersect()
	{
		auto initialTriangleSet = this->inInitialTriangleSet()->getData();
		DArray<Triangle> triangles = initialTriangleSet.getTriangles();
		DArray<int> intersected;
		intersected.resize(triangles.size());
		DArray<int> unintersected;
		intersected.resize(triangles.size());

		cuExecute(triangles.size(),
			CalIntersectedTris,
			triangles,
			intersected,
			unintersected,
			this->inMouseRay()->getData()
		);

		DArray<int> intersected_o;
		intersected_o.assign(intersected);

		thrust::exclusive_scan(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), intersected.begin());
		int intersected_size = intersected[intersected.size()-1]+1;
		DArray<Triangle> intersected_triangles;
		intersected_triangles.resize(intersected_size);

		thrust::exclusive_scan(thrust::device, unintersected.begin(), unintersected.begin() + unintersected.size(), unintersected.begin());
		int unintersected_size = unintersected[unintersected.size() - 1] + 1;
		DArray<Triangle> unintersected_triangles;
		unintersected_triangles.resize(unintersected_size);

		cuExecute(triangles.size(),
			AssignOutTriangles,
			triangles,
			intersected_triangles,
			unintersected_triangles,
			intersected,
			unintersected,
			intersected_o
		);

		this->stateSelectedTriangleSet()->getDataPtr()->setTriangles(intersected_triangles);
		this->stateOtherTriangleSet()->getDataPtr()->setTriangles(unintersected_triangles);
	}

	template <typename Triangle,typename Real>
	__global__ void CalIntersectedTris(
		DArray<Triangle> triangles,
		DArray<int> intersected,
		DArray<int> unintersected,
		TRay3D<Real> mouseray)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= triangles.size()) return;
		
		TPoint3D<Real> p;
		intersected[pId]=mouseray.intersect(TTriangle3D<Real>(triangles[pId].data[0], triangles[pId].data[1], triangles[pId].data[2]),p);
		unintersected[pId] = (intersected[pId] == 1 ? 0 : 1);
	}

	template <typename Triangle>
	__global__ void AssignOutTriangles(
		DArray<Triangle> triangles,
		DArray<Triangle> intersected_triangles,
		DArray<Triangle> unintersected_triangles,
		DArray<int> intersected,
		DArray<int> unintersected,
		DArray<int> intersected_o)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= triangles.size()) return;

		if (intersected_o[pId] == 1) 
		{
			intersected_triangles[intersected[pId]] = triangles[pId];
		}
		else 
		{
			unintersected_triangles[unintersected[pId]] = triangles[pId];

		}
	}

	DEFINE_CLASS(MouseIntersect);
}