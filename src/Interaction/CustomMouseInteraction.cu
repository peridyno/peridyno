#include "CustomMouseInteraction.h"
#include <thrust/sort.h>
#include <iostream>
#include <OrbitCamera.h>

namespace dyno
{
	template<typename TDataType>
	CustomMouseIteraction<TDataType>::CustomMouseIteraction()
	{
		this->ray1 = TRay3D<Real>();
		this->ray2 = TRay3D<Real>();
	}

	template<typename TDataType>
	void CustomMouseIteraction<TDataType>::onEvent(PMouseEvent event)
	{
		if (event.actionType == AT_PRESS)
		{
			printf("Mouse pressed: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
			this->ray1.origin = event.ray.origin;
			this->ray1.direction = event.ray.direction;
			this->x1 = event.x;
			this->y1 = event.y;
		}
		else if (event.actionType == AT_RELEASE)
		{
			printf("Mouse released: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
			this->ray2.origin = event.ray.origin;
			this->ray2.direction = event.ray.direction;
			this->x2 = event.x;
			this->y2 = event.y;
			this->calcIntersectClick();
		}
		else
		{
			printf("%f %f \n", event.x, event.y);
			printf("Mouse repeated: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
		}
	}

	template <typename Triangle, typename Real, typename Coord>
	__global__ void CalIntersectedTrisRay(
		DArray<Coord> points,
		DArray<Triangle> triangles,
		DArray<int> intersected,
		DArray<int> unintersected,
		TRay3D<Real> mouseray)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= triangles.size()) return;

		TPoint3D<Real> p;
		intersected[pId] = mouseray.intersect(TTriangle3D<Real>(points[triangles[pId].data[0]], points[triangles[pId].data[1]], points[triangles[pId].data[2]]), p);
		unintersected[pId] = (intersected[pId] == 1 ? 0 : 1);
	}

	template <typename Triangle, typename Real, typename Coord>
	__global__ void CalIntersectedTrisBox(
		DArray<Coord> points,
		DArray<Triangle> triangles,
		DArray<int> intersected,
		DArray<int> unintersected,
		TPlane3D<Real> plane13,
		TPlane3D<Real> plane42,
		TPlane3D<Real> plane14,
		TPlane3D<Real> plane32)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= triangles.size()) return;

		TPoint3D<Real> p1= TPoint3D<Real>(points[triangles[pId].data[0]]);
		TPoint3D<Real> p2 = TPoint3D<Real>(points[triangles[pId].data[1]]);
		TPoint3D<Real> p3 = TPoint3D<Real>(points[triangles[pId].data[2]]);
		bool flag = false;
		for (int i = 1; i <= 3; i++) {
			int flag1 = ((points[triangles[pId].data[i]] - plane13.origin).dot(plane13.normal)) * ((points[triangles[pId].data[i]] - plane42.origin).dot(plane42.normal));
			int flag2 = ((points[triangles[pId].data[i]] - plane14.origin).dot(plane14.normal)) * ((points[triangles[pId].data[i]] - plane32.origin).dot(plane32.normal));
			if (flag1 > 0 && flag2 > 0)
				flag = true;
				break;
		}
		if (flag)
			intersected[pId] = 1;
		else
			intersected[pId] = 0;
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

	template<typename TDataType>
	void CustomMouseIteraction<TDataType>::calcIntersectClick()
	{
		TriangleSet<TDataType> initialTriangleSet = this->inInitialTriangleSet()->getData();
		DArray<Coord> points = initialTriangleSet.getPoints();
		DArray<Triangle> triangles = initialTriangleSet.getTriangles();
		DArray<int> intersected;
		intersected.resize(triangles.size());
		DArray<int> unintersected;
		unintersected.resize(triangles.size());
		std::cout <<"Triangle Num:" << triangles.size() << std::endl;
		std::cout << "Point Num:" << points.size() << std::endl;
		cuExecute(triangles.size(),
			CalIntersectedTrisRay,
			points,
			triangles,
			intersected,
			unintersected,
			this->ray2
		);

		DArray<int> intersected_o;
		intersected_o.assign(intersected);

		int intersected_size = thrust::reduce(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), intersected.begin());
		DArray<Triangle> intersected_triangles;
		intersected_triangles.resize(intersected_size);

		int unintersected_size = thrust::reduce(thrust::device, unintersected.begin(), unintersected.begin() + unintersected.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, unintersected.begin(), unintersected.begin() + unintersected.size(), unintersected.begin());
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
		this->outSelectedTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedTriangleSet()->getDataPtr()->setTriangles(intersected_triangles);
		this->outOtherTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outOtherTriangleSet()->getDataPtr()->setTriangles(unintersected_triangles);
	}

	/*template<typename TDataType>
	void CustomMouseIteraction<TDataType>::calcIntersectDrag()
	{
		GlfwApp* activeWindow = (GlfwApp*)glfwGetWindowUserPointer(window);
		auto camera = activeWindow->activeCamera();
		TRay3D<Real> ray1 = this->ray1;
		TRay3D<Real> ray2 = this->ray2;
		TRay3D<Real> ray3 = camera->castRayInWorldSpace((float)x1, (float)y2);
		TRay3D<Real> ray4 = camera->castRayInWorldSpace((float)x2, (float)y1);

		TPlane3D<Real> plane13 = TPlane3D<Real>(ray1.origin, ray4.origin - ray1.origin);
		TPlane3D<Real> plane42 = TPlane3D<Real>(ray4.origin, ray1.origin - ray4.origin);
		TPlane3D<Real> plane14 = TPlane3D<Real>(ray1.origin, ray3.origin - ray1.origin);
		TPlane3D<Real> plane32 = TPlane3D<Real>(ray3.origin, ray1.origin - ray3.origin);

		TriangleSet<TDataType> initialTriangleSet = this->inInitialTriangleSet()->getData();
		DArray<Coord> points = initialTriangleSet.getPoints();
		DArray<Triangle> triangles = initialTriangleSet.getTriangles();
		DArray<int> intersected;
		intersected.resize(triangles.size());
		DArray<int> unintersected;
		unintersected.resize(triangles.size());
		std::cout << "Triangle Num:" << triangles.size() << std::endl;
		std::cout << "Point Num:" << points.size() << std::endl;
		cuExecute(triangles.size(),
			CalIntersectedTrisBox,
			points,
			triangles,
			intersected,
			unintersected,
			plane13,
			plane42,
			plane14,
			plane32
		);

		DArray<int> intersected_o;
		intersected_o.assign(intersected);

		int intersected_size = thrust::reduce(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), intersected.begin());
		DArray<Triangle> intersected_triangles;
		intersected_triangles.resize(intersected_size);

		int unintersected_size = thrust::reduce(thrust::device, unintersected.begin(), unintersected.begin() + unintersected.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, unintersected.begin(), unintersected.begin() + unintersected.size(), unintersected.begin());
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
		this->outSelectedTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedTriangleSet()->getDataPtr()->setTriangles(intersected_triangles);
		this->outOtherTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outOtherTriangleSet()->getDataPtr()->setTriangles(unintersected_triangles);
	}*/
	DEFINE_CLASS(CustomMouseIteraction);
}