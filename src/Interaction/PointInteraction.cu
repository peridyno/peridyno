#include "PointInteraction.h"
#include <thrust/sort.h>
#include <iostream>
#include <OrbitCamera.h>

namespace dyno
{
	template<typename TDataType>
	PointIteraction<TDataType>::PointIteraction()
	{
		this->ray1 = TRay3D<Real>();
		this->ray2 = TRay3D<Real>();
		this->isPressed = false;
		this->r = 0.05f;
	}

	template<typename TDataType>
	void PointIteraction<TDataType>::onEvent(PMouseEvent event)
	{
		if (camera == nullptr)
		{
			this->camera = event.camera;
		}
		if (event.actionType == AT_PRESS)
		{
			this->camera = event.camera;
			this->isPressed = true;
			printf("Mouse pressed: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
			this->ray1.origin = event.ray.origin;
			this->ray1.direction = event.ray.direction;
			this->x1 = event.x;
			this->y1 = event.y;
			this->calcIntersectClick();
		}
		else if (event.actionType == AT_RELEASE)
		{
			this->isPressed = false;
			printf("Mouse released: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
			this->ray2.origin = event.ray.origin;
			this->ray2.direction = event.ray.direction;
			this->x2 = event.x;
			this->y2 = event.y;
		}
		else
		{
			printf("%f %f \n", event.x, event.y);
			printf("Mouse repeated: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
			if (this->isPressed) {
				this->ray2.origin = event.ray.origin;
				this->ray2.direction = event.ray.direction;
				this->x2 = event.x;
				this->y2 = event.y;
				this->calcIntersectDrag();
			}
		}
	}

	template <typename Real, typename Coord>
	__global__ void CalIntersectedPointsRay(
		DArray<Coord> points,
		DArray<int> intersected,
		DArray<int> unintersected,
		TRay3D<Real> mouseray,
		Real radius)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= points.size()) return;

		TSphere3D<Real> sphere = TSphere3D<Real>(points[pId], radius);
		TSegment3D<Real> seg;
		int temp = mouseray.intersect(sphere, seg);
		if (temp >0 || intersected[pId] == 1)
			intersected[pId] = 1;
		else
			intersected[pId] = 0;
		unintersected[pId] = (intersected[pId] == 1 ? 0 : 1);
	}

	template <typename Real, typename Coord>
	__global__ void CalIntersectedPointsBox(
		DArray<Coord> points,
		DArray<int> intersected,
		DArray<int> unintersected,
		TPlane3D<Real> plane13,
		TPlane3D<Real> plane42,
		TPlane3D<Real> plane14,
		TPlane3D<Real> plane32)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= points.size()) return;

		bool flag = false;
		float temp1 = ((points[pId] - plane13.origin).dot(plane13.normal)) * ((points[pId] - plane42.origin).dot(plane42.normal));
		float temp2 = ((points[pId]- plane14.origin).dot(plane14.normal)) * ((points[pId] - plane32.origin).dot(plane32.normal));
		if (temp1 >= 0 && temp2 >= 0)
			flag = true;

		if (flag || intersected[pId] == 1)
			intersected[pId] = 1;
		else
			intersected[pId] = 0;
		unintersected[pId] = (intersected[pId] == 1 ? 0 : 1);
	}

	template <typename Coord>
	__global__ void AssignOutPoints(
		DArray<Coord> points,
		DArray<Coord> intersected_points,
		DArray<Coord> unintersected_points,
		DArray<int> intersected,
		DArray<int> unintersected,
		DArray<int> intersected_o)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= points.size()) return;

		if (intersected_o[pId] == 1)
		{
			intersected_points[intersected[pId]] = points[pId];
		}
		else
		{
			unintersected_points[unintersected[pId]] = points[pId];

		}
	}

	template<typename TDataType>
	void PointIteraction<TDataType>::calcIntersectClick()
	{
		TriangleSet<TDataType> initialTriangleSet = this->inInitialTriangleSet()->getData();
		DArray<Coord> points = initialTriangleSet.getPoints();
		DArray<int> intersected;
		intersected.resize(points.size());
		DArray<int> unintersected;
		unintersected.resize(points.size());
		std::cout << "Point Num:" << points.size() << std::endl;
		cuExecute(points.size(),
			CalIntersectedPointsRay,
			points,
			intersected,
			unintersected,
			this->ray1,
			this->r
		);

		DArray<int> intersected_o;
		intersected_o.assign(intersected);

		int intersected_size = thrust::reduce(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), intersected.begin());
		DArray<Coord> intersected_points;
		intersected_points.resize(intersected_size);

		int unintersected_size = thrust::reduce(thrust::device, unintersected.begin(), unintersected.begin() + unintersected.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, unintersected.begin(), unintersected.begin() + unintersected.size(), unintersected.begin());
		DArray<Coord> unintersected_points;
		unintersected_points.resize(unintersected_size);

		cuExecute(points.size(),
			AssignOutPoints,
			points,
			intersected_points,
			unintersected_points,
			intersected,
			unintersected,
			intersected_o
		);
		this->outSelectedPointSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedPointSet()->getDataPtr()->setPoints(intersected_points);
		this->outOtherPointSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outOtherPointSet()->getDataPtr()->setPoints(unintersected_points);
	}

	template<typename TDataType>
	void PointIteraction<TDataType>::calcIntersectDrag()
	{
		TRay3D<Real> ray1 = this->ray1;
		TRay3D<Real> ray2 = this->ray2;
		TRay3D<Real> ray3 = this->camera->castRayInWorldSpace((float)x1, (float)y2);
		TRay3D<Real> ray4 = this->camera->castRayInWorldSpace((float)x2, (float)y1);

		TPlane3D<Real> plane13 = TPlane3D<Real>(ray1.origin, ray1.direction.cross(ray3.direction));
		TPlane3D<Real> plane42 = TPlane3D<Real>(ray2.origin, ray2.direction.cross(ray4.direction));
		TPlane3D<Real> plane14 = TPlane3D<Real>(ray4.origin, ray1.direction.cross(ray4.direction));
		TPlane3D<Real> plane32 = TPlane3D<Real>(ray3.origin, ray2.direction.cross(ray3.direction));

		TriangleSet<TDataType> initialTriangleSet = this->inInitialTriangleSet()->getData();
		DArray<Coord> points = initialTriangleSet.getPoints();
		DArray<int> intersected;
		intersected.resize(points.size());
		DArray<int> unintersected;
		unintersected.resize(points.size());
		std::cout << "Point Num:" << points.size() << std::endl;
		cuExecute(points.size(),
			CalIntersectedPointsBox,
			points,
			intersected,
			unintersected,
			plane13,
			plane42,
			plane14,
			plane32
		);
		cuExecute(points.size(),
			CalIntersectedPointsRay,
			points,
			intersected,
			unintersected,
			this->ray1,
			this->r
		);

		DArray<int> intersected_o;
		intersected_o.assign(intersected);

		int intersected_size = thrust::reduce(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), intersected.begin());
		DArray<Coord> intersected_points;
		intersected_points.resize(intersected_size);

		int unintersected_size = thrust::reduce(thrust::device, unintersected.begin(), unintersected.begin() + unintersected.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, unintersected.begin(), unintersected.begin() + unintersected.size(), unintersected.begin());
		DArray<Coord> unintersected_points;
		unintersected_points.resize(unintersected_size);

		cuExecute(points.size(),
			AssignOutPoints,
			points,
			intersected_points,
			unintersected_points,
			intersected,
			unintersected,
			intersected_o
		);
		this->outSelectedPointSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedPointSet()->getDataPtr()->setPoints(intersected_points);
		this->outOtherPointSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outOtherPointSet()->getDataPtr()->setPoints(unintersected_points);
	}
	DEFINE_CLASS(PointIteraction);
}