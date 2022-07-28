#include "SurfaceInteraction.h"
#include <thrust/sort.h>
#include <iostream>
#include <OrbitCamera.h>

namespace dyno
{
	__global__ void SurfaceInitializeArray(
		DArray<int> intersected)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		intersected[pId] = 0;
	}

	__global__ void SurfaceMergeIntersectedIndex(
		DArray<int> intersected1,
		DArray<int> intersected2,
		DArray<int> outIntersected,
		DArray<int> outUnintersected)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected1.size()) return;

		if (intersected1[pId] == 0 && intersected2[pId] == 0)
			outIntersected[pId] = 0;
		else
			outIntersected[pId] = 1;

		outUnintersected[pId] = outIntersected[pId] == 1 ? 0 : 1;
	}

	template<typename TDataType>
	SurfaceInteraction<TDataType>::SurfaceInteraction()
	{
		this->ray1 = TRay3D<Real>();
		this->ray2 = TRay3D<Real>();
		this->isPressed = false;
	}

	template<typename TDataType>
	void SurfaceInteraction<TDataType>::onEvent(PMouseEvent event)
	{
		if (!event.altKeyPressed()) {
			if (camera == nullptr)
			{
				this->camera = event.camera;
			}
			this->varToggleMultiSelect()->setValue(false);
			if (event.controlKeyPressed()) 
			{
				this->varToggleMultiSelect()->setValue(true);
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

		TTriangle3D<Real> t = TTriangle3D<Real>(points[triangles[pId].data[0]], points[triangles[pId].data[1]], points[triangles[pId].data[2]]);
		int temp = 0;
		if (mouseray.direction.dot(t.normal()) < 0)
		{
			TPoint3D<Real> p;
			temp = mouseray.intersect(t, p);
		}
		if (temp == 1 || intersected[pId] == 1)
			intersected[pId] = 1;
		else
			intersected[pId] = 0;
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
		TPlane3D<Real> plane32,
		TRay3D<Real> mouseray)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= triangles.size()) return;


		TTriangle3D<Real> t = TTriangle3D<Real>(points[triangles[pId].data[0]], points[triangles[pId].data[1]], points[triangles[pId].data[2]]);
		bool flag = false;
		if (mouseray.direction.dot(t.normal()) < 0)
		{
			for (int i = 0; i < 3; i++) {
				float temp1 = ((points[triangles[pId].data[i]] - plane13.origin).dot(plane13.normal)) * ((points[triangles[pId].data[i]] - plane42.origin).dot(plane42.normal));
				float temp2 = ((points[triangles[pId].data[i]] - plane14.origin).dot(plane14.normal)) * ((points[triangles[pId].data[i]] - plane32.origin).dot(plane32.normal));
				if (temp1 >= 0 && temp2 >= 0)
				{
					flag = true;
					break;
				}
			}
		}
		if (flag || intersected[pId] == 1)
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
	void SurfaceInteraction<TDataType>::calcSurfaceIntersectClick()
	{
		TriangleSet<TDataType> initialTriangleSet = this->inInitialTriangleSet()->getData();
		DArray<Coord> points = initialTriangleSet.getPoints();
		DArray<Triangle> triangles = initialTriangleSet.getTriangles();
		DArray<int> intersected;
		intersected.resize(triangles.size());
		cuExecute(triangles.size(),
			SurfaceInitializeArray,
			intersected
		);
		DArray<int> unintersected;
		unintersected.resize(triangles.size());
		std::cout << "Triangle Num:" << triangles.size() << std::endl;
		cuExecute(triangles.size(),
			CalIntersectedTrisRay,
			points,
			triangles,
			intersected,
			unintersected,
			this->ray1
		);

		if (this->varToggleMultiSelect()->getData())
		{
			if (this->triIntersectedIndex.size() == 0) 
			{
				this->triIntersectedIndex.resize(triangles.size());
				cuExecute(triangles.size(),
					SurfaceInitializeArray,
					this->triIntersectedIndex
				)
			}
			DArray<int> outIntersected;
			outIntersected.resize(intersected.size());
			DArray<int> outUnintersected;
			outUnintersected.resize(unintersected.size());
			cuExecute(triangles.size(),
				SurfaceMergeIntersectedIndex,
				this->triIntersectedIndex,
				intersected,
				outIntersected,
				outUnintersected
			);
			intersected.assign(outIntersected);
			unintersected.assign(outUnintersected);
			this->triIntersectedIndex.assign(intersected);
		}
		else
		{
			this->triIntersectedIndex.assign(intersected);
		}

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
		std::cout << "Selected Triangles Num:" << intersected_triangles.size() << std::endl;
		this->outSelectedTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedTriangleSet()->getDataPtr()->setTriangles(intersected_triangles);
		this->outOtherTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outOtherTriangleSet()->getDataPtr()->setTriangles(unintersected_triangles);
		this->outTriangleIndex()->getDataPtr()->assign(intersected_o);
	}

	template<typename TDataType>
	void SurfaceInteraction<TDataType>::calcSurfaceIntersectDrag()
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
		DArray<Triangle> triangles = initialTriangleSet.getTriangles();
		DArray<int> intersected;
		intersected.resize(triangles.size());
		cuExecute(triangles.size(),
			SurfaceInitializeArray,
			intersected
		);
		DArray<int> unintersected;
		unintersected.resize(triangles.size());
		std::cout << "Triangle Num:" << triangles.size() << std::endl;
		cuExecute(triangles.size(),
			CalIntersectedTrisBox,
			points,
			triangles,
			intersected,
			unintersected,
			plane13,
			plane42,
			plane14,
			plane32,
			this->ray2
		);
		cuExecute(triangles.size(),
			CalIntersectedTrisRay,
			points,
			triangles,
			intersected,
			unintersected,
			this->ray1
		);

		if (this->varToggleMultiSelect()->getData())
		{
			if (this->triIntersectedIndex.size() == 0)
			{
				this->triIntersectedIndex.resize(triangles.size());
				cuExecute(triangles.size(),
					SurfaceInitializeArray,
					this->triIntersectedIndex
				)
			}
			DArray<int> outIntersected;
			outIntersected.resize(intersected.size());
			DArray<int> outUnintersected;
			outUnintersected.resize(unintersected.size());
			cuExecute(triangles.size(),
				SurfaceMergeIntersectedIndex,
				this->triIntersectedIndex,
				intersected,
				outIntersected,
				outUnintersected
			);
			intersected.assign(outIntersected);
			unintersected.assign(outUnintersected);
			this->triIntersectedIndex.assign(intersected);
		}
		else
		{
			this->triIntersectedIndex.assign(intersected);
		}


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
		std::cout << "Selected Triangles Num:" << intersected_triangles.size() << std::endl;
		this->outSelectedTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedTriangleSet()->getDataPtr()->setTriangles(intersected_triangles);
		this->outOtherTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outOtherTriangleSet()->getDataPtr()->setTriangles(unintersected_triangles);
		this->outTriangleIndex()->getDataPtr()->assign(intersected_o);
	}

	template<typename TDataType>
	void SurfaceInteraction<TDataType>::calcIntersectClick()
	{
		if (this->varTogglePicker()->getData())
			calcSurfaceIntersectClick();
	}

	template<typename TDataType>
	void SurfaceInteraction<TDataType>::calcIntersectDrag()
	{
		if (this->varTogglePicker()->getData())
			calcSurfaceIntersectDrag();
	}

	DEFINE_CLASS(SurfaceInteraction);
}