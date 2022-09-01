#include "PointInteraction.h"
#include <thrust/sort.h>
#include <iostream>
#include <OrbitCamera.h>

namespace dyno
{
	__global__ void PointInitializeArray(
		DArray<int> intersected)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		intersected[pId] = 0;
	}

	__global__ void SurfaceInitializeArrayP(
		DArray<int> intersected)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		intersected[pId] = 0;
	}

	template <typename Triangle, typename Real, typename Coord>
	__global__ void CalIntersectedTrisRayP(
		DArray<Coord> points,
		DArray<Triangle> triangles,
		DArray<int> intersected,
		DArray<int> unintersected,
		DArray<Coord> interPoints,
		TRay3D<Real> mouseray)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= triangles.size()) return;

		TTriangle3D<Real> t = TTriangle3D<Real>(points[triangles[pId].data[0]], points[triangles[pId].data[1]], points[triangles[pId].data[2]]);
		int temp = 0;

		TPoint3D<Real> p;
		temp = mouseray.intersect(t, p);

		if (temp == 1 || intersected[pId] == 1)
		{
			intersected[pId] = 1;
			interPoints[pId] = p.origin;
		}
		else
		{
			intersected[pId] = 0;
			interPoints[pId] = Vec3f(0);
		}
		unintersected[pId] = (intersected[pId] == 1 ? 0 : 1);
	}

	template <typename Real, typename Coord>
	__global__ void CalTrisDistanceP(
		DArray<Coord> interPoints,
		DArray<Real> trisDistance,
		DArray<int> intersected,
		TRay3D<Real> mouseray)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= interPoints.size()) return;
		if (intersected[pId] != 0)
		{
			TPoint3D<Real> origin = TPoint3D<Real>(mouseray.origin);
			TPoint3D<Real> p = TPoint3D<Real>(interPoints[pId]);
			trisDistance[pId] = origin.distance(TPoint3D<Real>(p));
		}
		else
		{
			trisDistance[pId] = 3.4E38;
		}
	}

	template <typename Coord, typename Real>
	__global__ void FindNearbyPoints(
		DArray<Coord> points,
		DArray<Coord> interPoints,
		DArray<int> intersected,
		DArray<int> unintersected,
		int min_index_t,
		Real intersectionRadius)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		if (intersected[pId] == 1)
		{
			TPoint3D<Real> p1 = TPoint3D<Real>(points[pId]);
			TPoint3D<Real> p2 = TPoint3D<Real>(interPoints[min_index_t]);
			if (p1.distance(p2) > intersectionRadius)
			{
				intersected[pId] = 0;
				unintersected[pId] = 1;
			}
		}
	}

	__global__ void PointMergeIntersectedIndexOR(
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

	__global__ void PointMergeIntersectedIndexXOR(
		DArray<int> intersected1,
		DArray<int> intersected2,
		DArray<int> outIntersected,
		DArray<int> outUnintersected)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected1.size()) return;

		if (intersected1[pId] == intersected2[pId])
			outIntersected[pId] = 0;
		else
			outIntersected[pId] = 1;

		outUnintersected[pId] = outIntersected[pId] == 1 ? 0 : 1;
	}

	__global__ void PointMergeIntersectedIndexC(
		DArray<int> intersected1,
		DArray<int> intersected2,
		DArray<int> outIntersected,
		DArray<int> outUnintersected)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected1.size()) return;

		if (intersected2[pId] == 1)
			outIntersected[pId] = 0;
		else
			outIntersected[pId] = intersected1[pId];

		outUnintersected[pId] = outIntersected[pId] == 1 ? 0 : 1;
	}

	template<typename TDataType>
	PointInteraction<TDataType>::PointInteraction()
	{
		this->ray1 = TRay3D<Real>();
		this->ray2 = TRay3D<Real>();
		this->isPressed = false;
	}

	template<typename TDataType>
	void PointInteraction<TDataType>::onEvent(PMouseEvent event)
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
				//printf("Mouse pressed: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
				this->ray1.origin = event.ray.origin;
				this->ray1.direction = event.ray.direction;
				this->x1 = event.x;
				this->y1 = event.y;
				if (this->varPointPickingType()->getValue() == PickingTypeSelection::Both || this->varPointPickingType()->getValue() == PickingTypeSelection::Click)
					this->calcIntersectClick();
			}
			else if (event.actionType == AT_RELEASE)
			{
				this->isPressed = false;
				//printf("Mouse released: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
				this->ray2.origin = event.ray.origin;
				this->ray2.direction = event.ray.direction;
				this->x2 = event.x;
				this->y2 = event.y;
				if (this->varToggleMultiSelect()->getValue() && this->varTogglePicker()->getValue())
					this->mergeIndex();
			}
			else
			{
				//printf("Mouse repeated: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
				if (this->isPressed) 
				{
					this->ray2.origin = event.ray.origin;
					this->ray2.direction = event.ray.direction;
					this->x2 = event.x;
					this->y2 = event.y;
					if (this->x2 == this->x1 && this->y2 == this->y1)
					{
						if (this->varPointPickingType()->getValue() == PickingTypeSelection::Both || this->varPointPickingType()->getValue() == PickingTypeSelection::Click)
							this->calcIntersectClick();
					}
					else 
					{
						if (this->varPointPickingType()->getValue() == PickingTypeSelection::Both || this->varPointPickingType()->getValue() == PickingTypeSelection::Drag)
							this->calcIntersectDrag();
					}
				}
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
		if (temp > 0 || intersected[pId] == 1)
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
		float temp2 = ((points[pId] - plane14.origin).dot(plane14.normal)) * ((points[pId] - plane32.origin).dot(plane32.normal));
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
	void PointInteraction<TDataType>::calcPointIntersectClick()
	{
		TriangleSet<TDataType> initialTriangleSet = this->inInitialTriangleSet()->getData();
		DArray<Coord> points = initialTriangleSet.getPoints();

		DArray<Triangle> triangles = initialTriangleSet.getTriangles();
		DArray<int> intersected_t;
		intersected_t.resize(triangles.size());

		cuExecute(triangles.size(),
			SurfaceInitializeArrayP,
			intersected_t
		);
		DArray<int> unintersected_t;
		unintersected_t.resize(triangles.size());
		//std::cout << "Triangle Num:" << triangles.size() << std::endl;

		DArray<Coord> interPoints;
		interPoints.resize(triangles.size());

		cuExecute(triangles.size(),
			CalIntersectedTrisRayP,
			points,
			triangles,
			intersected_t,
			unintersected_t,
			interPoints,
			this->ray1
		);


		DArray<Real> trisDistance;
		trisDistance.resize(interPoints.size());

		cuExecute(interPoints.size(),
			CalTrisDistanceP,
			interPoints,
			trisDistance,
			intersected_t,
			this->ray1
		);

		int min_index_t = thrust::min_element(thrust::device, trisDistance.begin(), trisDistance.begin() + trisDistance.size()) - trisDistance.begin();

		DArray<int> intersected;
		intersected.resize(points.size());
		cuExecute(points.size(),
			PointInitializeArray,
			intersected
		);
		DArray<int> unintersected;
		unintersected.resize(points.size());
		std::cout << "Point Num:" << points.size() << std::endl;
		cuExecute(points.size(),
			CalIntersectedPointsRay,
			points,
			intersected,
			unintersected,
			this->ray1,
			this->varInterationRadius()->getData()
		);

		cuExecute(points.size(),
			FindNearbyPoints,
			points,
			interPoints,
			intersected,
			unintersected,
			min_index_t,
			this->varInterationRadius()->getData()
		);

		this->tempPointIntersectedIndex.assign(intersected);

		if (this->varToggleMultiSelect()->getData())
		{
			if (this->pointIntersectedIndex.size() == 0)
			{
				this->pointIntersectedIndex.resize(triangles.size());
				cuExecute(points.size(),
					PointInitializeArray,
					this->pointIntersectedIndex
				)
			}
			DArray<int> outIntersected;
			outIntersected.resize(intersected.size());
			DArray<int> outUnintersected;
			outUnintersected.resize(unintersected.size());
			if (this->varMultiSelectionType()->getValue() == MultiSelectionType::OR)
			{
				cuExecute(points.size(),
					PointMergeIntersectedIndexOR,
					this->pointIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
			{
				cuExecute(points.size(),
					PointMergeIntersectedIndexXOR,
					this->pointIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
			{
				cuExecute(points.size(),
					PointMergeIntersectedIndexC,
					this->pointIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			intersected.assign(outIntersected);
			unintersected.assign(outUnintersected);
		}
		else
		{
			this->pointIntersectedIndex.assign(intersected);
		}
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
			std::cout << "Selected Points Num:" << intersected_points.size() << std::endl;
			this->outSelectedPointSet()->getDataPtr()->copyFrom(initialTriangleSet);
			this->outSelectedPointSet()->getDataPtr()->setPoints(intersected_points);
			this->outOtherPointSet()->getDataPtr()->copyFrom(initialTriangleSet);
			this->outOtherPointSet()->getDataPtr()->setPoints(unintersected_points);
			this->outPointIndex()->getDataPtr()->assign(intersected_o);
		}

		template<typename TDataType>
		void PointInteraction<TDataType>::calcPointIntersectDrag()
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
			cuExecute(points.size(),
				PointInitializeArray,
				intersected
			);
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
				this->varInterationRadius()->getData()
			);

			this->tempPointIntersectedIndex.assign(intersected);

			if (this->varToggleMultiSelect()->getData())
			{
				if (this->pointIntersectedIndex.size() == 0)
				{
					this->pointIntersectedIndex.resize(points.size());
					cuExecute(points.size(),
						PointInitializeArray,
						this->pointIntersectedIndex
					)
				}
				DArray<int> outIntersected;
				outIntersected.resize(intersected.size());
				DArray<int> outUnintersected;
				outUnintersected.resize(unintersected.size());
				if (this->varMultiSelectionType()->getValue() == MultiSelectionType::OR)
				{
					cuExecute(points.size(),
						PointMergeIntersectedIndexOR,
						this->pointIntersectedIndex,
						intersected,
						outIntersected,
						outUnintersected
					);
				}
				else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
				{
					cuExecute(points.size(),
						PointMergeIntersectedIndexXOR,
						this->pointIntersectedIndex,
						intersected,
						outIntersected,
						outUnintersected
					);
				}
				else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
				{
					cuExecute(points.size(),
						PointMergeIntersectedIndexC,
						this->pointIntersectedIndex,
						intersected,
						outIntersected,
						outUnintersected
					);
				}
				intersected.assign(outIntersected);
				unintersected.assign(outUnintersected);
			}
			else
			{
				this->pointIntersectedIndex.assign(intersected);
			}

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
			std::cout << "Selected Points Num:" << intersected_points.size() << std::endl;
			this->outSelectedPointSet()->getDataPtr()->copyFrom(initialTriangleSet);
			this->outSelectedPointSet()->getDataPtr()->setPoints(intersected_points);
			this->outOtherPointSet()->getDataPtr()->copyFrom(initialTriangleSet);
			this->outOtherPointSet()->getDataPtr()->setPoints(unintersected_points);
			this->outPointIndex()->getDataPtr()->assign(intersected_o);
		}

		template<typename TDataType>
		void PointInteraction<TDataType>::mergeIndex()
		{
			TriangleSet<TDataType> initialTriangleSet = this->inInitialTriangleSet()->getData();
			DArray<Coord> points = initialTriangleSet.getPoints();
			DArray<int> intersected;
			intersected.resize(points.size());
			cuExecute(points.size(),
				PointInitializeArray,
				intersected
			);
			DArray<int> unintersected;
			unintersected.resize(points.size());
			std::cout << "Points Num:" << points.size() << std::endl;

			DArray<int> outIntersected;
			outIntersected.resize(intersected.size());
			DArray<int> outUnintersected;
			outUnintersected.resize(unintersected.size());

			if (this->varMultiSelectionType()->getValue() == MultiSelectionType::OR)
			{
				cuExecute(points.size(),
					PointMergeIntersectedIndexOR,
					this->pointIntersectedIndex,
					this->tempPointIntersectedIndex,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
			{
				cuExecute(points.size(),
					PointMergeIntersectedIndexXOR,
					this->pointIntersectedIndex,
					this->tempPointIntersectedIndex,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
			{
				cuExecute(points.size(),
					PointMergeIntersectedIndexC,
					this->pointIntersectedIndex,
					this->tempPointIntersectedIndex,
					outIntersected,
					outUnintersected
				);
			}

			intersected.assign(outIntersected);
			unintersected.assign(outUnintersected);
			this->pointIntersectedIndex.assign(intersected);

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
			std::cout << "Selected Points Num:" << intersected_points.size() << std::endl;
			this->outSelectedPointSet()->getDataPtr()->copyFrom(initialTriangleSet);
			this->outSelectedPointSet()->getDataPtr()->setPoints(intersected_points);
			this->outOtherPointSet()->getDataPtr()->copyFrom(initialTriangleSet);
			this->outOtherPointSet()->getDataPtr()->setPoints(unintersected_points);
			this->outPointIndex()->getDataPtr()->assign(intersected_o);
		}

		template<typename TDataType>
		void PointInteraction<TDataType>::calcIntersectClick()
		{
			if (this->varTogglePicker()->getData())
				calcPointIntersectClick();
		}

		template<typename TDataType>
		void PointInteraction<TDataType>::calcIntersectDrag()
		{
			if (this->varTogglePicker()->getData())
				calcPointIntersectDrag();
		}

		DEFINE_CLASS(PointInteraction);
}