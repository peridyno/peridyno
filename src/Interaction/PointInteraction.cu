#include "PointInteraction.h"
#include <thrust/sort.h>
#include <iostream>
#include <OrbitCamera.h>

namespace dyno
{
	__global__ void PI_PointInitializeArray(
		DArray<int> intersected)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		intersected[pId] = 0;
	}

	__global__ void PI_PointMergeIntersectedIndexOR(
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

	__global__ void PI_PointMergeIntersectedIndexXOR(
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

	__global__ void PI_PointMergeIntersectedIndexC(
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
			if (event.shiftKeyPressed() || event.controlKeyPressed())
			{
				this->varToggleMultiSelect()->setValue(true);
				if (event.shiftKeyPressed() && !event.controlKeyPressed())
				{
					this->varMultiSelectionType()->getDataPtr()->setCurrentKey(0);
				}
				else if (!event.shiftKeyPressed() && event.controlKeyPressed())
				{
					this->varMultiSelectionType()->getDataPtr()->setCurrentKey(1);;
				}
				else if (event.shiftKeyPressed() && event.controlKeyPressed())
				{
					this->varMultiSelectionType()->getDataPtr()->setCurrentKey(2);;
				}
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
				{
					this->calcIntersectClick();
					this->printInfoClick();
				}
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
				{
					this->mergeIndex();
					this->printInfoDragRelease();
				}
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
						{
							this->calcIntersectDrag();
							this->printInfoDragging();
						}
					}
				}
			}
		}
	}

	template <typename Real, typename Coord>
	__global__ void PI_CalIntersectedPointsRay(
		DArray<Coord> points,
		DArray<int> intersected,
		DArray<int> unintersected,
		DArray<Real> pointDistance,
		TRay3D<Real> mouseray,
		Real radius)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= points.size()) return;

		TSphere3D<Real> sphere = TSphere3D<Real>(points[pId], radius);
		TSegment3D<Real> seg;
		int temp = mouseray.intersect(sphere, seg);
		if (temp > 0 || intersected[pId] == 1)
		{
			intersected[pId] = 1;
			pointDistance[pId] = abs(TPoint3D<Real>(points[pId][0], points[pId][1], points[pId][2]).distance(TPoint3D<Real>(mouseray.origin[0], mouseray.origin[1], mouseray.origin[2])));
		}
		else
		{
			intersected[pId] = 0;
			pointDistance[pId] = 3.4E38;
		}
		unintersected[pId] = (intersected[pId] == 1 ? 0 : 1);
	}

	__global__ void PI_CalPointsNearest(
		int min_index,
		DArray<int> intersected,
		DArray<int> unintersected
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		if (intersected[pId] == 1)
		{
			if (pId != min_index)
			{
				intersected[pId] = 0;
				unintersected[pId] = 1;
			}
		}
	}


	template <typename Real, typename Coord>
	__global__ void PI_CalIntersectedPointsBox(
		DArray<Coord> points,
		DArray<int> intersected,
		DArray<int> unintersected,
		TPlane3D<Real> plane13,
		TPlane3D<Real> plane42,
		TPlane3D<Real> plane14,
		TPlane3D<Real> plane32,
		Real radius,
		TRay3D<Real> mouseray)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= points.size()) return;

		bool flag = false;
		float temp1 = ((points[pId] - plane13.origin).dot(plane13.normal)) * ((points[pId] - plane42.origin).dot(plane42.normal));
		float temp2 = ((points[pId] - plane14.origin).dot(plane14.normal)) * ((points[pId] - plane32.origin).dot(plane32.normal));
		if (temp1 >= 0 && temp2 >= 0)
			flag = true;

		TSphere3D<Real> sphere = TSphere3D<Real>(points[pId], radius);
		TSegment3D<Real> seg;
		int temp = mouseray.intersect(sphere, seg);
		if (temp > 0)
			flag = flag || true;
		if (flag || intersected[pId] == 1)
			intersected[pId] = 1;
		else
			intersected[pId] = 0;
		unintersected[pId] = (intersected[pId] == 1 ? 0 : 1);
	}

	template <typename Coord>
	__global__ void PI_AssignOutPoints(
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
		PointSet<TDataType> initialPointSet = this->inInitialPointSet()->getData();
		DArray<Coord> points = initialPointSet.getPoints();

		DArray<int> intersected;
		intersected.resize(points.size());
		cuExecute(points.size(),
			PI_PointInitializeArray,
			intersected
		);
		DArray<int> unintersected;
		unintersected.resize(points.size());
		this->tempNumT = points.size();

		DArray<Real> pointDistance;
		pointDistance.resize(points.size());

		cuExecute(points.size(),
			PI_CalIntersectedPointsRay,
			points,
			intersected,
			unintersected,
			pointDistance,
			this->ray1,
			this->varInteractionRadius()->getData()
		);

		int min_index = thrust::min_element(thrust::device, pointDistance.begin(), pointDistance.begin() + pointDistance.size()) - pointDistance.begin();

		cuExecute(intersected.size(),
			PI_CalPointsNearest,
			min_index,
			intersected,
			unintersected
		);

		this->tempPointIntersectedIndex.assign(intersected);

		if (this->varToggleMultiSelect()->getData())
		{
			if (this->pointIntersectedIndex.size() == 0)
			{
				this->pointIntersectedIndex.resize(points.size());
				cuExecute(points.size(),
					PI_PointInitializeArray,
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
					PI_PointMergeIntersectedIndexOR,
					this->pointIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
			{
				cuExecute(points.size(),
					PI_PointMergeIntersectedIndexXOR,
					this->pointIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
			{
				cuExecute(points.size(),
					PI_PointMergeIntersectedIndexC,
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
				PI_AssignOutPoints,
				points,
				intersected_points,
				unintersected_points,
				intersected,
				unintersected,
				intersected_o
			);
			this->tempNumS = intersected_size;
			this->outSelectedPointSet()->getDataPtr()->copyFrom(initialPointSet);
			this->outSelectedPointSet()->getDataPtr()->setPoints(intersected_points);
			this->outOtherPointSet()->getDataPtr()->copyFrom(initialPointSet);
			this->outOtherPointSet()->getDataPtr()->setPoints(unintersected_points);
			this->outPointIndex()->getDataPtr()->assign(intersected_o);
		}

		template<typename TDataType>
		void PointInteraction<TDataType>::calcPointIntersectDrag()
		{
			if (x1 == x2)
			{
				x2 += 1.0f;
			}
			if (y1 == y2)
			{
				y2 += 1.0f;
			}
			TRay3D<Real> ray1 = this->camera->castRayInWorldSpace((float)x1, (float)y1);
			TRay3D<Real> ray2 = this->camera->castRayInWorldSpace((float)x2, (float)y2);
			TRay3D<Real> ray3 = this->camera->castRayInWorldSpace((float)x1, (float)y2);
			TRay3D<Real> ray4 = this->camera->castRayInWorldSpace((float)x2, (float)y1);

			TPlane3D<Real> plane13 = TPlane3D<Real>(ray1.origin, ray1.direction.cross(ray3.direction));
			TPlane3D<Real> plane42 = TPlane3D<Real>(ray2.origin, ray2.direction.cross(ray4.direction));
			TPlane3D<Real> plane14 = TPlane3D<Real>(ray4.origin, ray1.direction.cross(ray4.direction));
			TPlane3D<Real> plane32 = TPlane3D<Real>(ray3.origin, ray2.direction.cross(ray3.direction));

			PointSet<TDataType> initialPointSet = this->inInitialPointSet()->getData();
			DArray<Coord> points = initialPointSet.getPoints();
			DArray<int> intersected;
			intersected.resize(points.size());
			cuExecute(points.size(),
				PI_PointInitializeArray,
				intersected
			);
			DArray<int> unintersected;
			unintersected.resize(points.size());
			this->tempNumT = points.size();
			cuExecute(points.size(),
				PI_CalIntersectedPointsBox,
				points,
				intersected,
				unintersected,
				plane13,
				plane42,
				plane14,
				plane32,
				this->varInteractionRadius()->getData(),
				this->ray1
			);

			this->tempPointIntersectedIndex.assign(intersected);

			if (this->varToggleMultiSelect()->getData())
			{
				if (this->pointIntersectedIndex.size() == 0)
				{
					this->pointIntersectedIndex.resize(points.size());
					cuExecute(points.size(),
						PI_PointInitializeArray,
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
						PI_PointMergeIntersectedIndexOR,
						this->pointIntersectedIndex,
						intersected,
						outIntersected,
						outUnintersected
					);
				}
				else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
				{
					cuExecute(points.size(),
						PI_PointMergeIntersectedIndexXOR,
						this->pointIntersectedIndex,
						intersected,
						outIntersected,
						outUnintersected
					);
				}
				else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
				{
					cuExecute(points.size(),
						PI_PointMergeIntersectedIndexC,
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
				PI_AssignOutPoints,
				points,
				intersected_points,
				unintersected_points,
				intersected,
				unintersected,
				intersected_o
			);
			this->tempNumS = intersected_size;
			this->outSelectedPointSet()->getDataPtr()->copyFrom(initialPointSet);
			this->outSelectedPointSet()->getDataPtr()->setPoints(intersected_points);
			this->outOtherPointSet()->getDataPtr()->copyFrom(initialPointSet);
			this->outOtherPointSet()->getDataPtr()->setPoints(unintersected_points);
			this->outPointIndex()->getDataPtr()->assign(intersected_o);
		}

		template<typename TDataType>
		void PointInteraction<TDataType>::mergeIndex()
		{
			PointSet<TDataType> initialPointSet = this->inInitialPointSet()->getData();
			DArray<Coord> points = initialPointSet.getPoints();
			DArray<int> intersected;
			intersected.resize(points.size());
			cuExecute(points.size(),
				PI_PointInitializeArray,
				intersected
			);
			DArray<int> unintersected;
			unintersected.resize(points.size());
			this->tempNumT = points.size();

			DArray<int> outIntersected;
			outIntersected.resize(intersected.size());
			DArray<int> outUnintersected;
			outUnintersected.resize(unintersected.size());

			if (this->varMultiSelectionType()->getValue() == MultiSelectionType::OR)
			{
				cuExecute(points.size(),
					PI_PointMergeIntersectedIndexOR,
					this->pointIntersectedIndex,
					this->tempPointIntersectedIndex,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
			{
				cuExecute(points.size(),
					PI_PointMergeIntersectedIndexXOR,
					this->pointIntersectedIndex,
					this->tempPointIntersectedIndex,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
			{
				cuExecute(points.size(),
					PI_PointMergeIntersectedIndexC,
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
				PI_AssignOutPoints,
				points,
				intersected_points,
				unintersected_points,
				intersected,
				unintersected,
				intersected_o
			);
			this->tempNumS = intersected_size;
			this->outSelectedPointSet()->getDataPtr()->copyFrom(initialPointSet);
			this->outSelectedPointSet()->getDataPtr()->setPoints(intersected_points);
			this->outOtherPointSet()->getDataPtr()->copyFrom(initialPointSet);
			this->outOtherPointSet()->getDataPtr()->setPoints(unintersected_points);
			this->outPointIndex()->getDataPtr()->assign(intersected_o);
		}

		template<typename TDataType>
		void PointInteraction<TDataType>::printInfoClick()
		{
			std::cout << "----------point picking: click----------" << std::endl;
			std::cout << "multiple picking: " << this->varToggleMultiSelect()->getValue() << std::endl;
			std::cout << "Interation radius:" << this->varInteractionRadius()->getValue() << std::endl;
			std::cout << "selected num/ total num:" << this->tempNumS << "/" << this->tempNumT << std::endl;
		}

		template<typename TDataType>
		void PointInteraction<TDataType>::printInfoDragging()
		{
			std::cout << "----------point picking: dragging----------" << std::endl;
			std::cout << "multiple picking: " << this->varToggleMultiSelect()->getValue() << std::endl;
			std::cout << "Interation radius:" << this->varInteractionRadius()->getValue() << std::endl;
			std::cout << "selected num/ total num:" << this->tempNumS << "/" << this->tempNumT << std::endl;
		}

		template<typename TDataType>
		void PointInteraction<TDataType>::printInfoDragRelease()
		{
			std::cout << "----------point picking: drag release----------" << std::endl;
			std::cout << "multiple picking: " << this->varToggleMultiSelect()->getValue() << std::endl;
			std::cout << "Interation radius:" << this->varInteractionRadius()->getValue() << std::endl;
			std::cout << "selected num/ total num:" << this->tempNumS << "/" << this->tempNumT << std::endl;
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