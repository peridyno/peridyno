#include "EdgeInteraction.h"
#include <thrust/sort.h>
#include <iostream>
#include <OrbitCamera.h>

namespace dyno
{
	__global__ void EI_EdgeInitializeArray(
		DArray<int> intersected)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		intersected[pId] = 0;
	}

	__global__ void  EI_EdgeMergeIntersectedIndexOR(
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

	__global__ void  EI_EdgeMergeIntersectedIndexXOR(
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

	__global__ void  EI_EdgeMergeIntersectedIndexC(
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
	EdgeInteraction<TDataType>::EdgeInteraction()
	{
		this->ray1 = TRay3D<Real>();
		this->ray2 = TRay3D<Real>();
		this->isPressed = false;
	}

	template<typename TDataType>
	void EdgeInteraction<TDataType>::onEvent(PMouseEvent event)
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
				if (this->varEdgePickingType()->getValue() == PickingTypeSelection::Both || this->varEdgePickingType()->getValue() == PickingTypeSelection::Click)
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
						if (this->varEdgePickingType()->getValue() == PickingTypeSelection::Both || this->varEdgePickingType()->getValue() == PickingTypeSelection::Click)
							this->calcIntersectClick();
					}
					else
					{
						if (this->varEdgePickingType()->getValue() == PickingTypeSelection::Both || this->varEdgePickingType()->getValue() == PickingTypeSelection::Drag)
							this->calcIntersectDrag();
					}
				}
			}
		}
	}

	template <typename Edge, typename Real, typename Coord>
	__global__ void  EI_CalIntersectedEdgesRay(
		DArray<Coord> points,
		DArray<Edge> edges,
		DArray<int> intersected,
		DArray<int> unintersected,
		DArray<Real> lineDistance,
		TRay3D<Real> mouseray,
		Real radius)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= edges.size()) return;

		TSegment3D<Real> seg = TSegment3D<Real>(points[edges[pId].data[0]], points[edges[pId].data[1]]);

		bool flag = false;
		if (mouseray.distance(seg) <= radius)
		{
			flag = true;
			lineDistance[pId] = abs(TPoint3D<Real>(mouseray.origin[0], mouseray.origin[1], mouseray.origin[2]).distance(seg));
		}
		else 
		{
			flag = false;
			lineDistance[pId] = 3.4E38;
		}

		if (flag || intersected[pId] == 1)
			intersected[pId] = 1;
		else
			intersected[pId] = 0;
		unintersected[pId] = (intersected[pId] == 1 ? 0 : 1);
	}

	__global__ void  EI_CalEdgesNearest(
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

	template <typename Edge, typename Real, typename Coord>
	__global__ void  EI_CalIntersectedEdgesBox(
		DArray<Coord> points,
		DArray<Edge> edges,
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
		if (pId >= edges.size()) return;

		bool flag = false;

		TPoint3D<Real> p;

		TSegment3D<Real> s = TSegment3D<Real>(points[edges[pId].data[0]], points[edges[pId].data[1]]);
		bool temp1 = s.intersect(plane13, p);
		temp1 = temp1 && (((p.origin - plane14.origin).dot(plane14.normal)) * ((p.origin - plane32.origin).dot(plane32.normal))) > 0;
		if (temp1)
			flag = true;
		bool temp2 = s.intersect(plane42, p);
		temp2 = temp2 && (((p.origin - plane14.origin).dot(plane14.normal)) * ((p.origin - plane32.origin).dot(plane32.normal))) > 0;
		if (temp2)
			flag = true;
		bool temp3 = s.intersect(plane14, p);
		temp3 = temp3 && (((p.origin - plane13.origin).dot(plane13.normal)) * ((p.origin - plane42.origin).dot(plane42.normal))) > 0;
		if (temp3)
			flag = true;
		bool temp4 = s.intersect(plane32, p);
		temp4 = temp4 && (((p.origin - plane13.origin).dot(plane13.normal)) * ((p.origin - plane42.origin).dot(plane42.normal))) > 0;
		if (temp4)
			flag = true;

		for (int i = 0; i < 2; i++)
		{
			float temp1 = ((points[edges[pId].data[i]] - plane13.origin).dot(plane13.normal)) * ((points[edges[pId].data[i]] - plane42.origin).dot(plane42.normal));
			float temp2 = ((points[edges[pId].data[i]] - plane14.origin).dot(plane14.normal)) * ((points[edges[pId].data[i]] - plane32.origin).dot(plane32.normal));
			if (temp1 >= 0 && temp2 >= 0)
			{
				flag = flag||true;
				break;
			}
		}

		TSegment3D<Real> seg = TSegment3D<Real>(points[edges[pId].data[0]], points[edges[pId].data[1]]);

		if (mouseray.distance(seg) <= radius)
		{
			flag = flag || true;
		}

		if (flag || intersected[pId] == 1)
			intersected[pId] = 1;
		else
			intersected[pId] = 0;
		unintersected[pId] = (intersected[pId] == 1 ? 0 : 1);
	}

	template <typename Edge>
	__global__ void  EI_AssignOutEdges(
		DArray<Edge> edges,
		DArray<Edge> intersected_edges,
		DArray<Edge> unintersected_edges,
		DArray<int> intersected,
		DArray<int> unintersected,
		DArray<int> intersected_o)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= edges.size()) return;

		if (intersected_o[pId] == 1)
		{
			intersected_edges[intersected[pId]] = edges[pId];
		}
		else
		{
			unintersected_edges[unintersected[pId]] = edges[pId];
		}
	}

	template<typename TDataType>
	void EdgeInteraction<TDataType>::calcEdgeIntersectClick()
	{
		EdgeSet<TDataType> initialEdgeSet = this->inInitialEdgeSet()->getData();
		DArray<Edge> edges = initialEdgeSet.getEdges();
		DArray<Coord> points = initialEdgeSet.getPoints();

		DArray<int> intersected;
		intersected.resize(edges.size());
		cuExecute(edges.size(),
			EI_EdgeInitializeArray,
			intersected
		);
		DArray<int> unintersected;
		unintersected.resize(edges.size());
		std::cout << "Edge Num:" << edges.size() << std::endl;

		DArray<Real> lineDistance;
		lineDistance.resize(edges.size());

		cuExecute(edges.size(),
			EI_CalIntersectedEdgesRay,
			points,
			edges,
			intersected,
			unintersected,
			lineDistance,
			this->ray1,
			this->varInterationRadius()->getData()
		);

		int min_index = thrust::min_element(thrust::device, lineDistance.begin(), lineDistance.begin() + lineDistance.size()) - lineDistance.begin();

		cuExecute(intersected.size(),
			EI_CalEdgesNearest,
			min_index,
			intersected,
			unintersected
		);

		this->tempEdgeIntersectedIndex.assign(intersected);

		if (this->varToggleMultiSelect()->getData())
		{
			if (this->edgeIntersectedIndex.size() == 0)
			{
				this->edgeIntersectedIndex.resize(edges.size());
				cuExecute(edges.size(),
					EI_EdgeInitializeArray,
					this->edgeIntersectedIndex
				)
			}
			DArray<int> outIntersected;
			outIntersected.resize(intersected.size());
			DArray<int> outUnintersected;
			outUnintersected.resize(unintersected.size());
			if (this->varMultiSelectionType()->getValue() == MultiSelectionType::OR)
			{
				cuExecute(edges.size(),
					EI_EdgeMergeIntersectedIndexOR,
					this->edgeIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
			{
				cuExecute(edges.size(),
					EI_EdgeMergeIntersectedIndexXOR,
					this->edgeIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
			{
				cuExecute(edges.size(),
					EI_EdgeMergeIntersectedIndexC,
					this->edgeIntersectedIndex,
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
			this->edgeIntersectedIndex.assign(intersected);
		}

		DArray<int> intersected_o;
		intersected_o.assign(intersected);

		int intersected_size = thrust::reduce(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), intersected.begin());
		DArray<Edge> intersected_edges;
		intersected_edges.resize(intersected_size);

		int unintersected_size = thrust::reduce(thrust::device, unintersected.begin(), unintersected.begin() + unintersected.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, unintersected.begin(), unintersected.begin() + unintersected.size(), unintersected.begin());
		DArray<Edge> unintersected_edges;
		unintersected_edges.resize(unintersected_size);

		cuExecute(edges.size(),
			EI_AssignOutEdges,
			edges,
			intersected_edges,
			unintersected_edges,
			intersected,
			unintersected,
			intersected_o
		);
		std::cout << "Selected Edges Num:" << intersected_edges.size() << std::endl;
		this->outSelectedEdgeSet()->getDataPtr()->copyFrom(initialEdgeSet);
		this->outSelectedEdgeSet()->getDataPtr()->setEdges(intersected_edges);
		this->outOtherEdgeSet()->getDataPtr()->copyFrom(initialEdgeSet);
		this->outOtherEdgeSet()->getDataPtr()->setEdges(unintersected_edges);
		this->outEdgeIndex()->getDataPtr()->assign(intersected_o);
	}

	template<typename TDataType>
	void EdgeInteraction<TDataType>::calcEdgeIntersectDrag()
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
		
		EdgeSet<TDataType> initialEdgeSet = this->inInitialEdgeSet()->getData();
		DArray<Edge> edges = initialEdgeSet.getEdges();
		DArray<Coord> points = initialEdgeSet.getPoints();
		DArray<int> intersected;
		intersected.resize(edges.size());
		cuExecute(edges.size(),
			EI_EdgeInitializeArray,
			intersected
		);
		DArray<int> unintersected;
		unintersected.resize(edges.size());
		std::cout << "Edge Num:" << edges.size() << std::endl;
		cuExecute(edges.size(),
			EI_CalIntersectedEdgesBox,
			points,
			edges,
			intersected,
			unintersected,
			plane13,
			plane42,
			plane14,
			plane32,
			this->varInterationRadius()->getData(),
			this->ray1
		);

		this->tempEdgeIntersectedIndex.assign(intersected);

		if (this->varToggleMultiSelect()->getData())
		{
			if (this->edgeIntersectedIndex.size() == 0)
			{
				this->edgeIntersectedIndex.resize(edges.size());
				cuExecute(edges.size(),
					EI_EdgeInitializeArray,
					this->edgeIntersectedIndex
				)
			}
			DArray<int> outIntersected;
			outIntersected.resize(intersected.size());
			DArray<int> outUnintersected;
			outUnintersected.resize(unintersected.size());
			if (this->varMultiSelectionType()->getValue() == MultiSelectionType::OR)
			{
				cuExecute(edges.size(),
					EI_EdgeMergeIntersectedIndexOR,
					this->edgeIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
			{
				cuExecute(edges.size(),
					EI_EdgeMergeIntersectedIndexXOR,
					this->edgeIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
			{
				cuExecute(edges.size(),
					EI_EdgeMergeIntersectedIndexC,
					this->edgeIntersectedIndex,
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
			this->edgeIntersectedIndex.assign(intersected);
		}

		DArray<int> intersected_o;
		intersected_o.assign(intersected);

		int intersected_size = thrust::reduce(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), intersected.begin());
		DArray<Edge> intersected_edges;
		intersected_edges.resize(intersected_size);

		int unintersected_size = thrust::reduce(thrust::device, unintersected.begin(), unintersected.begin() + unintersected.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, unintersected.begin(), unintersected.begin() + unintersected.size(), unintersected.begin());
		DArray<Edge> unintersected_edges;
		unintersected_edges.resize(unintersected_size);

		cuExecute(edges.size(),
			EI_AssignOutEdges,
			edges,
			intersected_edges,
			unintersected_edges,
			intersected,
			unintersected,
			intersected_o
		);
		std::cout << "Selected Edges Num:" << intersected_edges.size() << std::endl;
		this->outSelectedEdgeSet()->getDataPtr()->copyFrom(initialEdgeSet);
		this->outSelectedEdgeSet()->getDataPtr()->setEdges(intersected_edges);
		this->outOtherEdgeSet()->getDataPtr()->copyFrom(initialEdgeSet);
		this->outOtherEdgeSet()->getDataPtr()->setEdges(unintersected_edges);
		this->outEdgeIndex()->getDataPtr()->assign(intersected_o);
	}

	template<typename TDataType>
	void EdgeInteraction<TDataType>::mergeIndex()
	{
		EdgeSet<TDataType> initialEdgeSet = this->inInitialEdgeSet()->getData();
		DArray<Edge> edges = initialEdgeSet.getEdges();
		DArray<Coord> points = initialEdgeSet.getPoints();
		DArray<int> intersected;
		intersected.resize(edges.size());
		cuExecute(edges.size(),
			EI_EdgeInitializeArray,
			intersected
		);
		DArray<int> unintersected;
		unintersected.resize(edges.size());
		std::cout << "Edge Num:" << edges.size() << std::endl;

		DArray<int> outIntersected;
		outIntersected.resize(intersected.size());
		DArray<int> outUnintersected;
		outUnintersected.resize(unintersected.size());

		if (this->varMultiSelectionType()->getValue() == MultiSelectionType::OR)
		{
			cuExecute(edges.size(),
				EI_EdgeMergeIntersectedIndexOR,
				this->edgeIntersectedIndex,
				this->tempEdgeIntersectedIndex,
				outIntersected,
				outUnintersected
			);
		}
		else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
		{
			cuExecute(edges.size(),
				EI_EdgeMergeIntersectedIndexXOR,
				this->edgeIntersectedIndex,
				this->tempEdgeIntersectedIndex,
				outIntersected,
				outUnintersected
			);
		}
		else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
		{
			cuExecute(edges.size(),
				EI_EdgeMergeIntersectedIndexC,
				this->edgeIntersectedIndex,
				this->tempEdgeIntersectedIndex,
				outIntersected,
				outUnintersected
			);
		}

		intersected.assign(outIntersected);
		unintersected.assign(outUnintersected);
		this->edgeIntersectedIndex.assign(intersected);

		DArray<int> intersected_o;
		intersected_o.assign(intersected);

		int intersected_size = thrust::reduce(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), intersected.begin());
		DArray<Edge> intersected_edges;
		intersected_edges.resize(intersected_size);

		int unintersected_size = thrust::reduce(thrust::device, unintersected.begin(), unintersected.begin() + unintersected.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, unintersected.begin(), unintersected.begin() + unintersected.size(), unintersected.begin());
		DArray<Edge> unintersected_edges;
		unintersected_edges.resize(unintersected_size);

		cuExecute(edges.size(),
			EI_AssignOutEdges,
			edges,
			intersected_edges,
			unintersected_edges,
			intersected,
			unintersected,
			intersected_o
		);
		std::cout << "Selected Edges Num:" << intersected_edges.size() << std::endl;
		this->outSelectedEdgeSet()->getDataPtr()->copyFrom(initialEdgeSet);
		this->outSelectedEdgeSet()->getDataPtr()->setEdges(intersected_edges);
		this->outOtherEdgeSet()->getDataPtr()->copyFrom(initialEdgeSet);
		this->outOtherEdgeSet()->getDataPtr()->setEdges(unintersected_edges);
		this->outEdgeIndex()->getDataPtr()->assign(intersected_o);
	}

	template<typename TDataType>
	void EdgeInteraction<TDataType>::calcIntersectClick()
	{
		if(this->varTogglePicker()->getData())
			calcEdgeIntersectClick();
	}

	template<typename TDataType>
	void EdgeInteraction<TDataType>::calcIntersectDrag()
	{
		if (this->varTogglePicker()->getData())
			calcEdgeIntersectDrag();
	}

	DEFINE_CLASS(EdgeInteraction);
}