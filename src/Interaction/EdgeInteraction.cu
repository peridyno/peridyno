#include "EdgeInteraction.h"
#include <thrust/sort.h>
#include <iostream>
#include <OrbitCamera.h>

namespace dyno
{
	__global__ void EdgeInitializeArray(
		DArray<int> intersected)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		intersected[pId] = 0;
	}

	__global__ void EdgeMergeIntersectedIndexOR(
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

	__global__ void EdgeMergeIntersectedIndexXOR(
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

	__global__ void EdgeMergeIntersectedIndexC(
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

	__global__ void SurfaceInitializeArrayE(
		DArray<int> intersected)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		intersected[pId] = 0;
	}

	template <typename Triangle, typename Real, typename Coord>
	__global__ void CalIntersectedTrisRayE(
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
	__global__ void CalTrisDistanceE(
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
	__global__ void CalIntersectedEdgesRay(
		DArray<Coord> points,
		DArray<Edge> edges,
		DArray<int> intersected,
		DArray<int> unintersected,
		TRay3D<Real> mouseray,
		Real radius)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= edges.size()) return;

		TSegment3D<Real> seg = TSegment3D<Real>(points[edges[pId].data[0]], points[edges[pId].data[1]]);

		int temp = 0;
		if (mouseray.distance(seg) <= radius)
			temp = 1;

		if (temp > 0 || intersected[pId] == 1)
			intersected[pId] = 1;
		else
			intersected[pId] = 0;
		unintersected[pId] = (intersected[pId] == 1 ? 0 : 1);
	}

	template <typename Coord,typename Edge, typename Real>
	__global__ void CalEdgesDistance(
		DArray<Coord> points,
		DArray<Edge> edges,
		DArray<Real> edgesDistance,
		DArray<int> intersected,
		TRay3D<Real> mouseray)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= edges.size()) return;

		TPoint3D<Real> origin = TPoint3D<Real>(mouseray.origin);
		if (intersected[pId] != 0) 
		{
			TSegment3D<Real> s = TSegment3D<Real>(points[edges[pId].data[0]],points[edges[pId].data[1]]);
			edgesDistance[pId] = origin.distance(s);
		}
		else
		{
			edgesDistance[pId] = 3.4E38;
		}
	}

	__global__ void CalEdgesNearest(
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

	template <typename Edge, typename Coord, typename Real>
	__global__ void FindNearbyEdges(
		DArray<Coord> points,
		DArray<Coord> interPoints,
		DArray<Edge> edges,
		DArray<int> intersected,
		DArray<int> unintersected,
		int min_index_t,
		Real intersectionRadius)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		if (intersected[pId] == 1) 
		{
			TSegment3D<Real> s = TSegment3D<Real>(points[edges[pId].data[0]],points[edges[pId].data[1]]);
			TPoint3D<Real> p = TPoint3D<Real>(interPoints[min_index_t]);
			if (p.distance(s) > intersectionRadius) 
			{
				intersected[pId] = 0;
				unintersected[pId] = 1;
			}
		}
	}

	template <typename Edge, typename Real, typename Coord>
	__global__ void CalIntersectedEdgesBox(
		DArray<Coord> points,
		DArray<Edge> edges,
		DArray<int> intersected,
		DArray<int> unintersected,
		TPlane3D<Real> plane13,
		TPlane3D<Real> plane42,
		TPlane3D<Real> plane14,
		TPlane3D<Real> plane32)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= edges.size()) return;

		bool flag = false;
		for (int i = 0; i < 2; i++)
		{
			float temp1 = ((points[edges[pId].data[i]] - plane13.origin).dot(plane13.normal)) * ((points[edges[pId].data[i]] - plane42.origin).dot(plane42.normal));
			float temp2 = ((points[edges[pId].data[i]] - plane14.origin).dot(plane14.normal)) * ((points[edges[pId].data[i]] - plane32.origin).dot(plane32.normal));
			if (temp1 >= 0 && temp2 >= 0)
			{
				flag = true;
				break;
			}
		}

		if (flag || intersected[pId] == 1)
			intersected[pId] = 1;
		else
			intersected[pId] = 0;
		unintersected[pId] = (intersected[pId] == 1 ? 0 : 1);
	}

	template <typename Edge>
	__global__ void AssignOutEdges(
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
		TriangleSet<TDataType> initialTriangleSet = this->inInitialTriangleSet()->getData();
		DArray<Edge> edges = initialTriangleSet.getEdges();
		DArray<Coord> points = initialTriangleSet.getPoints();
		DArray<Triangle> triangles = initialTriangleSet.getTriangles();
		DArray<int> intersected_t;
		intersected_t.resize(triangles.size());

		cuExecute(triangles.size(),
			SurfaceInitializeArrayE,
			intersected_t
		);
		DArray<int> unintersected_t;
		unintersected_t.resize(triangles.size());
		//std::cout << "Triangle Num:" << triangles.size() << std::endl;

		DArray<Coord> interPoints;
		interPoints.resize(triangles.size());

		cuExecute(triangles.size(),
			CalIntersectedTrisRayE,
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
			CalTrisDistanceE,
			interPoints,
			trisDistance,
			intersected_t,
			this->ray1
		);

		int min_index_t = thrust::min_element(thrust::device, trisDistance.begin(), trisDistance.begin() + trisDistance.size()) - trisDistance.begin();

		DArray<int> intersected;
		intersected.resize(edges.size());
		cuExecute(edges.size(),
			EdgeInitializeArray,
			intersected
		);
		DArray<int> unintersected;
		unintersected.resize(edges.size());
		std::cout << "Edge Num:" << edges.size() << std::endl;
		cuExecute(edges.size(),
			CalIntersectedEdgesRay,
			points,
			edges,
			intersected,
			unintersected,
			this->ray1,
			this->varInterationRadius()->getData()
		);

		cuExecute(edges.size(),
			FindNearbyEdges,
			points,
			interPoints,
			edges,
			intersected,
			unintersected,
			min_index_t,
			this->varInterationRadius()->getData()
		);

		this->tempEdgeIntersectedIndex.assign(intersected);

		if (this->varToggleMultiSelect()->getData())
		{
			if (this->edgeIntersectedIndex.size() == 0)
			{
				this->edgeIntersectedIndex.resize(edges.size());
				cuExecute(edges.size(),
					EdgeInitializeArray,
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
					EdgeMergeIntersectedIndexOR,
					this->edgeIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
			{
				cuExecute(edges.size(),
					EdgeMergeIntersectedIndexXOR,
					this->edgeIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
			{
				cuExecute(edges.size(),
					EdgeMergeIntersectedIndexC,
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
			AssignOutEdges,
			edges,
			intersected_edges,
			unintersected_edges,
			intersected,
			unintersected,
			intersected_o
		);
		std::cout << "Selected Edges Num:" << intersected_edges.size() << std::endl;
		this->outSelectedEdgeSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedEdgeSet()->getDataPtr()->setEdges(intersected_edges);
		this->outOtherEdgeSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outOtherEdgeSet()->getDataPtr()->setEdges(unintersected_edges);
		this->outEdgeIndex()->getDataPtr()->assign(intersected_o);
	}

	template<typename TDataType>
	void EdgeInteraction<TDataType>::calcEdgeIntersectDrag()
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
		DArray<Edge> edges = initialTriangleSet.getEdges();
		DArray<Coord> points = initialTriangleSet.getPoints();
		DArray<int> intersected;
		intersected.resize(edges.size());
		cuExecute(edges.size(),
			EdgeInitializeArray,
			intersected
		);
		DArray<int> unintersected;
		unintersected.resize(edges.size());
		std::cout << "Edge Num:" << edges.size() << std::endl;
		cuExecute(edges.size(),
			CalIntersectedEdgesBox,
			points,
			edges,
			intersected,
			unintersected,
			plane13,
			plane42,
			plane14,
			plane32
		);
		cuExecute(edges.size(),
			CalIntersectedEdgesRay,
			points,
			edges,
			intersected,
			unintersected,
			this->ray1,
			this->varInterationRadius()->getData()
		);

		this->tempEdgeIntersectedIndex.assign(intersected);

		if (this->varToggleMultiSelect()->getData())
		{
			if (this->edgeIntersectedIndex.size() == 0)
			{
				this->edgeIntersectedIndex.resize(edges.size());
				cuExecute(edges.size(),
					EdgeInitializeArray,
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
					EdgeMergeIntersectedIndexOR,
					this->edgeIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
			{
				cuExecute(edges.size(),
					EdgeMergeIntersectedIndexXOR,
					this->edgeIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
			{
				cuExecute(edges.size(),
					EdgeMergeIntersectedIndexC,
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
			AssignOutEdges,
			edges,
			intersected_edges,
			unintersected_edges,
			intersected,
			unintersected,
			intersected_o
		);
		std::cout << "Selected Edges Num:" << intersected_edges.size() << std::endl;
		this->outSelectedEdgeSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedEdgeSet()->getDataPtr()->setEdges(intersected_edges);
		this->outOtherEdgeSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outOtherEdgeSet()->getDataPtr()->setEdges(unintersected_edges);
		this->outEdgeIndex()->getDataPtr()->assign(intersected_o);
	}

	template<typename TDataType>
	void EdgeInteraction<TDataType>::mergeIndex()
	{
		TriangleSet<TDataType> initialTriangleSet = this->inInitialTriangleSet()->getData();
		DArray<Edge> edges = initialTriangleSet.getEdges();
		DArray<Coord> points = initialTriangleSet.getPoints();
		DArray<int> intersected;
		intersected.resize(edges.size());
		cuExecute(edges.size(),
			EdgeInitializeArray,
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
				EdgeMergeIntersectedIndexOR,
				this->edgeIntersectedIndex,
				this->tempEdgeIntersectedIndex,
				outIntersected,
				outUnintersected
			);
		}
		else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
		{
			cuExecute(edges.size(),
				EdgeMergeIntersectedIndexXOR,
				this->edgeIntersectedIndex,
				this->tempEdgeIntersectedIndex,
				outIntersected,
				outUnintersected
			);
		}
		else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
		{
			cuExecute(edges.size(),
				EdgeMergeIntersectedIndexC,
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
			AssignOutEdges,
			edges,
			intersected_edges,
			unintersected_edges,
			intersected,
			unintersected,
			intersected_o
		);
		std::cout << "Selected Edges Num:" << intersected_edges.size() << std::endl;
		this->outSelectedEdgeSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedEdgeSet()->getDataPtr()->setEdges(intersected_edges);
		this->outOtherEdgeSet()->getDataPtr()->copyFrom(initialTriangleSet);
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