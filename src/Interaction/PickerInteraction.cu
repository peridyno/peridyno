#include "PickerInteraction.h"
#include <thrust/sort.h>
#include <iostream>
#include <OrbitCamera.h>

namespace dyno
{
	template<typename TDataType>
	PickerInteraction<TDataType>::PickerInteraction()
	{
		this->ray1 = TRay3D<Real>();
		this->ray2 = TRay3D<Real>();
		this->isPressed = false;
	}

	template<typename TDataType>
	void PickerInteraction<TDataType>::onEvent(PMouseEvent event)
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
	void PickerInteraction<TDataType>::calcSurfaceIntersectClick()
	{
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
			CalIntersectedTrisRay,
			points,
			triangles,
			intersected,
			unintersected,
			this->ray1
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

	template<typename TDataType>
	void PickerInteraction<TDataType>::calcSurfaceIntersectDrag()
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

	template<typename TDataType>
	void PickerInteraction<TDataType>::calcEdgeIntersectClick()
	{
		TriangleSet<TDataType> initialTriangleSet = this->inInitialTriangleSet()->getData();
		DArray<Edge> edges = initialTriangleSet.getEdges();
		DArray<Coord> points = initialTriangleSet.getPoints();
		DArray<int> intersected;
		intersected.resize(edges.size());
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
		this->outSelectedEdgeSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedEdgeSet()->getDataPtr()->setEdges(intersected_edges);
		this->outOtherEdgeSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outOtherEdgeSet()->getDataPtr()->setEdges(unintersected_edges);
	}

	template<typename TDataType>
	void PickerInteraction<TDataType>::calcEdgeIntersectDrag()
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
		this->outSelectedEdgeSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedEdgeSet()->getDataPtr()->setEdges(intersected_edges);
		this->outOtherEdgeSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outOtherEdgeSet()->getDataPtr()->setEdges(unintersected_edges);
	}

	template<typename TDataType>
	void PickerInteraction<TDataType>::calcPointIntersectClick()
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
			this->varInterationRadius()->getData()
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
	void PickerInteraction<TDataType>::calcPointIntersectDrag()
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
			this->varInterationRadius()->getData()
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
	void PickerInteraction<TDataType>::calcIntersectClick()
	{
		std::cout << this->varToggleSurfacePicker()->getData() << std::endl;
		if (this->varToggleSurfacePicker()->getData())
			calcSurfaceIntersectClick();
		if(this->varToggleEdgePicker()->getData())
			calcEdgeIntersectClick();
		if (this->varTogglePointPicker()->getData())
			calcPointIntersectClick();
	}

	template<typename TDataType>
	void PickerInteraction<TDataType>::calcIntersectDrag()
	{
		std::cout << this->varToggleSurfacePicker()->getData() << std::endl;
		if (this->varToggleSurfacePicker()->getData())
			calcSurfaceIntersectDrag();
		if (this->varToggleEdgePicker()->getData())
			calcEdgeIntersectDrag();
		if (this->varTogglePointPicker()->getData())
			calcPointIntersectDrag();
	}

	DEFINE_CLASS(PickerInteraction);
}