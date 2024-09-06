#include "DragVertexInteraction.h"
#include <thrust/sort.h>
#include <iostream>
#include <OrbitCamera.h>
#define max_vel (10)
namespace dyno
{

	template<typename TDataType>
	DragVertexInteraction<TDataType>::DragVertexInteraction()
	{
		this->ray1 = TRay3D<Real>();
		this->ray2 = TRay3D<Real>();
		this->isPressed = false;
		this->intersectionCenter.resize(1);
		this->needInit = true;
	}

	template<typename TDataType>
	void DragVertexInteraction<TDataType>::onEvent(PMouseEvent event)
	{
		if (this->needInit) {
			this->restoreAttribute.assign(this->inAttribute()->getData());
			needInit = false;
			this->varCacheEvent()->setValue(true);
		}

		if (!event.altKeyPressed()) {
			if (camera == nullptr)
			{
				this->camera = event.camera;
			}
	
			if (event.actionType == AT_PRESS)
			{
				this->camera = event.camera;
				this->isPressed = true;
				this->ray1.origin = event.ray.origin;
				this->ray1.direction = event.ray.direction;
				this->x1 = event.x;
				this->y1 = event.y;
				
				this->InteractionClick();
				printf("Mouse pressed: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
			}
			else if (event.actionType == AT_RELEASE && this->isPressed)
			{
				this->isPressed = false;
				this->cancelVelocity();
				this->inAttribute()->getData().assign(this->restoreAttribute);
				this->intersectionCenter.reset();
				printf("Mouse released: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
				
			}
			else //pressed while moving(drag action) or just moving
			{
				
				if (this->isPressed) 
				{
					this->ray2.origin = event.ray.origin;
					this->ray2.direction = event.ray.direction;
					this->x2 = event.x;
					this->y2 = event.y;
					if (this->x2 == this->x1 && this->y2 == this->y1) 
					{
						//nothing need
					}
					else 
					{
						this->InteractionDrag();
						printf("Mouse repeated Draging: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
							
					}
					//swap
					this->ray1.origin = event.ray.origin;
					this->ray1.direction = event.ray.direction;
					this->x1 = event.x;
					this->y1 = event.y;
				}
			}
		}
	}

	__global__ void DV_PointInitializeArray(
		DArray<int> intersected)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		intersected[pId] = 0;
	}

	__global__ void DV_SurfaceInitializeArray(
		DArray<int> intersected)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		intersected[pId] = 0;
	}

	template <typename Triangle, typename Real, typename Coord>
	__global__ void DV_CalcIntersectedTrisRay(
		DArray<Coord> points,
		DArray<Triangle> triangles,
		DArray<int> intersected,
		DArray<Coord> interPoints,
		TRay3D<Real> mouseray)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= triangles.size()) return;

		TTriangle3D<Real> t = TTriangle3D<Real>(points[triangles[pId][0]], points[triangles[pId][1]], points[triangles[pId][2]]);
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
	}

	template <typename Real, typename Coord>
	__global__ void DV_CalcTrisDistance(
		DArray<Coord> interPoints,
		DArray<Real> trisDistance,
		DArray<int> intersected,
		TRay3D<Real> mouseray)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= interPoints.size()) return;
		if (intersected[pId] !=0) 
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

	template <typename Real, typename Coord>
	__global__ void DV_CalcIntersectedPointsRay(
		DArray<Coord> points,
		DArray<int> intersected,
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
	}

	template <typename Coord, typename Real>
	__global__ void DV_FindNearbyPoints(
		DArray<Coord> points,
		DArray<Coord> interPoints,
		DArray<int> intersected,
		int min_index_t,
		Real intersectionRadius,
		DArray<Coord> center)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		if (intersected[pId] == 1)
		{
			TPoint3D<Real> p1 = TPoint3D<Real>(points[pId]);
			TPoint3D<Real> p2 = TPoint3D<Real>(interPoints[min_index_t]);
			center[0] = p2.origin;
			if (p1.distance(p2) > intersectionRadius)
			{
				intersected[pId] = 0;
			}
		}
	}

	__global__ void DV_AssignOutPoints(
		DArray<int> intersected_points,
		DArray<int> intersected,
		DArray<int> intersected_o)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected_o.size()) return;

		if (intersected_o[pId] == 1)
		{
			intersected_points[intersected[pId]] = pId;
		}
	}


	template<typename TDataType>
	void DragVertexInteraction<TDataType>::calcVertexInteractClick()
	{
		
		auto& initialTriangleSet = this->inInitialTriangleSet()->getData();
		auto& points = initialTriangleSet.getPoints();
		auto& triangles = initialTriangleSet.getTriangles();
		DArray<int> intersected_t;
		intersected_t.resize(triangles.size());

		cuExecute(triangles.size(),
			DV_SurfaceInitializeArray,
			intersected_t
		);
	
		DArray<Coord> interPoints_t;
		interPoints_t.resize(triangles.size());

		cuExecute(triangles.size(),
			DV_CalcIntersectedTrisRay,
			points,
			triangles,
			intersected_t,
			interPoints_t,
			this->ray1
		);


		DArray<Real> trisDistance;
		trisDistance.resize(interPoints_t.size());

		cuExecute(interPoints_t.size(),
			DV_CalcTrisDistance,
			interPoints_t,
			trisDistance,
			intersected_t,
			this->ray1
		);

		int min_index_t = thrust::min_element(thrust::device, trisDistance.begin(), trisDistance.begin() + trisDistance.size()) - trisDistance.begin();
		
		DArray<int> intersected_v;
		intersected_v.resize(points.size());

		cuExecute(points.size(),
			DV_PointInitializeArray,
			intersected_v
		);


		cuExecute(points.size(),
			DV_CalcIntersectedPointsRay,
			points,
			intersected_v,
			this->ray1,
			this->varInterationRadius()->getData()
		);

		this->intersectionCenter.resize(1);
		
		cuExecute(points.size(),
			DV_FindNearbyPoints,
			points,
			interPoints_t,
			intersected_v,
			min_index_t,
			this->varInterationRadius()->getData(),
			this->intersectionCenter
		);

		this->verIntersectedIndex.assign(intersected_v);

		DArray<int> intersected_o;
		intersected_o.assign(intersected_v);

		int intersected_size = thrust::reduce(thrust::device, intersected_v.begin(), intersected_v.begin() + intersected_v.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, intersected_v.begin(), intersected_v.begin() + intersected_v.size(), intersected_v.begin());
		this->intersected_vertex.resize(intersected_size);

		cuExecute(points.size(),
			DV_AssignOutPoints,
			intersected_vertex,
			intersected_v,
			intersected_o
		);
		
		intersected_o.clear();
		intersected_t.clear();
		intersected_v.clear();
		
	}

	template <typename Real, typename Coord>
	__global__ void DV_SetFollowed(
		DArray<Attribute> att,
		DArray<Attribute> restore_att,
		DArray<Coord> Velocity,
		DArray<Coord> Position,
		Real timestep) {
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= att.size()) return;

		if (!restore_att[pId].isFixed() && att[pId].isFixed()) { //set fixed by drag
			Position[pId] += Velocity[pId] * timestep;
		}
		
	}

	template<typename TDataType>
	void DragVertexInteraction<TDataType>::cancelVelocity() {

		CArray<Coord> movement_c;
		movement_c.resize(1);
		movement_c[0] = Coord(0);
		DArray<Coord> movement;
		movement.resize(1);
		movement.assign(movement_c);

		cuExecute(this->intersected_vertex.size(),
			DV_SetupVelocity,
			this->inVelocity()->getData(),
			movement,
			this->intersected_vertex,
			this->inTimeStep()->getData()
		);
		movement_c.clear();
		movement.clear();
	}

	template <typename Real, typename Coord>
	__global__ void DV_SetupVelocity(
		DArray<Coord> Velocity,
		DArray<Coord> movement,
		DArray<int> intersectionList,
		Real timestep)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersectionList.size()) return;

		Coord velocity_candidate = movement[0] / timestep;
		if (velocity_candidate.norm() >= max_vel)
			velocity_candidate /= (velocity_candidate.norm() / max_vel);
	
			Velocity[intersectionList[pId]] = velocity_candidate;	
	}
	
	template<typename TDataType>
	void DragVertexInteraction<TDataType>::calcVertexInteractDrag()
		
	{
		
		TRay3D<Real> ray1 = this->ray1;
		TRay3D<Real> ray2 = this->ray2;

		TRay3D<Real> ray_o = this->camera->castRayInWorldSpace((float)0.0, (float)0.0);

		float height = this->camera->viewportHeight()/2.0;
		float width = this->camera->viewportWidth()/2.0;
		TRay3D<Real> ray_horizon = this->camera->castRayInWorldSpace(width, (float)0.0);
		TRay3D<Real> ray_vertical = this->camera->castRayInWorldSpace((float)0.0, height);
		
        //printf("(x1,y1) = %f,%f    (x2,y2) = %f,%f\n", x1, y1, x2, y2);
		auto ray_h = ray_o.origin - ray_horizon.origin;
		auto ray_v = ray_o.origin - ray_vertical.origin;
		auto N_plane = ray_h.cross(ray_v);

		if (N_plane.norm() == 0.0) //error
			return;

		N_plane /= N_plane.norm();
		CArray<Coord> movement_c; //this will be used to calc velocity
		CArray<Coord> center; //this is previous intersectionCenter
		center.resize(1);
		center.assign(this->intersectionCenter);
		movement_c.resize(1);
		movement_c.assign(this->intersectionCenter);

		TPlane3D<Real> focal_plane = TPlane3D<Real>(center[0], N_plane);
		TPoint3D<Real> p;
		int temp = ray2.intersect(focal_plane, p);
		movement_c[0] = p.origin-center[0]; 

		DArray<Coord> movement;
		movement.resize(1);
		movement.assign(movement_c);

		Real dt = (Real)this->inTimeStep()->getData();
		Coord velocity_candidate = movement_c[0] / dt;
		if (velocity_candidate.norm() >= max_vel)
			velocity_candidate /= (velocity_candidate.norm() / max_vel);

		center[0] += velocity_candidate * dt;
		this->intersectionCenter.assign(center);

		cuExecute(this->intersected_vertex.size(),
			DV_SetupVelocity,
			this->inVelocity()->getData(),
			movement,
			this->intersected_vertex,
			this->inTimeStep()->getData()
			);
		cuSynchronize();

		cuExecute(this->inPosition()->getData().size(),
			DV_SetFollowed,
			this->inAttribute()->getData(),
			this->restoreAttribute,
			this->inVelocity()->getData(),
			this->inPosition()->getData(),
			this->inTimeStep()->getData()
			);
		movement.clear();
		movement_c.clear();
		center.clear();
		//printf("==============================end intersection================================\n");
	}

	
	__global__ void DV_FixedPoint(
		DArray<int> intersected_vertex,
		DArray<Attribute> inAtt) {

		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= intersected_vertex.size()) return;
		
		Attribute& att = inAtt[intersected_vertex[vId]];
		att.setFixed();
	}

	template<typename TDataType>
	void DragVertexInteraction<TDataType>::setVertexFixed() {
		cuExecute(this->intersected_vertex.size(),
			DV_FixedPoint,
			this->intersected_vertex,
			this->inAttribute()->getData()
		);
	}

	
	template<typename TDataType>
	void DragVertexInteraction<TDataType>::InteractionClick()
	{
		
		calcVertexInteractClick();
		this->setVertexFixed();
	

	}

	template<typename TDataType>
	void DragVertexInteraction<TDataType>::InteractionDrag()
	{
	
		calcVertexInteractDrag();
		
	}

	DEFINE_CLASS(DragVertexInteraction);
}