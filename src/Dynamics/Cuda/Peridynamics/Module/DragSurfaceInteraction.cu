#include "DragSurfaceInteraction.h"

#include "Primitive/Primitive3D.h"

#include <thrust/sort.h>
#include <iostream>
#include <OrbitCamera.h>
#define max_vel (10)
namespace dyno
{
	

	template<typename TDataType>
	DragSurfaceInteraction<TDataType>::DragSurfaceInteraction()
	{
		this->ray1 = TRay3D<Real>();
		this->ray2 = TRay3D<Real>();
		this->isPressed = false;
		this->intersectionCenter.resize(1);

	}

	template<typename TDataType>
	void DragSurfaceInteraction<TDataType>::onEvent(PMouseEvent event)
	{
		
		if (!event.altKeyPressed() && event.controlKeyPressed()) {
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
				this->restoreAttribute.assign(this->inAttribute()->getData());

				this->InteractionClick();
				//printf("Mouse pressed: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
			}
			else if (event.actionType == AT_RELEASE)
			{
				this->isPressed = false;
				this->cancelVelocity();
				this->inAttribute()->getData().assign(this->restoreAttribute);
				this->intersectionCenter.reset();
				//printf("Mouse released: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
				
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
						//change velocity to zero yet 
						this->cancelVelocity();
					}
					else 
					{
						this->InteractionDrag();
						//printf("Mouse repeated Draging: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
							
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

	__global__ void DS_SurfaceInitializeArray(
		DArray<int> intersected)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		intersected[pId] = 0;
	}

	template <typename Triangle, typename Real, typename Coord>
	__global__ void DS_CalcIntersectedTrisRay(
		DArray<Coord> points,
		DArray<Triangle> triangles,
		DArray<int> intersected,
		DArray<int> unintersected,
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
		unintersected[pId] = (intersected[pId] == 1 ? 0 : 1);
	}

	template <typename Real, typename Coord>
	__global__ void DS_CalcTrisDistance(
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

	__global__ void DS_CalcTrisNearest(
		int min_index,
		DArray<int> intersected,
		DArray<int> unintersected)
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

	template <typename Triangle, typename Coord, typename Real>
	__global__ void DS_FindNearbyTris(
		DArray<Coord> points,
		DArray<Triangle> triangles,
		DArray<Coord> interPoints,
		DArray<int> intersected,
		DArray<int> unintersected,
		DArray<Coord> center,
		int min_index,
		Real intersectionRadius
		)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		
			TPoint3D<Real> p = TPoint3D<Real>(interPoints[min_index]);
			TSegment3D<Real> s1 = TSegment3D<Real>(points[triangles[pId][0]], points[triangles[pId][1]]);
			TSegment3D<Real> s2 = TSegment3D<Real>(points[triangles[pId][1]], points[triangles[pId][2]]);
			TSegment3D<Real> s3 = TSegment3D<Real>(points[triangles[pId][2]], points[triangles[pId][0]]);
			if (pId == min_index)
				center[0] = interPoints[min_index];

			Real d1 = p.distance(s1);
			Real d2 = p.distance(s2);
			Real d3 = p.distance(s3);
			if (!(d1 > intersectionRadius && d2 > intersectionRadius && d3 > intersectionRadius) || intersected[pId]==1)
			{
				intersected[pId] = 1;
				unintersected[pId] = 0;
			}
			else
			{
				intersected[pId] = 0;
				unintersected[pId] = 1;
			}
	}


	__global__ void DS_AssignOutTriangles(
		DArray<int> intersected_triangles,
		DArray<int> intersected,
		DArray<int> intersected_o)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected_o.size()) return;

		if (intersected_o[pId] == 1)
		{
			intersected_triangles[intersected[pId]] = pId;
		}
	}

	template<typename TDataType>
	void DragSurfaceInteraction<TDataType>::calcSurfaceInteractClick()
		
	{
		
		auto& initialTriangleSet = this->inInitialTriangleSet()->getData();
		auto& points = initialTriangleSet.getPoints();
		auto& triangles = initialTriangleSet.getTriangles();

		DArray<int> intersected;
		intersected.resize(triangles.size());
		cuExecute(triangles.size(),
			DS_SurfaceInitializeArray,
			intersected
		);

		DArray<int> unintersected;
		unintersected.resize(triangles.size());
	
		DArray<Coord> interPoints;
		interPoints.resize(triangles.size());

		cuExecute(triangles.size(),
			DS_CalcIntersectedTrisRay,
			points,
			triangles,
			intersected,
			unintersected,
			interPoints,
			this->ray1
		);


		DArray<Real> trisDistance;
		trisDistance.resize(interPoints.size());

		cuExecute(interPoints.size(),
			DS_CalcTrisDistance,
			interPoints,
			trisDistance,
			intersected,
			this->ray1
		);

		int min_index = thrust::min_element(thrust::device, trisDistance.begin(), trisDistance.begin() + trisDistance.size()) - trisDistance.begin();
		this->intersectionCenterIndex = min_index;
		cuExecute(intersected.size(),
			DS_CalcTrisNearest,
			min_index,
			intersected,
			unintersected
		);
		
		cuExecute(triangles.size(),
			DS_FindNearbyTris,
			points,
			triangles,
			interPoints,
			intersected,
			unintersected,
			this->intersectionCenter,
			min_index,
			this->varInterationRadius()->getData()
		);

		
		this->triIntersectedIndex.assign(intersected);

		DArray<int> intersected_o;
		intersected_o.assign(intersected);

		int intersected_size = thrust::reduce(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), intersected.begin());
		
		this->intersected_triangles.resize(intersected_size);

		cuExecute(triangles.size(),
			DS_AssignOutTriangles,
			this->intersected_triangles,
			intersected,
			intersected_o
		);

		intersected_o.clear();
		
	}

	template <typename Real, typename Coord>
	__global__ void DS_SetFollowed(
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
	void DragSurfaceInteraction<TDataType>::cancelVelocity() {
		CArray<Coord> movement_c;
		movement_c.resize(1);
		movement_c[0] = Coord(0);
		DArray<Coord> movement;
		movement.resize(1);
		movement.assign(movement_c);

		auto& initialTriangleSet = this->inInitialTriangleSet()->getData();
		auto& triangles = initialTriangleSet.getTriangles();

		cuExecute(this->intersected_triangles.size(),
			DS_SetupVelocity,
			this->inVelocity()->getData(),
			movement,
			triangles,
			this->intersected_triangles,
			this->inTimeStep()->getData()
		);
		movement_c.clear();
		movement.clear();
	}

	template <typename Triangle, typename Real, typename Coord>
	__global__ void DS_SetupVelocity(
		DArray<Coord> Velocity,
		DArray<Coord> movement,
		DArray<Triangle> triangles,
		DArray<int> intersectionList,
		Real timestep)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersectionList.size()) return;

		Triangle tri = triangles[intersectionList[pId]];
		Coord velocity_candidate = movement[0] / timestep;
		if (velocity_candidate.norm() >= max_vel)
			velocity_candidate /= (velocity_candidate.norm() / max_vel);
	
		for (int index_tri = 0; index_tri < 3; ++index_tri)
		{
			Velocity[tri[index_tri]] = velocity_candidate;
		}
		
	}
	
	template<typename TDataType>
	void DragSurfaceInteraction<TDataType>::calcSurfaceInteractDrag()
		
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

		if (N_plane.norm() ==0.0) //error
			return;

		N_plane /= N_plane.norm();
		CArray<Coord> center; //this is a copy of intersectionCenter
		CArray<Coord> movement_c; // this is a copy used to calc velocity
		center.resize(1);
		center.assign(this->intersectionCenter);
		movement_c.resize(1);
		movement_c.assign(this->intersectionCenter);

		TPlane3D<Real> focal_plane = TPlane3D<Real>(center[0], N_plane);
		TPoint3D<Real> p;
		int temp = ray2.intersect(focal_plane, p);
		movement_c[0] = p.origin-center[0]; 
	
		Real dt = (Real)this->inTimeStep()->getData();
		Coord velocity_candidate = movement_c[0] / dt;
		if (velocity_candidate.norm() >= max_vel)
			velocity_candidate /= (velocity_candidate.norm() / max_vel);

		center[0] += velocity_candidate * dt;
		this->intersectionCenter.assign(center);

		DArray<Coord> movement;
		movement.resize(1);
		movement.assign(movement_c);
		auto& initialTriangleSet = this->inInitialTriangleSet()->getData();
		auto& triangles = initialTriangleSet.getTriangles();

		cuExecute(this->intersected_triangles.size(),
			DS_SetupVelocity,
			this->inVelocity()->getData(),
			movement,
			triangles,
			this->intersected_triangles,
			this->inTimeStep()->getData()
			);
		cuSynchronize();

		cuExecute(this->inPosition()->getData().size(),
			DS_SetFollowed,
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

	template <typename Triangle>
	__global__ void DS_FixedPoint(
		DArray<Triangle> triangles,
		DArray<int> intersected_tri,
		DArray<Attribute> inAtt) {

		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= intersected_tri.size()) return;
		
		int t_intersected_index = intersected_tri[tId];
		const Triangle& t = triangles[t_intersected_index];
		for (int index_t = 0; index_t < 3; ++index_t) {
			Attribute& att = inAtt[t[index_t]];
			att.setFixed();
		}
		
	}

	template<typename TDataType>
	void DragSurfaceInteraction<TDataType>::setTriFixed() {
		auto& TriSet = this->inInitialTriangleSet()->getData();
		auto& triangles = TriSet.getTriangles();
		int tNum = triangles.size();
		cuExecute(tNum,
			DS_FixedPoint,
			triangles,
			this->intersected_triangles,
			this->inAttribute()->getData()
		);
	}

	
	template<typename TDataType>
	void DragSurfaceInteraction<TDataType>::InteractionClick()
	{
		
		calcSurfaceInteractClick();
		this->setTriFixed();

	}

	template<typename TDataType>
	void DragSurfaceInteraction<TDataType>::InteractionDrag()
	{
	
		calcSurfaceInteractDrag();
		
	}

	DEFINE_CLASS(DragSurfaceInteraction);
}