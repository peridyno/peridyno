#include "SurfaceInteraction.h"
#include <thrust/sort.h>
#include <iostream>
#include <OrbitCamera.h>

namespace dyno
{
	__global__ void SI_SurfaceInitializeArray(
		DArray<int> intersected)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		intersected[pId] = 0;
	}

	__global__ void SI_SurfaceMergeIntersectedIndexOR(
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

	__global__ void SI_SurfaceMergeIntersectedIndexXOR(
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

	__global__ void SI_SurfaceMergeIntersectedIndexC(
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
			if (event.shiftKeyPressed() || event.controlKeyPressed())
			{
				this->varToggleMultiSelect()->setValue(true);
				if (event.shiftKeyPressed() && !event.controlKeyPressed())
				{
					this->varMultiSelectionType()->getDataPtr()->setCurrentKey(0);
				}
				else if (!event.shiftKeyPressed() && event.controlKeyPressed())
				{
					this->varMultiSelectionType()->getDataPtr()->setCurrentKey(1);
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
				if (this->varSurfacePickingType()->getValue() == PickingTypeSelection::Both || this->varSurfacePickingType()->getValue() == PickingTypeSelection::Click)
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
					if (abs(this->x2 - this->x1) <= 3 && abs(this->y2 - this->y1) <= 3)
					{
						if (this->varSurfacePickingType()->getValue() == PickingTypeSelection::Both || this->varSurfacePickingType()->getValue() == PickingTypeSelection::Click)
							this->calcIntersectClick();
					}
					else
					{
						if (this->varSurfacePickingType()->getValue() == PickingTypeSelection::Both || this->varSurfacePickingType()->getValue() == PickingTypeSelection::Drag)
						{
							this->calcIntersectDrag();
							this->printInfoDragging();
						}
					}
				}
			}
		}
	}

	template <typename Triangle, typename Real, typename Coord>
	__global__ void SI_CalIntersectedTrisRay(
		DArray<Coord> points,
		DArray<Triangle> triangles,
		DArray<int> intersected,
		DArray<int> unintersected,
		DArray<Real> triDistance,
		TRay3D<Real> mouseray)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= triangles.size()) return;

		TTriangle3D<Real> t = TTriangle3D<Real>(points[triangles[pId][0]], points[triangles[pId][1]], points[triangles[pId][2]]);
		int temp = 0;

		TPoint3D<Real> p;
		temp = mouseray.intersect(t, p);

		if (temp == 1)
		{
			intersected[pId] = 1;
			triDistance[pId] = (mouseray.origin - p.origin).norm();
		}
		else
		{
			intersected[pId] = 0;
			triDistance[pId] = 3.4E38;
		}
		unintersected[pId] = (intersected[pId] == 1 ? 0 : 1);
	}

	__global__ void SI_CalTrisNearest(
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

	template <typename Triangle, typename Real, typename Coord>
	__global__ void SI_CalIntersectedTrisBox(
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

		TTriangle3D<Real> t = TTriangle3D<Real>(points[triangles[pId][0]], points[triangles[pId][1]], points[triangles[pId][2]]);
		TSegment3D<Real> s1 = TSegment3D<Real>(points[triangles[pId][0]], points[triangles[pId][1]]);
		TSegment3D<Real> s2 = TSegment3D<Real>(points[triangles[pId][1]], points[triangles[pId][2]]);
		TSegment3D<Real> s3 = TSegment3D<Real>(points[triangles[pId][2]], points[triangles[pId][0]]);

		bool flag = false;
		TPoint3D<Real> p;
		bool temp11 = s1.intersect(plane13, p);
		temp11 = temp11 && (((p.origin - plane14.origin).dot(plane14.normal)) * ((p.origin - plane32.origin).dot(plane32.normal))) > 0;
		if (temp11)
			flag = true;
		bool temp12 = s1.intersect(plane42, p);
		temp12 = temp12 && (((p.origin - plane14.origin).dot(plane14.normal)) * ((p.origin - plane32.origin).dot(plane32.normal))) > 0;
		if (temp12)
			flag = true;
		bool temp13 = s1.intersect(plane14, p);
		temp13 = temp13 && (((p.origin - plane13.origin).dot(plane13.normal)) * ((p.origin - plane42.origin).dot(plane42.normal))) > 0;
		if (temp13)
			flag = true;
		bool temp14 = s1.intersect(plane32, p);
		temp14 = temp14 && (((p.origin - plane13.origin).dot(plane13.normal)) * ((p.origin - plane42.origin).dot(plane42.normal))) > 0;
		if (temp14)
			flag = true;
		bool temp21 = s2.intersect(plane13, p);
		temp21 = temp21 && (((p.origin - plane14.origin).dot(plane14.normal)) * ((p.origin - plane32.origin).dot(plane32.normal))) > 0;
		if (temp21)
			flag = true;
		bool temp22 = s2.intersect(plane42, p);
		temp22 = temp22 && (((p.origin - plane14.origin).dot(plane14.normal)) * ((p.origin - plane32.origin).dot(plane32.normal))) > 0;
		if (temp22)
			flag = true;
		bool temp23 = s2.intersect(plane14, p);
		temp23 = temp23 && (((p.origin - plane13.origin).dot(plane13.normal)) * ((p.origin - plane42.origin).dot(plane42.normal))) > 0;
		if (temp23)
			flag = true;
		bool temp24 = s2.intersect(plane32, p);
		temp24 = temp24 && (((p.origin - plane13.origin).dot(plane13.normal)) * ((p.origin - plane42.origin).dot(plane42.normal))) > 0;
		if (temp24)
			flag = true;
		bool temp31 = s3.intersect(plane13, p);
		temp31 = temp31 && (((p.origin - plane14.origin).dot(plane14.normal)) * ((p.origin - plane32.origin).dot(plane32.normal))) > 0;
		if (temp31)
			flag = true;
		bool temp32 = s3.intersect(plane42, p);
		temp32 = temp32 && (((p.origin - plane14.origin).dot(plane14.normal)) * ((p.origin - plane32.origin).dot(plane32.normal))) > 0;
		if (temp32)
			flag = true;
		bool temp33 = s3.intersect(plane14, p);
		temp33 = temp33 && (((p.origin - plane13.origin).dot(plane13.normal)) * ((p.origin - plane42.origin).dot(plane42.normal))) > 0;
		if (temp33)
			flag = true;
		bool temp34 = s3.intersect(plane32, p);
		temp34 = temp34 && (((p.origin - plane13.origin).dot(plane13.normal)) * ((p.origin - plane42.origin).dot(plane42.normal))) > 0;
		if (temp34)
			flag = true;

		for (int i = 0; i < 3; i++)
		{
			float temp1 = ((points[triangles[pId][i]] - plane13.origin).dot(plane13.normal)) * ((points[triangles[pId][i]] - plane42.origin).dot(plane42.normal));
			float temp2 = ((points[triangles[pId][i]] - plane14.origin).dot(plane14.normal)) * ((points[triangles[pId][i]] - plane32.origin).dot(plane32.normal));
			if (temp1 >= 0 && temp2 >= 0)
			{
				flag = flag || true;
				break;
			}
		}

		int temp = mouseray.intersect(t, p);
		if (temp == 1)
		{
			flag = flag || true;
		}

		if (flag || intersected[pId] == 1)
			intersected[pId] = 1;
		else
			intersected[pId] = 0;
		unintersected[pId] = (intersected[pId] == 1 ? 0 : 1);
	}

	template <typename Triangle>
	__global__ void SI_AssignOutTriangles(
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

	template <typename Triangle, typename Coord, typename Real>
	__global__ void SI_NeighborTrisDiffuse(
		DArray<Triangle> triangles,
		DArray<Coord> points,
		DArray<int> intersected,
		DArray<int> unintersected,
		Real diffusionAngle)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i >= triangles.size() || j >= triangles.size()) return;

		if (intersected[i] == 1)
		{
			if ((triangles[i][0] == triangles[j][0] && triangles[i][1] == triangles[j][1]) ||
				(triangles[i][0] == triangles[j][1] && triangles[i][1] == triangles[j][0]) ||
				(triangles[i][0] == triangles[j][0] && triangles[i][1] == triangles[j][2]) ||
				(triangles[i][0] == triangles[j][2] && triangles[i][1] == triangles[j][0]) ||
				(triangles[i][0] == triangles[j][1] && triangles[i][1] == triangles[j][2]) ||
				(triangles[i][0] == triangles[j][2] && triangles[i][1] == triangles[j][1]) ||

				(triangles[i][0] == triangles[j][0] && triangles[i][2] == triangles[j][1]) ||
				(triangles[i][0] == triangles[j][1] && triangles[i][2] == triangles[j][0]) ||
				(triangles[i][0] == triangles[j][0] && triangles[i][2] == triangles[j][2]) ||
				(triangles[i][0] == triangles[j][2] && triangles[i][2] == triangles[j][0]) ||
				(triangles[i][0] == triangles[j][1] && triangles[i][2] == triangles[j][2]) ||
				(triangles[i][0] == triangles[j][2] && triangles[i][2] == triangles[j][1]) ||

				(triangles[i][1] == triangles[j][0] && triangles[i][2] == triangles[j][1]) ||
				(triangles[i][1] == triangles[j][1] && triangles[i][2] == triangles[j][0]) ||
				(triangles[i][1] == triangles[j][0] && triangles[i][2] == triangles[j][2]) ||
				(triangles[i][1] == triangles[j][2] && triangles[i][2] == triangles[j][0]) ||
				(triangles[i][1] == triangles[j][1] && triangles[i][2] == triangles[j][2]) ||
				(triangles[i][1] == triangles[j][2] && triangles[i][2] == triangles[j][1])
				)
			{
				TTriangle3D<Real> t1(points[triangles[i][0]], points[triangles[i][1]], points[triangles[i][2]]);
				TTriangle3D<Real> t2(points[triangles[j][0]], points[triangles[j][1]], points[triangles[j][2]]);

				if (t1.normal().dot(t2.normal()) >= cosf(diffusionAngle))
				{
					if (intersected[j] == 0 && unintersected[j] == 1)
					{
						intersected[j] = 1;
						unintersected[j] = 0;
					}
				}
			}
		}
	}

	template <typename Triangle, typename Coord, typename Real>
	__global__ void SI_VisibleFilter(
		DArray<Triangle> triangles,
		DArray<Coord> points,
		DArray<int> intersected,
		DArray<int> unintersected,
		TRay3D<Real> mouseray)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= triangles.size()) return;

		if (intersected[pId] == 1)
		{
			TTriangle3D<Real> t = TTriangle3D<Real>(points[triangles[pId][0]], points[triangles[pId][1]], points[triangles[pId][2]]);
			if (mouseray.direction.dot(t.normal()) >= 0)
			{
				intersected[pId] = 0;
				unintersected[pId] = 1;
			}
		}
	}

	__global__ void SI_Tri2Quad(
		DArray<int> intersected,
		DArray<int> unintersected)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected.size()) return;

		if (intersected[pId] == 1)
		{
			if (pId % 2 == 1)
			{
				if (intersected[pId - 1] == 0)
				{
					intersected[pId - 1] = 1;
					unintersected[pId - 1] = 0;
				}
			}
			else 
			{

				if (intersected[pId + 1] == 0)
				{
					intersected[pId + 1] = 1;
					unintersected[pId + 1] = 0;
				}
			}
		}
	}

	__global__ void SI_QuadOutput(
		DArray<int> intersected_o,
		DArray<int> intersected_q
	) 
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= intersected_o.size()) return;


		intersected_q[pId / 2] = 0;
		if (intersected_o[pId] == 1)
		{
			intersected_q[pId / 2] = 1;
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
			SI_SurfaceInitializeArray,
			intersected
		);
		DArray<int> unintersected;
		unintersected.resize(triangles.size());
		this->tempNumT = triangles.size();

		DArray<Real> triDistance;
		triDistance.resize(triangles.size());

		cuExecute(triangles.size(),
			SI_CalIntersectedTrisRay,
			points,
			triangles,
			intersected,
			unintersected,
			triDistance,
			this->ray1
		);

		int min_index = thrust::min_element(thrust::device, triDistance.begin(), triDistance.begin() + triDistance.size()) - triDistance.begin();

		cuExecute(intersected.size(),
			SI_CalTrisNearest,
			min_index,
			intersected,
			unintersected
		);

		if (this->varToggleFlood()->getValue())
		{
			int intersected_size_t_o = 0;
			int intersected_size_t = 1;
			while (intersected_size_t > intersected_size_t_o && intersected_size_t < triangles.size())
			{
				intersected_size_t_o = intersected_size_t;
				cuExecute2D(make_uint2(triangles.size(), triangles.size()),
					SI_NeighborTrisDiffuse,
					triangles,
					points,
					intersected,
					unintersected,
					Real(this->varFloodAngle()->getValue()/180.0f*M_PI));
				intersected_size_t = thrust::reduce(thrust::device, intersected.begin(), intersected.begin() + intersected.size(), (int)0, thrust::plus<int>());
			}
		}

		if (this->varToggleVisibleFilter()->getValue())
		{
			cuExecute(triangles.size(),
				SI_VisibleFilter,
				triangles,
				points,
				intersected,
				unintersected,
				this->ray1
			);
		}

		if (this->varToggleQuad()->getValue())
		{
			cuExecute(triangles.size(),
				SI_Tri2Quad,
				intersected,
				unintersected
			)
		}

		this->tempTriIntersectedIndex.assign(intersected);

		if (this->varToggleMultiSelect()->getData())
		{
			if (this->triIntersectedIndex.size() == 0)
			{
				this->triIntersectedIndex.resize(triangles.size());
				cuExecute(triangles.size(),
					SI_SurfaceInitializeArray,
					this->triIntersectedIndex
				)
			}
			DArray<int> outIntersected;
			outIntersected.resize(intersected.size());
			DArray<int> outUnintersected;
			outUnintersected.resize(unintersected.size());
			if (this->varMultiSelectionType()->getValue() == MultiSelectionType::OR)
			{
				cuExecute(triangles.size(),
					SI_SurfaceMergeIntersectedIndexOR,
					this->triIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
			{
				cuExecute(triangles.size(),
					SI_SurfaceMergeIntersectedIndexXOR,
					this->triIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
			{
				cuExecute(triangles.size(),
					SI_SurfaceMergeIntersectedIndexC,
					this->triIntersectedIndex,
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
			SI_AssignOutTriangles,
			triangles,
			intersected_triangles,
			unintersected_triangles,
			intersected,
			unintersected,
			intersected_o
		);

		if (this->varToggleQuad()->getValue())
		{
			DArray<int> intersected_q;
			intersected_q.resize(triangles.size() / 2);
			cuExecute(triangles.size(),
				SI_QuadOutput,
				intersected_o,
				intersected_q
			);
			intersected_o.assign(intersected_q);
		}

		this->tempNumS = intersected_size;
		this->outSelectedTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedTriangleSet()->getDataPtr()->setTriangles(intersected_triangles);
		this->outOtherTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outOtherTriangleSet()->getDataPtr()->setTriangles(unintersected_triangles);
		this->outTriangleIndex()->getDataPtr()->assign(intersected_o);
	}

	template<typename TDataType>
	void SurfaceInteraction<TDataType>::calcSurfaceIntersectDrag()
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

		TriangleSet<TDataType> initialTriangleSet = this->inInitialTriangleSet()->getData();
		DArray<Coord> points = initialTriangleSet.getPoints();
		DArray<Triangle> triangles = initialTriangleSet.getTriangles();
		DArray<int> intersected;
		intersected.resize(triangles.size());
		cuExecute(triangles.size(),
			SI_SurfaceInitializeArray,
			intersected
		);
		DArray<int> unintersected;
		unintersected.resize(triangles.size());
		this->tempNumT = triangles.size();

		cuExecute(triangles.size(),
			SI_CalIntersectedTrisBox,
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

		if (this->varToggleVisibleFilter()->getValue())
		{
			cuExecute(triangles.size(),
				SI_VisibleFilter,
				triangles,
				points,
				intersected,
				unintersected,
				this->ray1
			);
		}

		if (this->varToggleQuad()->getValue())
		{
			cuExecute(triangles.size(),
				SI_Tri2Quad,
				intersected,
				unintersected
			)
		}

		this->tempTriIntersectedIndex.assign(intersected);

		if (this->varToggleMultiSelect()->getData())
		{
			if (this->triIntersectedIndex.size() == 0)
			{
				this->triIntersectedIndex.resize(triangles.size());
				cuExecute(triangles.size(),
					SI_SurfaceInitializeArray,
					this->triIntersectedIndex
				)
			}
			DArray<int> outIntersected;
			outIntersected.resize(intersected.size());
			DArray<int> outUnintersected;
			outUnintersected.resize(unintersected.size());
			if (this->varMultiSelectionType()->getValue() == MultiSelectionType::OR)
			{
				cuExecute(triangles.size(),
					SI_SurfaceMergeIntersectedIndexOR,
					this->triIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
			{
				cuExecute(triangles.size(),
					SI_SurfaceMergeIntersectedIndexXOR,
					this->triIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
			{
				cuExecute(triangles.size(),
					SI_SurfaceMergeIntersectedIndexC,
					this->triIntersectedIndex,
					intersected,
					outIntersected,
					outUnintersected
				);
			}
			intersected.assign(outIntersected);
			unintersected.assign(outUnintersected);
			//this->triIntersectedIndex.assign(intersected);
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
			SI_AssignOutTriangles,
			triangles,
			intersected_triangles,
			unintersected_triangles,
			intersected,
			unintersected,
			intersected_o
		);

		if (this->varToggleQuad()->getValue())
		{
			DArray<int> intersected_q;
			intersected_q.resize(triangles.size() / 2);
			cuExecute(triangles.size(),
				SI_QuadOutput,
				intersected_o,
				intersected_q
			);
			intersected_o.assign(intersected_q);
		}

		this->tempNumS = intersected_size;
		this->outSelectedTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedTriangleSet()->getDataPtr()->setTriangles(intersected_triangles);
		this->outOtherTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outOtherTriangleSet()->getDataPtr()->setTriangles(unintersected_triangles);
		this->outTriangleIndex()->getDataPtr()->assign(intersected_o);
	}

	template<typename TDataType>
	void SurfaceInteraction<TDataType>::mergeIndex()
	{
		TriangleSet<TDataType> initialTriangleSet = this->inInitialTriangleSet()->getData();
		DArray<Coord> points = initialTriangleSet.getPoints();
		DArray<Triangle> triangles = initialTriangleSet.getTriangles();
		DArray<int> intersected;
		intersected.resize(triangles.size());
		cuExecute(triangles.size(),
			SI_SurfaceInitializeArray,
			intersected
		);
		DArray<int> unintersected;
		unintersected.resize(triangles.size());
		this->tempNumT = triangles.size();

		DArray<int> outIntersected;
		outIntersected.resize(intersected.size());
		DArray<int> outUnintersected;
		outUnintersected.resize(unintersected.size());

		if (this->varMultiSelectionType()->getValue() == MultiSelectionType::OR)
		{
			cuExecute(triangles.size(),
				SI_SurfaceMergeIntersectedIndexOR,
				this->triIntersectedIndex,
				this->tempTriIntersectedIndex,
				outIntersected,
				outUnintersected
			);
		}
		else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
		{
			cuExecute(triangles.size(),
				SI_SurfaceMergeIntersectedIndexXOR,
				this->triIntersectedIndex,
				this->tempTriIntersectedIndex,
				outIntersected,
				outUnintersected
			);
		}
		else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
		{
			cuExecute(triangles.size(),
				SI_SurfaceMergeIntersectedIndexC,
				this->triIntersectedIndex,
				this->tempTriIntersectedIndex,
				outIntersected,
				outUnintersected
			);
		}

		intersected.assign(outIntersected);
		unintersected.assign(outUnintersected);
		this->triIntersectedIndex.assign(intersected);

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
			SI_AssignOutTriangles,
			triangles,
			intersected_triangles,
			unintersected_triangles,
			intersected,
			unintersected,
			intersected_o
		);
		this->tempNumS = intersected_size;
		this->outSelectedTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedTriangleSet()->getDataPtr()->setTriangles(intersected_triangles);
		this->outOtherTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outOtherTriangleSet()->getDataPtr()->setTriangles(unintersected_triangles);
		this->outTriangleIndex()->getDataPtr()->assign(intersected_o);
	}

	template<typename TDataType>
	void SurfaceInteraction<TDataType>::printInfoClick()
	{
		std::cout << "----------surface picking: click----------" << std::endl;
		std::cout << "multiple picking: " << this->varToggleMultiSelect()->getValue() << std::endl;
		std::cout << "selected num/ total num:" << this->tempNumS << "/" << this->tempNumT << std::endl;
	}

	template<typename TDataType>
	void SurfaceInteraction<TDataType>::printInfoDragging()
	{
		std::cout << "----------surface picking: dragging----------" << std::endl;
		std::cout << "multiple picking: " << this->varToggleMultiSelect()->getValue() << std::endl;
		std::cout << "selected num/ total num:" << this->tempNumS << "/" << this->tempNumT << std::endl;
	}

	template<typename TDataType>
	void SurfaceInteraction<TDataType>::printInfoDragRelease()
	{
		std::cout << "----------surface picking: drag release----------" << std::endl;
		std::cout << "multiple picking: " << this->varToggleMultiSelect()->getValue() << std::endl;
		std::cout << "selected num/ total num:" << this->tempNumS << "/" << this->tempNumT << std::endl;
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