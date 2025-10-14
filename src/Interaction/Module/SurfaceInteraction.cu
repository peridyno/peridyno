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
		if (this->outSelectedTriangleSet()->isEmpty())
			this->outSelectedTriangleSet()->allocate();
		if (this->outOtherTriangleSet()->isEmpty())
			this->outOtherTriangleSet()->allocate();
		if (this->outTriangleIndex()->isEmpty())
			this->outTriangleIndex()->allocate();
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
						//printf("Mouse repeated Clicking: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
					}
					else
					{
						//printf("Mouse repeated Draging: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
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
		DArray<int> outTriangleIndex,
		DArray<int> intersected,
		DArray<int> unintersected,
		DArray<int> intersected_o)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= triangles.size()) return;

		if (intersected_o[pId] == 1)
		{
			intersected_triangles[intersected[pId]] = triangles[pId];
			outTriangleIndex[intersected[pId]] = pId;
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
		DArray<TopologyModule::Tri2Edg> tri2Edg,
		DArray<TopologyModule::Edg2Tri> edg2Tri,
		DArray<int> intersected,
		DArray<int> unintersected,
		Real diffusionAngle)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= triangles.size()) return;

		if (intersected[i] == 1)
		{
			TTriangle3D<Real> t0, t1;
			t0 = TTriangle3D<Real>(points[triangles[i][0]], points[triangles[i][1]], points[triangles[i][2]]);
			for (int ii = 0; ii < 3; ii++)
			{
				for (int iii = 0; iii < 2; iii++)
				{
					if (intersected[edg2Tri[tri2Edg[i][ii]][iii]] != 1)
					{
						t1 = TTriangle3D<Real>(points[triangles[edg2Tri[tri2Edg[i][ii]][iii]][0]], points[triangles[edg2Tri[tri2Edg[i][ii]][iii]][1]], points[triangles[edg2Tri[tri2Edg[i][ii]][iii]][2]]);
						if (abs(t0.normal().dot(t1.normal())) >= cosf(diffusionAngle))
						{
							intersected[edg2Tri[tri2Edg[i][ii]][iii]] = 1;
							unintersected[edg2Tri[tri2Edg[i][ii]][iii]] = 0;
						}
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
		if (pId >= intersected_q.size()) return;

		intersected_q[pId] = 0;
		if (intersected_o[pId*2] == 1||intersected_o[pId*2+1]==1)
		{
			intersected_q[pId] = 1;
		}
	}

	__global__ void SI_QuadIndexOutput(
		DArray<int> outTriangleIndex,
		DArray<int> outQuadIndex
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= outQuadIndex.size()) return;

		outQuadIndex[pId] = outTriangleIndex[pId * 2] / 2;
	}

	__global__ void SI_InitialS2PSelected(
		DArray<int> s2PSelected)
	{
		int pId = threadIdx.x + blockIdx.x * blockDim.x;
		if (pId >= s2PSelected.size()) return;

		s2PSelected[pId] = 0;
	}

	template <typename Triangle>
	__global__ void SI_Surface2Point(
		DArray<Triangle> triangles,
		DArray<int> outTriangleIndex,
		DArray<int> s2PSelected) 
	{
		int pId = threadIdx.x + blockIdx.x * blockDim.x;
		if (pId >= outTriangleIndex.size()) return;

		s2PSelected[triangles[outTriangleIndex[pId]][0]] = 1;
		s2PSelected[triangles[outTriangleIndex[pId]][1]] = 1;
		s2PSelected[triangles[outTriangleIndex[pId]][2]] = 1;
	}

	__global__ void SI_s2PIndexOut(
		DArray<int> s2PIndex,
		DArray<int> s2PIndex_o,
		DArray<int> s2PIndexOut)
	{
		int pId = threadIdx.x + blockIdx.x * blockDim.x;
		if (pId >= s2PIndex.size()) return;

		if (s2PIndex_o[pId] == 1)
		{
			s2PIndexOut[s2PIndex[pId]] = pId;
		}
	}


	template<typename TDataType>
	void SurfaceInteraction<TDataType>::calcSurfaceIntersectClick()
	{
		auto& initialTriangleSet = this->inInitialTriangleSet()->getData();
		auto& points = initialTriangleSet.getPoints();
		auto& triangles = initialTriangleSet.triangleIndices();
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
			auto& Tri2Edg = initialTriangleSet.triangle2Edge();
			auto& Edge2Tri = initialTriangleSet.edge2Triangle();
			int intersected_size_t_o = 0;
			int intersected_size_t = 1;
			while (intersected_size_t > intersected_size_t_o && intersected_size_t < triangles.size())
			{
				intersected_size_t_o = intersected_size_t;

				cuExecute(triangles.size(),
					SI_NeighborTrisDiffuse,
					triangles,
					points,
					Tri2Edg,
					Edge2Tri,
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
		DArray<int> outTriangleIndex;
		outTriangleIndex.resize(intersected_size);
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
			outTriangleIndex,
			intersected,
			unintersected,
			intersected_o
		);

		DArray<int> s2PSelected;
		s2PSelected.resize(points.size());

		cuExecute(points.size(),
			SI_InitialS2PSelected,
			s2PSelected
		);

		cuExecute(outTriangleIndex.size(),
			SI_Surface2Point,
			triangles,
			outTriangleIndex,
			s2PSelected
		);
		int s2PSelectedSize = thrust::reduce(thrust::device, s2PSelected.begin(), s2PSelected.begin() + s2PSelected.size(), (int)0, thrust::plus<int>());
		DArray<int> s2PSelected_o;
		s2PSelected_o.assign(s2PSelected);

		thrust::exclusive_scan(thrust::device, s2PSelected.begin(), s2PSelected.begin() + s2PSelected.size(), s2PSelected.begin());

		DArray<int> s2PSelectedIndex;
		s2PSelectedIndex.resize(s2PSelectedSize);
		cuExecute(s2PSelected.size(),
			SI_s2PIndexOut,
			s2PSelected,
			s2PSelected_o,
			s2PSelectedIndex
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

			DArray<int> outQuadIndex;
			outQuadIndex.resize(outTriangleIndex.size()/2);
			cuExecute(outQuadIndex.size(),
				SI_QuadIndexOutput,
				outTriangleIndex,
				outQuadIndex
			);
			outTriangleIndex.assign(outQuadIndex);
		}

		this->tempNumS = intersected_size;
		this->outSelectedTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedTriangleSet()->getDataPtr()->setTriangles(intersected_triangles);
		this->outOtherTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outOtherTriangleSet()->getDataPtr()->setTriangles(unintersected_triangles);
		if (this->varToggleIndexOutput()->getValue())
		{
			this->outTriangleIndex()->getDataPtr()->assign(outTriangleIndex);
			this->outSur2PointIndex()->getDataPtr()->assign(s2PSelectedIndex);
		}
		else
		{
			this->outTriangleIndex()->getDataPtr()->assign(intersected_o);
			this->outSur2PointIndex()->getDataPtr()->assign(s2PSelected_o);
		}
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

		auto& initialTriangleSet = this->inInitialTriangleSet()->getData();
		auto& points = initialTriangleSet.getPoints();
		auto& triangles = initialTriangleSet.triangleIndices();
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
		DArray<int> outTriangleIndex;
		outTriangleIndex.resize(intersected_size);
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
			outTriangleIndex,
			intersected,
			unintersected,
			intersected_o
		);

		DArray<int> s2PSelected;
		s2PSelected.resize(points.size());

		cuExecute(points.size(),
			SI_InitialS2PSelected,
			s2PSelected
		);

		cuExecute(outTriangleIndex.size(),
			SI_Surface2Point,
			triangles,
			outTriangleIndex,
			s2PSelected
		);
		int s2PSelectedSize = thrust::reduce(thrust::device, s2PSelected.begin(), s2PSelected.begin() + s2PSelected.size(), (int)0, thrust::plus<int>());
		DArray<int> s2PSelected_o;
		s2PSelected_o.assign(s2PSelected);

		thrust::exclusive_scan(thrust::device, s2PSelected.begin(), s2PSelected.begin() + s2PSelected.size(), s2PSelected.begin());

		DArray<int> s2PSelectedIndex;
		s2PSelectedIndex.resize(s2PSelectedSize);
		cuExecute(s2PSelected.size(),
			SI_s2PIndexOut,
			s2PSelected,
			s2PSelected_o,
			s2PSelectedIndex
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

			DArray<int> outQuadIndex;
			outQuadIndex.resize(outTriangleIndex.size() / 2);
			cuExecute(outQuadIndex.size(),
				SI_QuadIndexOutput,
				outTriangleIndex,
				outQuadIndex
			);
			outTriangleIndex.assign(outQuadIndex);
		}

		this->tempNumS = intersected_size;
		this->outSelectedTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedTriangleSet()->getDataPtr()->setTriangles(intersected_triangles);
		this->outOtherTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outOtherTriangleSet()->getDataPtr()->setTriangles(unintersected_triangles);
		if (this->varToggleIndexOutput()->getValue())
		{
			this->outTriangleIndex()->getDataPtr()->assign(outTriangleIndex);
			this->outSur2PointIndex()->getDataPtr()->assign(s2PSelectedIndex);
		}
		else
		{
			this->outTriangleIndex()->getDataPtr()->assign(intersected_o);
			this->outSur2PointIndex()->getDataPtr()->assign(s2PSelected_o);
		}
	}

	template<typename TDataType>
	void SurfaceInteraction<TDataType>::mergeIndex()
	{
		auto& initialTriangleSet = this->inInitialTriangleSet()->getData();
		auto& points = initialTriangleSet.getPoints();
		auto& triangles = initialTriangleSet.triangleIndices();
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
		DArray<int> outTriangleIndex;
		outTriangleIndex.resize(intersected_size);
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
			outTriangleIndex,
			intersected,
			unintersected,
			intersected_o
		);

		DArray<int> s2PSelected;
		s2PSelected.resize(points.size());

		cuExecute(points.size(),
			SI_InitialS2PSelected,
			s2PSelected
		);

		cuExecute(outTriangleIndex.size(),
			SI_Surface2Point,
			triangles,
			outTriangleIndex,
			s2PSelected
		);
		int s2PSelectedSize = thrust::reduce(thrust::device, s2PSelected.begin(), s2PSelected.begin() + s2PSelected.size(), (int)0, thrust::plus<int>());
		DArray<int> s2PSelected_o;
		s2PSelected_o.assign(s2PSelected);

		thrust::exclusive_scan(thrust::device, s2PSelected.begin(), s2PSelected.begin() + s2PSelected.size(), s2PSelected.begin());

		DArray<int> s2PSelectedIndex;
		s2PSelectedIndex.resize(s2PSelectedSize);
		cuExecute(s2PSelected.size(),
			SI_s2PIndexOut,
			s2PSelected,
			s2PSelected_o,
			s2PSelectedIndex
		);

		this->tempNumS = intersected_size;
		this->outSelectedTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outSelectedTriangleSet()->getDataPtr()->setTriangles(intersected_triangles);
		this->outOtherTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
		this->outOtherTriangleSet()->getDataPtr()->setTriangles(unintersected_triangles);
		if (this->varToggleIndexOutput()->getValue())
		{
			this->outTriangleIndex()->getDataPtr()->assign(outTriangleIndex);
			this->outSur2PointIndex()->getDataPtr()->assign(s2PSelectedIndex);
		}
		else
		{
			this->outTriangleIndex()->getDataPtr()->assign(intersected_o);
			this->outSur2PointIndex()->getDataPtr()->assign(s2PSelected_o);
		}
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