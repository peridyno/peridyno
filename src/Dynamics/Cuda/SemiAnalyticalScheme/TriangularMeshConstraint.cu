#include "TriangularMeshConstraint.h"

#include "Primitive/Primitive3D.h"
#include "Primitive/PrimitiveSweep3D.h"

namespace dyno
{
	IMPLEMENT_TCLASS(TriangularMeshConstraint, TDataType)

	template<typename TDataType>
	TriangularMeshConstraint<TDataType>::TriangularMeshConstraint()
		: ConstraintModule()
	{
	}

	template<typename TDataType>
	TriangularMeshConstraint<TDataType>::~TriangularMeshConstraint()
	{
		mPosBuffer.clear();

		mPreviousPosition.clear();
		mPrivousVertex.clear();
	}

	//Implement the triangle clustering algorithm, refer to Section 4 in "Semi-analytical Solid Boundary Conditions for Free Surface Flows" by Chang et al., Pacific Graphics 2020.
// 	template <typename Coord>
// 	__global__ void VC_Sort_Neighbors_Collide(	
// 		DArray<Coord> position,
// 		DArray<TopologyModule::Triangle> m_triangle_index,
// 		DArray<Coord> positionTri,
// 		DArrayList<int> neighborsTri)
// 	{
// 		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
// 		if (pId >= position.size()) return;
// 
// 		List<int>& nbrTriIds_i = neighborsTri[pId];
// 		int nbSizeTri = nbrTriIds_i.size();
// 		
// 		Coord pos_i = position[pId];
// 
// 
// 		for (int ne = nbSizeTri / 2 - 1; ne >= 0; ne--)
// 		{
// 			int start = ne;
// 			int end = nbSizeTri - 1;
// 			int c = start;
// 			int l = 2 * c + 1;
// 			int tmp = nbrTriIds_i[c];
// 			for (; l <= end; c = l, l = 2 * l + 1)
// 			{
// 
// 				if (l < end)
// 				{
// 					bool judge = false;
// 					{
// 						int idx1, idx2;
// 						idx1 = nbrTriIds_i[l];
// 						idx2 = nbrTriIds_i[l + 1];
// 						
// 						Triangle3D t3d1(positionTri[m_triangle_index[idx1][0]], positionTri[m_triangle_index[idx1][1]], positionTri[m_triangle_index[idx1][2]]);
// 						Triangle3D t3d2(positionTri[m_triangle_index[idx2][0]], positionTri[m_triangle_index[idx2][1]], positionTri[m_triangle_index[idx2][2]]);
// 
// 						Coord normal1 = t3d1.normal().normalize();
// 						Coord normal2 = t3d2.normal().normalize();
// 
// 						Point3D p3d(pos_i);
// 
// 						Plane3D PL1(positionTri[m_triangle_index[idx1][0]], t3d1.normal());
// 						Plane3D PL2(positionTri[m_triangle_index[idx2][0]], t3d2.normal());
// 
// 						Real dis1 = p3d.distance(PL1);
// 						Real dis2 = p3d.distance(PL2);
// 
// 						if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON && abs(normal1[1] - normal2[1]) < EPSILON)
// 							judge = normal1[2] < normal2[2] ? true : false;
// 						else if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON)
// 							judge = normal1[1] < normal2[1] ? true : false;
// 						else if (abs(dis1 - dis2) < EPSILON)
// 							judge = normal1[0] < normal2[0] ? true : false;
// 						else
// 							judge = dis1 <= dis2 ? true : false;
// 
// 					}
// 					if (judge)
// 						l++;
// 				}
// 				bool judge = false;
// 				{
// 					int idx1, idx2;
// 					idx1 = nbrTriIds_i[l];
// 					idx2 = tmp;
// 					
// 					Triangle3D t3d1(positionTri[m_triangle_index[idx1][0]], positionTri[m_triangle_index[idx1][1]], positionTri[m_triangle_index[idx1][2]]);
// 					Triangle3D t3d2(positionTri[m_triangle_index[idx2][0]], positionTri[m_triangle_index[idx2][1]], positionTri[m_triangle_index[idx2][2]]);
// 
// 					Coord normal1 = t3d1.normal().normalize();
// 					Coord normal2 = t3d2.normal().normalize();
// 
// 					Point3D p3d(pos_i);
// 
// 					Plane3D PL1(positionTri[m_triangle_index[idx1][0]], t3d1.normal());
// 					Plane3D PL2(positionTri[m_triangle_index[idx2][0]], t3d2.normal());
// 
// 					Real dis1 = p3d.distance(PL1);
// 					Real dis2 = p3d.distance(PL2);
// 
// 					if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON && abs(normal1[1] - normal2[1]) < EPSILON)
// 						judge = normal1[2] <= normal2[2] ? true : false;
// 					else if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON)
// 						judge = normal1[1] <= normal2[1] ? true : false;
// 					else if (abs(dis1 - dis2) < EPSILON)
// 						judge = normal1[0] <= normal2[0] ? true : false;
// 					else
// 						judge = dis1 <= dis2 ? true : false;
// 
// 				}
// 				if (judge)
// 					break;
// 				else
// 				{
// 					
// 					nbrTriIds_i[c] = nbrTriIds_i[l];
// 					nbrTriIds_i[l] = tmp;
// 				}
// 			}
// 		}
// 		for (int ne = nbSizeTri - 1; ne > 0; ne--)
// 		{
// 			int swap_tmp = nbrTriIds_i[0];
// 	
// 			nbrTriIds_i[0] = nbrTriIds_i[ne];
// 			nbrTriIds_i[ne] = swap_tmp;
// 
// 			int start = 0;
// 			int end = ne - 1;
// 			int c = start;
// 			int l = 2 * c + 1;
// 			int tmp = nbrTriIds_i[c];
// 			for (; l <= end; c = l, l = 2 * l + 1)
// 			{
// 
// 				if (l < end)
// 				{
// 					bool judge = false;
// 					{
// 						int idx1, idx2;
// 						idx1 = nbrTriIds_i[l];
// 						idx2 = nbrTriIds_i[l + 1];
// 						
// 						Triangle3D t3d1(positionTri[m_triangle_index[idx1][0]], positionTri[m_triangle_index[idx1][1]], positionTri[m_triangle_index[idx1][2]]);
// 						Triangle3D t3d2(positionTri[m_triangle_index[idx2][0]], positionTri[m_triangle_index[idx2][1]], positionTri[m_triangle_index[idx2][2]]);
// 
// 						Coord normal1 = t3d1.normal().normalize();
// 						Coord normal2 = t3d2.normal().normalize();
// 
// 						Point3D p3d(pos_i);
// 
// 						Plane3D PL1(positionTri[m_triangle_index[idx1][0]], t3d1.normal());
// 						Plane3D PL2(positionTri[m_triangle_index[idx2][0]], t3d2.normal());
// 
// 						Real dis1 = p3d.distance(PL1);
// 						Real dis2 = p3d.distance(PL2);
// 
// 						if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON && abs(normal1[1] - normal2[1]) < EPSILON)
// 							judge = normal1[2] < normal2[2] ? true : false;
// 						else if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON)
// 							judge = normal1[1] < normal2[1] ? true : false;
// 						else if (abs(dis1 - dis2) < EPSILON)
// 							judge = normal1[0] < normal2[0] ? true : false;
// 						else
// 							judge = dis1 < dis2 ? true : false;
// 					}
// 					if (judge)
// 						l++;
// 				}
// 				bool judge = false;
// 				{
// 					int idx1, idx2;
// 					idx1 = nbrTriIds_i[l];
// 					idx2 = tmp;
// 					
// 					Triangle3D t3d1(positionTri[m_triangle_index[idx1][0]], positionTri[m_triangle_index[idx1][1]], positionTri[m_triangle_index[idx1][2]]);
// 					Triangle3D t3d2(positionTri[m_triangle_index[idx2][0]], positionTri[m_triangle_index[idx2][1]], positionTri[m_triangle_index[idx2][2]]);
// 
// 					Plane3D PL1(positionTri[m_triangle_index[idx1][0]], t3d1.normal());
// 					Plane3D PL2(positionTri[m_triangle_index[idx2][0]], t3d2.normal());
// 
// 					Coord normal1 = t3d1.normal().normalize();
// 					Coord normal2 = t3d2.normal().normalize();
// 
// 					Point3D p3d(pos_i);
// 
// 					Real dis1 = p3d.distance(PL1);
// 					Real dis2 = p3d.distance(PL2);
// 
// 					if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON && abs(normal1[1] - normal2[1]) < EPSILON)
// 						judge = normal1[2] <= normal2[2] ? true : false;
// 					else if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON)
// 						judge = normal1[1] <= normal2[1] ? true : false;
// 					else if (abs(dis1 - dis2) < EPSILON)
// 						judge = normal1[0] <= normal2[0] ? true : false;
// 					else
// 						judge = dis1 <= dis2 ? true : false;
// 				}
// 				if (judge)
// 					break;
// 				else
// 				{
// 					
// 					nbrTriIds_i[c] = nbrTriIds_i[l];
// 					nbrTriIds_i[l] = tmp;
// 				}
// 			}
// 		}
// 		return;
// 	}

	template<typename Real, typename Coord>
	__global__ void K_CCD_MESH (
		DArray<Coord> particle_position,
		DArray<Coord> particle_velocity,
		DArray<Coord> particle_position_previous,
		DArray<Coord> triangle_vertex,
		DArray<Coord> triangle_vertex_previous,
		DArray<TopologyModule::Triangle> triangle_index,
		DArrayList<int> triangle_neighbors,
		Real threshold,
		Real dt)
	{
		typedef typename TPoint3D<Real> Point3D;
		typedef typename TTriangle3D<Real> Triangle3D;
		typedef typename TPointSweep3D<Real> PointSweep3D;

		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= particle_position.size()) return;

		List<int>& nbrTriIds_i = triangle_neighbors[pId];
		int nbrSize = nbrTriIds_i.size();
		
		Coord pos_i = particle_position[pId];
		Coord pos_i_old = particle_position_previous[pId];
		Coord vel_i = particle_velocity[pId];

		Point3D start_point(particle_position[pId]);
		Point3D end_point(particle_position[pId]);
		PointSweep3D ptSweep(start_point, end_point);

		Real t = 100.0f;
		Real min_t = 10.0f;
		bool bIntersected = false;
		int min_j = -1;

		int interNum = 0;
		Coord total_pos(0);
		Coord delta_pos(0);

		for (int ne = 0; ne < nbrSize; ne++)
		{
			int j = nbrTriIds_i[ne];

			Point3D p3d(pos_i);
			Point3D p3dp(pos_i);

			Triangle3D t3d(triangle_vertex[triangle_index[j][0]], triangle_vertex[triangle_index[j][1]], triangle_vertex[triangle_index[j][2]]);
			Real min_distance = abs(p3d.distance(t3d));
			if (ne < nbrSize - 1)
			{
				int jn;
				do
				{
					jn = nbrTriIds_i[ne + 1];
					Triangle3D t3d_n(triangle_vertex[triangle_index[jn][0]], triangle_vertex[triangle_index[jn][1]], triangle_vertex[triangle_index[jn][2]]);
					if ((t3d.normal().cross(t3d_n.normal())).norm() > EPSILON * t3d_n.normal().norm() * t3d.normal().norm()) break;

					if (abs(p3d.distance(t3d_n)) < abs(min_distance))
					{
						j = jn;
						min_distance = abs(p3d.distance(t3d_n));
					}

					ne++;
				} while (ne < nbrSize - 1);
			}
		

			Triangle3D start_triangle(triangle_vertex_previous[triangle_index[j][0]], triangle_vertex_previous[triangle_index[j][1]], triangle_vertex_previous[triangle_index[j][2]]);
			Triangle3D end_triangle(triangle_vertex[triangle_index[j][0]], triangle_vertex[triangle_index[j][1]], triangle_vertex[triangle_index[j][2]]);

			TriangleSweep3D triSweep(start_triangle, end_triangle);

			typename Triangle3D::Param baryc;
			int num = ptSweep.intersect(triSweep, baryc, t, threshold);

			if (num > 0)
			{
				interNum++;
				bool tmp = false;
				Coord proj_j = end_triangle.computeLocation(baryc);

				Coord dir_j = start_point.origin - proj_j;
				if (dir_j.norm() > REAL_EPSILON)
				{
					dir_j.normalize();
				}


				Coord newpos = proj_j + dir_j * threshold;

				total_pos += newpos;// end_triangle.computeLocation(tParam) + threshold * end_triangle.normal();
				auto d_pos_i = newpos - pos_i;
				if (d_pos_i.norm() > threshold)
				{
					d_pos_i *= 0.0;// threshold / d_pos_i.norm();
				}
				delta_pos += d_pos_i;
			}
		}

		if (interNum > 0)
		{
			total_pos /= interNum;
			delta_pos /= interNum;

			particle_position[pId] = total_pos;

			Real norm_v = vel_i.norm();
			Coord old_vel = vel_i;

			vel_i += delta_pos / dt;

			particle_velocity[pId] = vel_i;
		}
	}

	template<typename TDataType>
	void TriangularMeshConstraint<TDataType>::constrain()
	{
		Real threshold = this->varThreshold()->getData();

		Real dt = this->inTimeStep()->getData();

		auto& positions = this->inPosition()->getData();
		auto& velocities = this->inVelocity()->getData();

		auto& vertices = this->inTriangleVertex()->getData();
		auto& triangles = this->inTriangleIndex()->getData();
		
		auto& neighborIds = this->inTriangleNeighborIds()->getData();

		int p_num = positions.size();
		if (p_num == 0) return;

		if (positions.size() != mPosBuffer.size()) {
			mPosBuffer.resize(p_num);
			mPreviousPosition.assign(positions);
		}

		mPrivousVertex.assign(vertices);

		if (neighborIds.size() > 0) {
// 			cuExecute(p_num, VC_Sort_Neighbors_Collide,
// 				positions,
// 				triangles,
// 				vertices,
// 				this->inTriangleNeighborIds()->getData()
// 			);
			cuExecute(p_num, K_CCD_MESH,
				positions,
				velocities,
				mPreviousPosition,
				vertices,
				mPrivousVertex,
				triangles,
				neighborIds,
				threshold,
				dt);
		}

		mPrivousVertex.assign(vertices);
		mPreviousPosition.assign(positions);
	}

	DEFINE_CLASS(TriangularMeshConstraint);
}