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

	template<typename Real, typename Coord>
	__global__ void TMC_MeshConstrain(
		DArray<Coord> particle_position,
		DArray<Coord> particle_velocity,
		DArray<Coord> particle_position_previous,
		DArray<Coord> triangle_vertex,
		DArray<Coord> triangle_vertex_previous,
		DArray<TopologyModule::Triangle> triangle_index,
		DArrayList<int> triangle_neighbors,
		Real friction_normal,
		Real friction_tangent,
		Real thickness,	//thickness of the boundary
		Real dt)
	{
		typedef typename ::dyno::TPoint3D<Real> Point3D;
		typedef typename ::dyno::TTriangle3D<Real> Triangle3D;
		typedef typename ::dyno::TPointSweep3D<Real> PointSweep3D;

		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= particle_position.size()) return;

		List<int>& nbrTriIds_i = triangle_neighbors[pId];
		int nbrSize = nbrTriIds_i.size();
		
		Coord pos_i = particle_position[pId];
		Coord pos_i_old = particle_position_previous[pId];
		Coord vel_i = particle_velocity[pId];

		Point3D point_start(pos_i_old);
		Point3D point_end(pos_i);
		PointSweep3D ptSweep(point_start, point_end);

		//Intersection number
		int num = 0;

		Coord weighted_pos(0);
		Coord weighted_normal(0);


		for (int ne = 0; ne < nbrSize; ne++)
		{
			int j = nbrTriIds_i[ne];
	
			Triangle3D triangle_start(triangle_vertex_previous[triangle_index[j][0]], triangle_vertex_previous[triangle_index[j][1]], triangle_vertex_previous[triangle_index[j][2]]);
			Triangle3D triangle_end(triangle_vertex[triangle_index[j][0]], triangle_vertex[triangle_index[j][1]], triangle_vertex[triangle_index[j][2]]);

			TriangleSweep3D triSweep(triangle_start, triangle_end);

			//TODO: implement CCD
// 			Real t;
// 			typename Triangle3D::Param baryc;
// 			bool intersected = ptSweep.intersect(triSweep, baryc, t);
// 			if (intersected) {
// 				//If the particle intersects a triangle according to CCD, move its position back to the middle point
// 				point_end = ptSweep.interpolate(t / 2);
// 			}

			TPoint3D<Real> proj_end = point_end.project(triangle_end);

			Real dist_end = (proj_end.origin - point_end.origin).norm();

			if (dist_end <= thickness)
			{
				Coord dir_j = point_end.origin - proj_end.origin;

				if (dir_j.norm() > REAL_EPSILON) {
					dir_j.normalize();
				}

				Coord newpos = proj_end.origin + dir_j * thickness;

				weighted_normal += dir_j;
				weighted_pos += newpos;

				num++;
			}
		}

		if (num > 0)
		{
			//Calculate a weighted displacement
			weighted_pos /= num;
			weighted_normal /= num;

			particle_position[pId] = weighted_pos;

			vel_i += (weighted_pos - pos_i) / dt;

			Real vel_mag_i = vel_i.dot(weighted_normal);
			Coord vel_norm_i = vel_mag_i * weighted_normal;
			Coord vel_tan_i = vel_i - vel_norm_i;
			vel_norm_i *= (1.0f - friction_normal);
			vel_tan_i *= (1.0f - friction_tangent);

			particle_velocity[pId] = vel_norm_i + vel_tan_i;
		}
	}

	template<typename TDataType>
	void TriangularMeshConstraint<TDataType>::constrain()
	{
		Real thickness = this->varThickness()->getValue();

		Real dt = this->inTimeStep()->getValue();

		auto& positions = this->inPosition()->getData();
		auto& velocities = this->inVelocity()->getData();

		auto ts = this->inTriangleSet()->constDataPtr();

		auto& vertices = ts->getPoints();
		auto& triangles = ts->getTriangles();
		
		auto& neighborIds = this->inTriangleNeighborIds()->getData();

		int p_num = positions.size();
		if (p_num == 0) return;

		if (positions.size() != mPosBuffer.size()) {
			mPosBuffer.resize(p_num);
			mPreviousPosition.assign(positions);
		}

		mPrivousVertex.assign(vertices);

		if (neighborIds.size() > 0) {
			cuExecute(p_num, 
				TMC_MeshConstrain,
				positions,
				velocities,
				mPreviousPosition,
				vertices,
				mPrivousVertex,
				triangles,
				neighborIds,
				this->varNormalFriction()->getValue(),
				this->varTangentialFriction()->getValue(),
				thickness,
				dt);
		}

		mPrivousVertex.assign(vertices);
		mPreviousPosition.assign(positions);
	}

	DEFINE_CLASS(TriangularMeshConstraint);
}