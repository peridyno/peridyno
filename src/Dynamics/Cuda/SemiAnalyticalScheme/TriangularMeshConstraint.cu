#include "TriangularMeshConstraint.h"

#include "Primitive/Primitive3D.h"
#include "Primitive/PrimitiveSweep3D.h"
#include "Algorithm/Reduction.h"

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

	template<typename Real, typename Coord, typename Matrix>
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
		Real thickness, // thickness of the boundary
		Real dt,
		// outputs
		DArray<Coord> linear_impulse,
		DArray<Coord> angular_impulse,
		DArray<Coord> delta_translation,
		DArray<Coord> delta_rotation,
		// params for impulse->motion conversion
		Real particle_mass,
		Real mesh_mass,
		Matrix mesh_inertia,
		Coord mesh_center)
	{
		typedef typename ::dyno::TPoint3D<Real> Point3D;
		typedef typename ::dyno::TTriangle3D<Real> Triangle3D;
		typedef typename ::dyno::TPointSweep3D<Real> PointSweep3D;

		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= particle_position.size()) return;

		// initialize outputs to zero by default
		linear_impulse[pId] = Coord(0);
		angular_impulse[pId] = Coord(0);
		delta_translation[pId] = Coord(0);
		delta_rotation[pId] = Coord(0);

		List<int>& nbrTriIds_i = triangle_neighbors[pId];
		int nbrSize = nbrTriIds_i.size();
		
		Coord pos_i = particle_position[pId];
		Coord pos_i_old = particle_position_previous[pId];
		Coord vel_old = particle_velocity[pId];

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
			// Real t;
			// typename Triangle3D::Param baryc;
			// bool intersected = ptSweep.intersect(triSweep, baryc, t);
			// if (intersected) {
			// 	//If the particle intersects a triangle according to CCD, move its position back to the middle point
			// 	point_end = ptSweep.interpolate(t / 2);
			// }

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

			Coord vel_tmp = vel_old + (weighted_pos - pos_i) / dt;

			Real vel_mag_i = vel_tmp.dot(weighted_normal);
			Coord vel_norm_i = vel_mag_i * weighted_normal;
			Coord vel_tan_i = vel_tmp - vel_norm_i;
			vel_norm_i *= (1.0f - friction_normal);
			vel_tan_i *= (1.0f - friction_tangent);

			Coord vel_new = vel_norm_i + vel_tan_i;
			particle_velocity[pId] = vel_new;

			// Compute 6D impulse on the mesh from this particle
			Coord dvel_particle = vel_new - vel_old;
			Coord J_lin_mesh = -particle_mass * dvel_particle; // impulse on mesh
			Coord r = weighted_pos - mesh_center;
			Coord tau_impulse = r.cross(J_lin_mesh);

			linear_impulse[pId] = J_lin_mesh;
			angular_impulse[pId] = tau_impulse;

			// Convert impulse to mesh motion increments using fake mass/inertia
			Coord dv_mesh = (mesh_mass > Real(0)) ? (J_lin_mesh / mesh_mass) : Coord(0);
			// small-angle approx for rotation increment
			Coord dw_mesh = mesh_inertia.inverse() * tau_impulse;

			delta_translation[pId] = dv_mesh * dt;
			delta_rotation[pId] = dw_mesh * dt;
		}
	}

	template<typename TDataType>
	void TriangularMeshConstraint<TDataType>::constrain()
	{
		using Real = typename TDataType::Real;
		using Coord = typename TDataType::Coord;
		using Matrix = typename TDataType::Matrix;

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

		// resize outputs
		LinearImpulse.resize(p_num);
		AngularImpulse.resize(p_num);
		DeltaTranslation.resize(p_num);
		DeltaRotation.resize(p_num);
		LinearImpulse.reset();
		AngularImpulse.reset();
		DeltaTranslation.reset();
		DeltaRotation.reset();

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
				dt,
				// outputs
				LinearImpulse,
				AngularImpulse,
				DeltaTranslation,
				DeltaRotation,
				// params
				this->varParticleMass()->getValue(),
				this->varMeshMass()->getValue(),
				this->varMeshInertia()->getValue(),
				this->varMeshCenter()->getValue());
		}


		// aggregated totals
		Reduction<Coord> reduce;
		Coord totalLinJ = reduce.accumulate(LinearImpulse.begin(), LinearImpulse.size());
		Coord totalAngJ = reduce.accumulate(AngularImpulse.begin(), AngularImpulse.size());
		Coord totalDx = reduce.accumulate(DeltaTranslation.begin(), DeltaTranslation.size());
		Coord totalDtheta = reduce.accumulate(DeltaRotation.begin(), DeltaRotation.size());

		std::cout << totalLinJ.x << totalLinJ.y << totalLinJ.z << std::endl;
		std::cout << totalAngJ.x << totalAngJ.y << totalAngJ.z << std::endl;


		mPrivousVertex.assign(vertices);
		mPreviousPosition.assign(positions);
	}

	DEFINE_CLASS(TriangularMeshConstraint);
}