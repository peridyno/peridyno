#include "Coupling.h"

#include "Math/Lerp.h"

#include "Primitive/Primitive3D.h"

namespace dyno
{
	template<typename TDataType>
	Coupling<TDataType>::Coupling()
		: Node()
	{
	}

	template<typename TDataType>
	Coupling<TDataType>::~Coupling()
	{
		mForce.clear();
		mTorque.clear();
	}

	template<typename TDataType>
	void Coupling<TDataType>::resetStates()
	{
	}

	template<typename Coord, typename Triangle>
	__global__ void C_ComputeForceAndTorque(
		DArray<Coord> force,
		DArray<Coord> torque,
		DArray<Coord> vertices,
		DArray<Triangle> indices,
		DArray2D<Coord> heights,
		Coord barycenter,
		Coord gravity,
		Coord origin,
		Real spacing,
		Real rho)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= indices.size()) return;

		Triangle index_i = indices[tId];

		Coord v0 = vertices[index_i[0]];
		Coord v1 = vertices[index_i[1]];
		Coord v2 = vertices[index_i[2]];

		Triangle3D triangle(v0, v1, v2);

		//Triangle normal
		Coord normal_i = (v2 - v0).cross(v1 - v0);
		normal_i.normalize();

		Coord triangle_center = (v0 + v1 + v2) / Real(3);

		Coord d_i = bilinear(heights, (triangle_center.x - origin.x) / spacing, (triangle_center.z - origin.z) / spacing);

		//Calculate buoyancy
		Real sea_level = d_i.y;
		Real h = triangle_center.y < sea_level ? (sea_level - triangle_center.y) : Real(0);

		Real pressure = rho * gravity.norm() * h;

		Coord force_i = pressure * triangle.area() * normal_i;
		Coord torque_i = -force_i.cross(triangle_center - barycenter);

		force[tId] = force_i;
		torque[tId] = torque_i;
	}

	template<typename TDataType>
	void Coupling<TDataType>::updateStates()
	{
		Real dt = this->stateTimeStep()->getData();

		auto vessels = this->getVessels();
		auto ocean = this->getOcean();

		for (auto mesh : vessels)
		{
			auto& triangles = mesh->stateEnvelope()->getData();

			Real mass = mesh->stateMass()->getData();
			Coord barycenter = mesh->stateBarycenter()->getData();
			Coord velocity = mesh->stateVelocity()->getData();
			Coord angular_velocity = mesh->stateAngularVelocity()->getData();
			Matrix inertia = mesh->stateInertia()->getData();

			Coord gravity = mesh->varGravity()->getData();

			auto& vertices = triangles.getPoints();
			auto& indices = triangles.getTriangles();

			uint num = indices.size();

			if (mForce.size() != num) {
				mForce.resize(num);
				mTorque.resize(num);
			}

			auto heights = ocean->stateHeightField()->getDataPtr();
			auto& displacements = heights->getDisplacement();
			Coord origin = heights->getOrigin();
			Real h = heights->getGridSpacing();

			cuExecute(num,
				C_ComputeForceAndTorque,
				mForce,
				mTorque,
				vertices,
				indices,
				displacements,
				barycenter,
				gravity,
				origin,
				h,
				Real(1000));

			Coord F_total = mReduce.accumulate(mForce.begin(), mForce.size());
			Coord T_total = mReduce.accumulate(mTorque.begin(), mTorque.size());

			velocity += dt * F_total / mass;
			angular_velocity += dt * inertia.inverse() * T_total;

			velocity *= this->varDamping()->getValue();
			angular_velocity *= this->varRotationalDamping()->getValue();

			mesh->stateVelocity()->setValue(velocity);
			mesh->stateAngularVelocity()->setValue(angular_velocity);
		}
	}

	DEFINE_CLASS(Coupling);
}
