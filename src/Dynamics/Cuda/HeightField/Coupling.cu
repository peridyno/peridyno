#include "Coupling.h"

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

	template<typename Coord>
	__device__ Coord GetDisplacement(Coord pos, DArray2D<Coord>& displacement, Coord origin, Real h)
	{
		Real x = (pos.x - origin.x) / h;
		Real z = (pos.z - origin.z) / h;

		int i = floor(x);
		int j = floor(z);

		float fx = x - i;
		float fz = z - j;

		i = clamp((int)i, (int)0, (int)displacement.nx() - 1);
		j = clamp((int)j, (int)0, (int)displacement.ny() - 1);

		if (i == displacement.nx() - 1){
			i = displacement.nx() - 2;
			fx = 1.0f;
		}

		if (j == displacement.ny() - 1){
			j = displacement.ny() - 2;
			fz = 1.0f;
		}

		Coord d00 = displacement(i, j);
		Coord d10 = displacement(i + 1, j);
		Coord d01 = displacement(i, j + 1);
		Coord d11 = displacement(i + 1, j + 1);

		return d00 * (1 - fx) * (1 - fz) + d10 * fx * (1 - fz) + d01 * (1 - fx) * fz + d11 * fx * fz;
	}

	template<typename Coord, typename Triangle>
	__global__ void C_ComputeForceAndTorque(
		DArray<Coord> force,
		DArray<Coord> torque,
		DArray<Coord> vertices,
		DArray<Triangle> indices,
		DArray2D<Coord> heights,
		Coord center,
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

		Coord d_i = GetDisplacement(triangle_center, heights, origin, spacing);

		//Calculate buoyancy
		Real sea_level = d_i.y;
		Real h = triangle_center.y < sea_level ? (sea_level - triangle_center.y) : Real(0);

		Real pressure = rho * gravity.norm() * h;

		Coord force_i = pressure * triangle.area() * normal_i;
		Coord torque_i = force_i.cross(triangle_center - center);

		force[tId] = force_i;
		torque[tId] = torque_i;
	}

	template<typename TDataType>
	void Coupling<TDataType>::updateStates()
	{
		Real dt = this->stateTimeStep()->getData();

		auto mesh = this->getRigidMesh();
		auto ocean = this->getOcean();

		auto& triangles = mesh->stateEnvelope()->getData();

		Real mass = mesh->stateMass()->getData();
		Coord center = mesh->stateCenter()->getData();
		Coord velocity = mesh->stateVelocity()->getData();
		Coord angular_velocity = mesh->stateAngularVelocity()->getData();
		Matrix inertia = mesh->stateInertia()->getData();

		Coord gravity = mesh->varGravity()->getData();

		auto& vertices = triangles.getPoints();
		auto& indices = triangles.getTriangles();

		uint num = indices.size();

		if (mForce.size() != num){
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
			center,
			gravity,
			origin,
			h,
			Real(1000));

		Coord F_total = mReduce.accumulate(mForce.begin(), mForce.size());
		Coord T_total = mReduce.accumulate(mTorque.begin(), mTorque.size());

		velocity += dt * F_total / mass;
		angular_velocity += dt * inertia.inverse() * T_total;

		velocity *= this->varDragging()->getData();
		angular_velocity *= this->varDragging()->getData();

		mesh->stateVelocity()->setValue(velocity);
	}

	DEFINE_CLASS(Coupling);
}
