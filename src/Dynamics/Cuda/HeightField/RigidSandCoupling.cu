#include "RigidSandCoupling.h"

#include "Math/Lerp.h"

#include "Primitive/Primitive3D.h"

namespace dyno
{
	template<typename TDataType>
	RigidSandCoupling<TDataType>::RigidSandCoupling()
		: Node()
	{
	}

	template<typename TDataType>
	RigidSandCoupling<TDataType>::~RigidSandCoupling()
	{
	}

	template<typename TDataType>
	void RigidSandCoupling<TDataType>::resetStates()
	{
	}

	template<typename Real, typename Coord3D, typename Coord4D, typename Box3D>
	__global__ void CQWS_CoupleSandWithBoxes(
		DArray2D<Coord4D> grid,
		DArray2D<Coord3D> displacements,
		DArray<Box3D> boxes,
		DArray<Coord3D> boxVel,
		DArray<Coord3D> boxAngularVel,
		Coord3D origin,
		Real spacing,
		uint offset)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;

		uint nx = displacements.nx();
		uint ny = displacements.ny();

		if (i < nx && j < ny)
		{
			Coord3D v = origin + spacing * Coord3D(i, 0, j) + displacements(i, j);
			Point3D p = Point3D(v);

			uint nb = boxes.size();
			int boxId = -1;
			Real Dmin = 100000.0f;
			for (uint k = 0; k < nb; k++)
			{
				Real d = p.distance(boxes[k]);
				if (d < Dmin)
				{
					Dmin = d;
					boxId = k;
				}
			}

			if (Dmin < 0 && boxId != -1)
			{
				Coord4D gij = grid(i + 1, j + 1);

				Coord3D c = boxes[boxId].center;

				Coord3D v_box = boxVel[offset + boxId];
				Coord3D omega_box = boxAngularVel[offset + boxId];

				Coord3D vv = v_box + omega_box.cross(v - c);

				Real duh = gij.x * vv.x;
				Real dvh = gij.x * vv.z;
				Real dw = abs(Dmin) * 5;
				atomicAdd(&grid(i + 1, j + 1).y, duh);
				atomicAdd(&grid(i + 1, j + 1).z, dvh);

				atomicAdd(&grid(i + 2, j + 1).y, 0.25 * dw);
				atomicAdd(&grid(i + 0, j + 1).y, -0.25 * dw);

				atomicAdd(&grid(i + 1, j + 2).z, 0.25 * dw);
				atomicAdd(&grid(i + 1, j + 0).z, -0.25 * dw);
			}
		}
	}


	template<typename Real, typename Coord3D, typename Coord4D, typename Sphere3D>
	__global__ void CQWS_CoupleSandAndSpheres(
		DArray2D<Coord4D> grid,
		DArray2D<Coord3D> displacements,
		DArray<Sphere3D> spheres,
		DArray<Coord3D> ele_vel,
		DArray<Coord3D> ele_vel_angular,
		Coord3D origin,
		Real spacing,
		uint offset)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;

		uint nx = displacements.nx();
		uint ny = displacements.ny();

		if (i < nx && j < ny)
		{
			Coord3D v = origin + spacing * Coord3D(i, 0, j) + displacements(i, j);
			Point3D p = Point3D(v);

			uint nb = spheres.size();
			int sphereId = -1;
			Real Dmin = 100000.0f;
			for (uint k = 0; k < nb; k++)
			{
				Real d = p.distance(spheres[k]);
				if (d < Dmin)
				{
					Dmin = d;
					sphereId = k;
				}
			}

			if (Dmin < 0 && sphereId != -1)
			{
				Coord4D gij = grid(i + 1, j + 1);

				Coord3D c = spheres[sphereId].center;

				Coord3D v_i = ele_vel[offset + sphereId];
				Coord3D omega_i = ele_vel_angular[offset + sphereId];

				Coord3D vv = v_i + omega_i.cross(v - c);

				Real duh = gij.x * vv.x;
				Real dvh = gij.x * vv.z;
				Real dw = abs(Dmin) * 5;
				atomicAdd(&grid(i + 1, j + 1).y, duh);
				atomicAdd(&grid(i + 1, j + 1).z, dvh);

				atomicAdd(&grid(i + 2, j + 1).y, 0.25 * dw);
				atomicAdd(&grid(i + 0, j + 1).y, -0.25 * dw);

				atomicAdd(&grid(i + 1, j + 2).z, 0.25 * dw);
				atomicAdd(&grid(i + 1, j + 0).z, -0.25 * dw);
			}
		}
	}

	template<typename Real, typename Coord3D, typename Coord4D, typename Capsule3D>
	__global__ void CQWS_CoupleSandAndCapsules(
		DArray2D<Coord4D> grid,
		DArray2D<Coord3D> displacements,
		DArray<Capsule3D> capsules,
		DArray<Coord3D> ele_vel,
		DArray<Coord3D> ele_vel_angular,
		Coord3D origin,
		Real spacing,
		uint offset)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;

		uint nx = displacements.nx();
		uint ny = displacements.ny();

		if (i < nx && j < ny)
		{
			Coord3D v = origin + spacing * Coord3D(i, 0, j) + displacements(i, j);
			Point3D p = Point3D(v);

			uint nb = capsules.size();
			int sphereId = -1;
			Real Dmin = 100000.0f;
			for (uint k = 0; k < nb; k++)
			{
				Real d = p.distance(capsules[k]);
				if (d < Dmin)
				{
					Dmin = d;
					sphereId = k;
				}
			}

			if (Dmin < 0 && sphereId != -1)
			{
				Coord4D gij = grid(i + 1, j + 1);

				Coord3D c = capsules[sphereId].center;

				Coord3D v_i = ele_vel[offset + sphereId];
				Coord3D omega_i = ele_vel_angular[offset + sphereId];

				Coord3D vv = v_i + omega_i.cross(v - c);

				Real duh = gij.x * vv.x;
				Real dvh = gij.x * vv.z;
				Real dw = abs(Dmin) * 5;
				atomicAdd(&grid(i + 1, j + 1).y, duh);
				atomicAdd(&grid(i + 1, j + 1).z, dvh);

				atomicAdd(&grid(i + 2, j + 1).y, 0.25 * dw);
				atomicAdd(&grid(i + 0, j + 1).y, -0.25 * dw);

				atomicAdd(&grid(i + 1, j + 2).z, 0.25 * dw);
				atomicAdd(&grid(i + 1, j + 0).z, -0.25 * dw);
			}
		}
	}

	template<typename TDataType>
	void RigidSandCoupling<TDataType>::updateStates()
	{
		Real dt = this->stateTimeStep()->getData();

		auto sand = this->getGranularMedia();
		auto rigidbody = this->getRigidBodySystem();

		auto heights = sand->stateHeightField()->constDataPtr();
		auto elements = rigidbody->stateTopology()->constDataPtr();

		auto& grid2d = sand->stateGrid()->getData();

		auto& ele_vel = rigidbody->stateVelocity()->getData();
		auto& ele_angular_vel = rigidbody->stateAngularVelocity()->getData();

		auto& boxes = elements->boxesInGlobal();
		auto& spheres = elements->spheresInGlobal();
		auto& capsules = elements->capsulesInGlobal();

		auto offset = elements->calculateElementOffset();

		auto& disp = heights->getDisplacement();
		auto origin = heights->getOrigin();
		auto spacing = heights->getGridSpacing();

		uint boxOffset = offset.boxIndex();
		uint sphereOffset = offset.sphereIndex();
		uint capsuleOffset = offset.capsuleIndex();

		cuExecute2D(make_uint2(disp.nx(), disp.ny()),
			CQWS_CoupleSandWithBoxes,
			grid2d,
			disp,
			boxes,
			ele_vel,
			ele_angular_vel,
			origin,
			spacing,
			boxOffset);

		cuExecute2D(make_uint2(disp.nx(), disp.ny()),
			CQWS_CoupleSandAndSpheres,
			grid2d,
			disp,
			spheres,
			ele_vel,
			ele_angular_vel,
			origin,
			spacing,
			sphereOffset);

		cuExecute2D(make_uint2(disp.nx(), disp.ny()),
			CQWS_CoupleSandAndCapsules,
			grid2d,
			disp,
			capsules,
			ele_vel,
			ele_angular_vel,
			origin,
			spacing,
			capsuleOffset);
	}

	DEFINE_CLASS(RigidSandCoupling);
}
