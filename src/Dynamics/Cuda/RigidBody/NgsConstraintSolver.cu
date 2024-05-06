#include "NgsConstraintSolver.h"

namespace dyno
{
	IMPLEMENT_TCLASS(NgsConstraintSolver, TDataType)

	template<typename TDataType>
	NgsConstraintSolver<TDataType>::NgsConstraintSolver()
		:ConstraintModule()
	{
		this->inContacts()->tagOptional(true);
		this->inBallAndSocketJoints()->tagOptional(true);
		this->inSliderJoints()->tagOptional(true);
		this->inHingeJoints()->tagOptional(true);
		this->inFixedJoints()->tagOptional(true);
		this->inPointJoints()->tagOptional(true);
	}

	template<typename TDataType>
	NgsConstraintSolver<TDataType>::~NgsConstraintSolver()
	{

	}

	template<typename Coord>
	__global__ void IntegrationVelocity(
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> impulse
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= velocity.size())
			return;

		velocity[tId] += impulse[2 * tId];
		angular_velocity[tId] += impulse[2 * tId + 1];
	}

	template<typename Coord, typename Matrix, typename Quat>
	__global__ void IntegrationGesture(
		DArray<Coord> pos,
		DArray<Quat> rotQuat,
		DArray<Matrix> rotMat,
		DArray<Matrix> inertia,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Matrix> inertia_init,
		Real dt
	)
	{
		int tId = threadIdx.x + blockDim.x * blockIdx.x;
		if (tId >= pos.size())
			return;

		pos[tId] += velocity[tId] * dt;
		rotQuat[tId] += dt * 0.5 * Quat(angular_velocity[tId][0], angular_velocity[tId][1], angular_velocity[tId][2], 0.0) * (rotQuat[tId]);
		
		rotQuat[tId] = rotQuat[tId].normalize();
		
		rotMat[tId] = rotQuat[tId].toMatrix3x3();

		inertia[tId] = rotMat[tId] * inertia_init[tId] * rotMat[tId].inverse();
	}

	template<typename Coord, typename Matrix, typename Constraint>
	__global__ void calculateJacobianAndB(
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Coord> pos,
		DArray<Matrix> inertia,
		DArray<Real> mass,
		DArray<Constraint> constraints,
		DArray<Matrix> rotMat
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= constraints.size())
			return;

		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		if (constraints[tId].type == ConstraintType::CN_NONPENETRATION || constraints[tId].type == ConstraintType::CN_FRICTION)
		{
			Coord n = constraints[tId].normal1;
			Coord r1 = constraints[tId].pos1 - pos[idx1];
			Coord rcn_1 = r1.cross(n);

			J[4 * tId] = -n;
			J[4 * tId + 1] = -rcn_1;
			B[4 * tId] = -n / mass[idx1];
			B[4 * tId + 1] = -inertia[idx1].inverse() * rcn_1;

			if (idx2 != INVALID)
			{
				Coord r2 = constraints[tId].pos2 - pos[idx2];
				Coord rcn_2 = r2.cross(n);
				J[4 * tId + 2] = n;
				J[4 * tId + 3] = rcn_2;
				B[4 * tId + 2] = n / mass[idx2];
				B[4 * tId + 3] = inertia[idx2].inverse() * rcn_2;
			}
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_1)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;

			J[4 * tId] = Coord(-1, 0, 0);
			J[4 * tId + 1] = Coord(0, -r1[2], r1[1]);
			J[4 * tId + 2] = Coord(1, 0, 0);
			J[4 * tId + 3] = Coord(0, r2[2], -r2[1]);

			B[4 * tId] = Coord(-1, 0, 0) / mass[idx1];
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(0, -r1[2], r1[1]);
			B[4 * tId + 2] = Coord(1, 0, 0) / mass[idx2];
			B[4 * tId + 3] = inertia[idx2].inverse() * Coord(0, r2[2], -r2[1]);
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_2)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;

			J[4 * tId] = Coord(0, -1, 0);
			J[4 * tId + 1] = Coord(r1[2], 0, -r1[0]);
			J[4 * tId + 2] = Coord(0, 1, 0);
			J[4 * tId + 3] = Coord(-r2[2], 0, r2[0]);

			B[4 * tId] = Coord(0, -1, 0) / mass[idx1];
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(r1[2], 0, -r1[0]);
			B[4 * tId + 2] = Coord(0, 1, 0) / mass[idx2];
			B[4 * tId + 3] = inertia[idx2].inverse() * Coord(-r2[2], 0, r2[0]);
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_3)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;

			J[4 * tId] = Coord(0, 0, -1);
			J[4 * tId + 1] = Coord(-r1[1], r1[0], 0);
			J[4 * tId + 2] = Coord(0, 0, 1);
			J[4 * tId + 3] = Coord(r2[1], -r2[0], 0);

			B[4 * tId] = Coord(0, 0, -1) / mass[idx1];
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(-r1[1], r1[0], 0);
			B[4 * tId + 2] = Coord(0, 0, 1) / mass[idx2];
			B[4 * tId + 3] = inertia[idx2].inverse() * Coord(r2[1], -r2[0], 0);
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_TRANS_1)
		{
			Coord r1 = constraints[tId].pos1;
			Coord r2 = constraints[tId].pos2;

			Coord n1 = constraints[tId].normal1;

			J[4 * tId] = -n1;
			J[4 * tId + 1] = -(pos[idx2] + r2 - pos[idx1]).cross(n1);
			J[4 * tId + 2] = n1;
			J[4 * tId + 3] = r2.cross(n1);

			B[4 * tId] = -n1 / mass[idx1];
			B[4 * tId + 1] = inertia[idx1].inverse() * J[4 * tId + 1];
			B[4 * tId + 2] = n1 / mass[idx2];
			B[4 * tId + 3] = inertia[idx2].inverse() * J[4 * tId + 3];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_TRANS_2)
		{
			Coord r1 = constraints[tId].pos1;
			Coord r2 = constraints[tId].pos2;

			Coord n2 = constraints[tId].normal2;

			J[4 * tId] = -n2;
			J[4 * tId + 1] = -(pos[idx2] + r2 - pos[idx1]).cross(n2);
			J[4 * tId + 2] = n2;
			J[4 * tId + 3] = r2.cross(n2);

			B[4 * tId] = -n2 / mass[idx1];
			B[4 * tId + 1] = inertia[idx1].inverse() * J[4 * tId + 1];
			B[4 * tId + 2] = n2 / mass[idx2];
			B[4 * tId + 3] = inertia[idx2].inverse() * J[4 * tId + 3];
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_1)
		{
			J[4 * tId] = Coord(0);
			J[4 * tId + 1] = Coord(-1, 0, 0);
			J[4 * tId + 2] = Coord(0);
			J[4 * tId + 3] = Coord(1, 0, 0);

			B[4 * tId] = Coord(0);
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(-1, 0, 0);
			B[4 * tId + 2] = Coord(0);
			B[4 * tId + 3] = inertia[idx2].inverse() * Coord(1, 0, 0);
		}
		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_2)
		{
			J[4 * tId] = Coord(0);
			J[4 * tId + 1] = Coord(0, -1, 0);
			J[4 * tId + 2] = Coord(0);
			J[4 * tId + 3] = Coord(0, 1, 0);

			B[4 * tId] = Coord(0);
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(0, -1, 0);
			B[4 * tId + 2] = Coord(0);
			B[4 * tId + 3] = inertia[idx2].inverse() * Coord(0, 1, 0);
		}
		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_3)
		{
			J[4 * tId] = Coord(0);
			J[4 * tId + 1] = Coord(0, 0, -1);
			J[4 * tId + 2] = Coord(0);
			J[4 * tId + 3] = Coord(0, 0, 1);

			B[4 * tId] = Coord(0);
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(0, 0, -1);
			B[4 * tId + 2] = Coord(0);
			B[4 * tId + 3] = inertia[idx2].inverse() * Coord(0, 0, 1);
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MOTER)
		{
			if (constraints[tId].isValid)
			{
				Coord n = constraints[tId].axis;

				J[4 * tId] = n;
				J[4 * tId + 1] = Coord(0);
				J[4 * tId + 2] = -n;
				J[4 * tId + 3] = Coord(0);

				B[4 * tId] = n / mass[idx1];
				B[4 * tId + 1] = Coord(0);
				B[4 * tId + 2] = -n / mass[idx2];
				B[4 * tId + 3] = Coord(0);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MIN)
		{
			if (constraints[tId].isValid)
			{
				Coord a = constraints[tId].axis;
				Coord r1 = constraints[tId].pos1;
				Coord r2 = constraints[tId].pos2;

				J[4 * tId] = -a;
				J[4 * tId + 1] = -(pos[idx2] + r2 - pos[idx1]).cross(a);
				J[4 * tId + 2] = a;
				J[4 * tId + 3] = r2.cross(a);

				B[4 * tId] = -a / mass[idx1];
				B[4 * tId + 1] = inertia[idx1].inverse() * J[4 * tId + 1];
				B[4 * tId + 2] = a / mass[idx2];
				B[4 * tId + 3] = inertia[idx2].inverse() * J[4 * tId + 3];
			}
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MAX)
		{
			if (constraints[tId].isValid)
			{
				Coord a = constraints[tId].axis;
				Coord r1 = constraints[tId].pos1;
				Coord r2 = constraints[tId].pos2;

				J[4 * tId] = a;
				J[4 * tId + 1] = (pos[idx2] + r2 - pos[idx1]).cross(a);
				J[4 * tId + 2] = -a;
				J[4 * tId + 3] = -r2.cross(a);

				B[4 * tId] = a / mass[idx1];
				B[4 * tId + 1] = inertia[idx1].inverse() * J[4 * tId + 1];
				B[4 * tId + 2] = -a / mass[idx2];
				B[4 * tId + 3] = inertia[idx2].inverse() * J[4 * tId + 3];
			}
		}

		if (constraints[tId].type == ConstraintType::CN_ALLOW_ROT1D_1)
		{
			Coord b2 = constraints[tId].pos1;
			Coord a1 = constraints[tId].axis;

			J[4 * tId] = Coord(0);
			J[4 * tId + 1] = -b2.cross(a1);
			J[4 * tId + 2] = Coord(0);
			J[4 * tId + 3] = b2.cross(a1);

			B[4 * tId] = Coord(0);
			B[4 * tId + 1] = inertia[idx1].inverse() * J[4 * tId + 1];
			B[4 * tId + 2] = Coord(0);
			B[4 * tId + 3] = inertia[idx2].inverse() * J[4 * tId + 3];
		}

		if (constraints[tId].type == ConstraintType::CN_ALLOW_ROT1D_2)
		{
			Coord c2 = constraints[tId].pos2;
			Coord a1 = constraints[tId].axis;

			J[4 * tId] = Coord(0);
			J[4 * tId + 1] = -c2.cross(a1);
			J[4 * tId + 2] = Coord(0);
			J[4 * tId + 3] = c2.cross(a1);

			B[4 * tId] = Coord(0);
			B[4 * tId + 1] = inertia[idx1].inverse() * J[4 * tId + 1];
			B[4 * tId + 2] = Coord(0);
			B[4 * tId + 3] = inertia[idx2].inverse() * J[4 * tId + 3];
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MIN)
		{
			if (constraints[tId].isValid)
			{
				Coord a = constraints[tId].axis;
				J[4 * tId] = Coord(0);
				J[4 * tId + 1] = -a;
				J[4 * tId + 2] = Coord(0);
				J[4 * tId + 3] = a;

				B[4 * tId] = Coord(0);
				B[4 * tId + 1] = inertia[idx1].inverse() * (-a);
				B[4 * tId + 2] = Coord(0);
				B[4 * tId + 3] = inertia[idx2].inverse() * (a);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MAX)
		{
			if (constraints[tId].isValid)
			{
				Coord a = constraints[tId].axis;
				J[4 * tId] = Coord(0);
				J[4 * tId + 1] = a;
				J[4 * tId + 2] = Coord(0);
				J[4 * tId + 3] = -a;

				B[4 * tId] = Coord(0);
				B[4 * tId + 1] = inertia[idx1].inverse() * (a);
				B[4 * tId + 2] = Coord(0);
				B[4 * tId + 3] = inertia[idx2].inverse() * (-a);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MOTER)
		{
			if (constraints[tId].isValid)
			{
				Coord a = constraints[tId].axis;
				J[4 * tId] = Coord(0);
				J[4 * tId + 1] = -a;
				J[4 * tId + 2] = Coord(0);
				J[4 * tId + 3] = a;

				B[4 * tId] = Coord(0);
				B[4 * tId + 1] = inertia[idx1].inverse() * (-a);
				B[4 * tId + 2] = Coord(0);
				B[4 * tId + 3] = inertia[idx2].inverse() * a;
			}
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_1)
		{
			J[4 * tId] = Coord(1.0, 0, 0);
			J[4 * tId + 1] = Coord(0);
			J[4 * tId + 2] = Coord(0);
			J[4 * tId + 3] = Coord(0);

			B[4 * tId] = Coord(1 / mass[idx1], 0, 0);
			B[4 * tId + 1] = Coord(0);
			B[4 * tId + 2] = Coord(0);
			B[4 * tId + 3] = Coord(0);
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_2)
		{
			J[4 * tId] = Coord(0, 1.0, 0);
			J[4 * tId + 1] = Coord(0);
			J[4 * tId + 2] = Coord(0);
			J[4 * tId + 3] = Coord(0);

			B[4 * tId] = Coord(0, 1.0 / mass[idx1], 0);
			B[4 * tId + 1] = Coord(0);
			B[4 * tId + 2] = Coord(0);
			B[4 * tId + 3] = Coord(0);
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_3)
		{
			J[4 * tId] = Coord(0, 0, 1.0);
			J[4 * tId + 1] = Coord(0);
			J[4 * tId + 2] = Coord(0);
			J[4 * tId + 3] = Coord(0);

			B[4 * tId] = Coord(0, 0, 1.0 / mass[idx1]);
			B[4 * tId + 1] = Coord(0);
			B[4 * tId + 2] = Coord(0);
			B[4 * tId + 3] = Coord(0);
		}
	}

	template<typename Coord, typename Real>
	__global__ void calculateDiagonals(
		DArray<Real> D,
		DArray<Coord> J,
		DArray<Coord> B
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= D.size())
			return;
		Real d = J[4 * tId].dot(B[4 * tId]) + J[4 * tId + 1].dot(B[4 * tId + 1]) + J[4 * tId + 2].dot(B[4 * tId + 2]) + J[4 * tId + 3].dot(B[4 * tId + 3]);
		
		D[tId] = d;
	}


	template<typename Coord, typename Constraint, typename Real>
	__global__ void calculateEta(
		DArray<Real> eta,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Constraint> constraints,
		DArray<Coord> J
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= constraints.size())
			return;


		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		Real eta_i = Real(0);

		eta_i -= J[4 * tId].dot(velocity[idx1]);
		eta_i -= J[4 * tId + 1].dot(angular_velocity[idx1]);

		if (idx2 != INVALID)
		{
			eta_i -= J[4 * tId + 2].dot(velocity[idx2]);
			eta_i -= J[4 * tId + 3].dot(angular_velocity[idx2]);
		}

		eta[tId] = eta_i;
	}

	template<typename Contact, typename Constraint>
	__global__ void setUpContactAndFrictionConstraints(
		DArray<Constraint> constraints,
		DArray<Contact> contacts,
		int contact_size,
		bool hasFriction
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contact_size)
			return;


		constraints[tId].bodyId1 = contacts[tId].bodyId1;
		constraints[tId].bodyId2 = contacts[tId].bodyId2;
		constraints[tId].pos1 = contacts[tId].pos1;
		constraints[tId].pos2 = contacts[tId].pos2;
		constraints[tId].normal1 = contacts[tId].normal1;
		constraints[tId].normal2 = contacts[tId].normal2;
		constraints[tId].interpenetration = -contacts[tId].interpenetration;
		constraints[tId].type = ConstraintType::CN_NONPENETRATION;

		printf("%d pos1: (%lf, %lf, %lf)\n", tId, contacts[tId].pos1[0], constraints[tId].pos1[1], constraints[tId].pos1[2]);
		printf("%d pos2: (%lf, %lf, %lf)\n", tId, contacts[tId].pos2[0], constraints[tId].pos2[1], constraints[tId].pos2[2]);
		printf("%d normal1: (%lf, %lf, %lf)\n", tId, contacts[tId].normal1[0], constraints[tId].normal1[1], constraints[tId].normal1[2]);
		printf("%d normal2: (%lf, %lf, %lf)\n", tId, contacts[tId].normal2[0], constraints[tId].normal2[1], constraints[tId].normal2[2]);
		
		if (hasFriction)
		{
			Vector<Real, 3> n = contacts[tId].normal1;
			n = n.normalize();

			Vector<Real, 3> u1, u2;

			if (abs(n[1]) > EPSILON || abs(n[2]) > EPSILON)
			{
				u1 = Vector<Real, 3>(0, n[2], -n[1]);
				u1 = u1.normalize();
			}
			else if (abs(n[0]) > EPSILON)
			{
				u1 = Vector<Real, 3>(n[2], 0, -n[0]);
				u1 = u1.normalize();
			}

			u2 = u1.cross(n);
			u2 = u2.normalize();

			constraints[tId * 2 + contact_size].bodyId1 = contacts[tId].bodyId1;
			constraints[tId * 2 + contact_size].bodyId2 = contacts[tId].bodyId2;
			constraints[tId * 2 + contact_size].pos1 = contacts[tId].pos1;
			constraints[tId * 2 + contact_size].pos2 = contacts[tId].pos2;
			constraints[tId * 2 + contact_size].normal1 = u1;
			constraints[tId * 2 + contact_size].normal2 = -u1;
			constraints[tId * 2 + contact_size].type = ConstraintType::CN_FRICTION;

			constraints[tId * 2 + 1 + contact_size].bodyId1 = contacts[tId].bodyId1;
			constraints[tId * 2 + 1 + contact_size].bodyId2 = contacts[tId].bodyId2;
			constraints[tId * 2 + 1 + contact_size].pos1 = contacts[tId].pos1;
			constraints[tId * 2 + 1 + contact_size].pos2 = contacts[tId].pos2;
			constraints[tId * 2 + 1 + contact_size].normal1 = u2;
			constraints[tId * 2 + 1 + contact_size].normal2 = -u2;
			constraints[tId * 2 + 1 + contact_size].type = ConstraintType::CN_FRICTION;
		}
	}

	template<typename Joint, typename Constraint, typename Coord, typename Matrix>
	__global__ void setUpBallAndSocketJointConstraints(
		DArray<Constraint> constraints,
		DArray<Joint> joints,
		DArray<Coord> pos,
		DArray<Matrix> rotMat,
		int begin_index
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= joints.size())
			return;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;

		Coord r1 = rotMat[idx1] * joints[tId].r1;
		Coord r2 = rotMat[idx2] * joints[tId].r2;

		int baseIndex = 3 * tId + begin_index;
		
		constraints[baseIndex].bodyId1 = idx1;
		constraints[baseIndex].bodyId2 = idx2;
		constraints[baseIndex].normal1 = r1;
		constraints[baseIndex].normal2 = r2;
		constraints[baseIndex].type = ConstraintType::CN_ANCHOR_EQUAL_1;

		constraints[baseIndex + 1].bodyId1 = idx1;
		constraints[baseIndex + 1].bodyId2 = idx2;
		constraints[baseIndex + 1].normal1 = r1;
		constraints[baseIndex + 1].normal2 = r2;
		constraints[baseIndex + 1].type = ConstraintType::CN_ANCHOR_EQUAL_2;

		constraints[baseIndex + 2].bodyId1 = idx1;
		constraints[baseIndex + 2].bodyId2 = idx2;
		constraints[baseIndex + 2].normal1 = r1;
		constraints[baseIndex + 2].normal2 = r2;
		constraints[baseIndex + 2].type = ConstraintType::CN_ANCHOR_EQUAL_3;
	}


	template<typename Coord, typename Constraint>
	__global__ void takeOneJacobiIteration(
		DArray<Real> lambda,
		DArray<Coord> impulse,
		DArray<Real> d,
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Real> eta,
		DArray<Real> mass,
		DArray<Constraint> constraints,
		Real mu,
		Real g,
		Real dt,
		int isPositionIteration
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= constraints.size())
			return;

		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		if (constraints[tId].type == ConstraintType::CN_FRICTION && isPositionIteration)
			return;

		Real tmp = eta[tId];

		tmp -= J[4 * tId].dot(impulse[idx1 * 2]);
		tmp -= J[4 * tId + 1].dot(impulse[idx1 * 2 + 1]);

		if (idx2 != INVALID)
		{
			tmp -= J[4 * tId + 2].dot(impulse[idx2 * 2]);
			tmp -= J[4 * tId + 3].dot(impulse[idx2 * 2 + 1]);
		}

		if (d[tId] > EPSILON)
		{
			int stepInverse = 10;
			Real delta_lambda = tmp / (d[tId] * stepInverse);
			Real lambda_new = lambda[tId] + delta_lambda;


			// Projection to Bound
			if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
			{
				if (lambda_new < 0)
				{
					lambda_new = 0;
					delta_lambda = lambda_new - lambda[tId];
				}
			}

			if (constraints[tId].type == ConstraintType::CN_FRICTION)
			{
				Real mass_avl;
				if (idx2 != INVALID)
				{
					mass_avl = (mass[idx1] + mass[idx2]) / 2;
				}
				else
				{
					mass_avl = mass[idx1];
				}

				lambda_new = (abs(lambda_new) > mu * mass_avl * g * dt) ? (lambda_new < 0 ? -mu * mass_avl * g * dt : mu * mass_avl * g * dt) : lambda_new;
				delta_lambda = lambda_new - lambda[tId];
			}

			lambda[tId] += delta_lambda;

			atomicAdd(&impulse[idx1 * 2][0], B[4 * tId][0] * delta_lambda);
			atomicAdd(&impulse[idx1 * 2][1], B[4 * tId][1] * delta_lambda);
			atomicAdd(&impulse[idx1 * 2][2], B[4 * tId][2] * delta_lambda);

			atomicAdd(&impulse[idx1 * 2 + 1][0], B[4 * tId + 1][0] * delta_lambda);
			atomicAdd(&impulse[idx1 * 2 + 1][1], B[4 * tId + 1][1] * delta_lambda);
			atomicAdd(&impulse[idx1 * 2 + 1][2], B[4 * tId + 1][2] * delta_lambda);

			if (idx2 != INVALID)
			{
				atomicAdd(&impulse[idx2 * 2][0], B[4 * tId + 2][0] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2][1], B[4 * tId + 2][1] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2][2], B[4 * tId + 2][2] * delta_lambda);

				atomicAdd(&impulse[idx2 * 2 + 1][0], B[4 * tId + 3][0] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2 + 1][1], B[4 * tId + 3][1] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2 + 1][2], B[4 * tId + 3][2] * delta_lambda);
			}
		}
	}

	template<typename TDataType>
	void NgsConstraintSolver<TDataType>::initializeJacobian(Real dt, int isPosition)
	{
		int constraint_size = 0;
		int contact_size = this->inContacts()->size();

		int ballAndSocketJoint_size = this->inBallAndSocketJoints()->size();
		
		if (this->varFrictionEnabled()->getData())
		{
			constraint_size += 3 * contact_size;
		}
		else
		{
			constraint_size += contact_size;
		}

		if (ballAndSocketJoint_size != 0)
		{
			constraint_size += 3 * ballAndSocketJoint_size;
		}


		if (constraint_size == 0)
		{
			return;
		}

		mAllConstraints.resize(constraint_size);

		if (contact_size != 0)
		{
			auto& contacts = this->inContacts()->getData();
			std::cout << "begin" << std::endl;
			std::cout << contacts.size() << std::endl;
			std::cout << constraint_size << std::endl;
			cuExecute(contact_size,
				setUpContactAndFrictionConstraints,
				mAllConstraints,
				contacts,
				contact_size,
				this->varFrictionEnabled()->getData());
			std::cout << "end" << std::endl;
		}

		if (ballAndSocketJoint_size != 0)
		{
			auto& joints = this->inBallAndSocketJoints()->getData();
			int begin_index = contact_size;
			if (this->varFrictionEnabled()->getData())
			{
				begin_index += 2 * contact_size;
			}
			cuExecute(ballAndSocketJoint_size,
				setUpBallAndSocketJointConstraints,
				mAllConstraints,
				joints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				begin_index);
		}

		mJ.resize(4 * constraint_size);
		mB.resize(4 * constraint_size);
		mD.resize(constraint_size);
		mEta.resize(constraint_size);
		mLambda.resize(constraint_size);
		
		mJ.reset();
		mB.reset();
		mD.reset();
		mEta.reset();
		mLambda.reset();

		cuExecute(constraint_size,
			calculateJacobianAndB,
			mJ,
			mB,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			mAllConstraints,
			this->inRotationMatrix()->getData());

		cuExecute(constraint_size,
			calculateDiagonals,
			mD,
			mJ,
			mB);

		cuExecute(constraint_size,
			calculateEta,
			mEta,
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			mAllConstraints,
			mJ
		);
	}


	template<typename Coord>
	__global__ void setUpGravity(
		DArray<Coord> impulse_ext,
		Real g,
		Real dt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= impulse_ext.size() / 2)
			return;

		impulse_ext[2 * tId] = Coord(0, -g, 0) * dt;
		impulse_ext[2 * tId + 1] = Coord(0);
	}

	template<typename TDataType>
	void NgsConstraintSolver<TDataType>::constrain()
	{
		int bodyNum = this->inCenter()->size();

		mImpulseC.resize(bodyNum * 2);
		mImpulseExt.resize(bodyNum * 2);
		mImpulseC.reset();
		mImpulseExt.reset();

		Real dt = this->inTimeStep()->getData();

		if (this->varGravityEnabled()->getData())
		{
			cuExecute(bodyNum,
				setUpGravity,
				mImpulseExt,
				this->varGravityValue()->getData(),
				dt);
		}

		cuExecute(bodyNum,
			IntegrationVelocity,
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			mImpulseExt);

		if (!this->inContacts()->isEmpty() || !this->inBallAndSocketJoints()->isEmpty())
		{
			initializeJacobian(dt, 0);


			int constraint_size = mAllConstraints.size();

			for (int i = 0; i < this->varVelocityIterationNumber()->getData(); i++)
			{
				cuExecute(constraint_size,
					takeOneJacobiIteration,
					mLambda,
					mImpulseC,
					mD,
					mJ,
					mB,
					mEta,
					this->inMass()->getData(),
					mAllConstraints,
					this->varFrictionCoefficient()->getData(),
					this->varGravityValue()->getData(),
					dt,
					0);
			}
		}

		cuExecute(bodyNum,
			IntegrationVelocity,
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			mImpulseC);

		cuExecute(bodyNum,
			IntegrationGesture,
			this->inCenter()->getData(),
			this->inQuaternion()->getData(),
			this->inRotationMatrix()->getData(),
			this->inInertia()->getData(),
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			this->inInitialInertia()->getData(),
			dt);

	}
	DEFINE_CLASS(NgsConstraintSolver);
}