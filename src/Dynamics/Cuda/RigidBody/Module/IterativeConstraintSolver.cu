#include "IterativeConstraintSolver.h"

namespace dyno
{
	IMPLEMENT_TCLASS(IterativeConstraintSolver, TDataType)

	template<typename TDataType>
	IterativeConstraintSolver<TDataType>::IterativeConstraintSolver()
		: ConstraintModule()
	{
		this->inContacts()->tagOptional(true);
	}

	template<typename TDataType>
	IterativeConstraintSolver<TDataType>::~IterativeConstraintSolver()
	{
	}

	template<typename Coord>
	__global__ void RB_updateVelocity(
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> impulse_ext,
		DArray<Coord> impulse_constrain
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= velocity.size())
			return;

		velocity[tId] += impulse_ext[2 * tId] + impulse_constrain[2 * tId];
		angular_velocity[tId] += impulse_ext[2 * tId + 1] + impulse_constrain[2 * tId + 1];
	}

	template<typename Coord, typename Matrix, typename Quat>
	__global__ void RB_updateGesture(
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
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= pos.size())
			return;

		pos[tId] += velocity[tId] * dt;

		rotQuat[tId] = rotQuat[tId].normalize();

		rotQuat[tId] += dt * 0.5f *
			Quat(angular_velocity[tId][0], angular_velocity[tId][1], angular_velocity[tId][2], 0.0)
			* (rotQuat[tId]);

		rotQuat[tId] = rotQuat[tId].normalize();

		rotMat[tId] = rotQuat[tId].toMatrix3x3();

		inertia[tId] = rotMat[tId] * inertia_init[tId] * rotMat[tId].inverse();
	}

	template<typename ContactPair>
	__global__ void calculateNbrCons(
		DArray<ContactPair> nbc,
		DArray<int> nbrCnt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nbc.size())
			return;

		int idx1 = nbc[tId].bodyId1;
		int idx2 = nbc[tId].bodyId2;

		atomicAdd(&nbrCnt[idx1], 1);

		if (idx2 != INVALID)
			atomicAdd(&nbrCnt[idx2], 1);
	}

	template<typename Joint>
	__global__ void calculateJoints(
		DArray<Joint> joints,
		DArray<int> jointCnt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= joints.size())
			return;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;

		atomicAdd(&jointCnt[idx1], 1);
		atomicAdd(&jointCnt[idx2], 1);
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
			if (constraints[tId].isValid == 1)
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
			if (constraints[tId].isValid == 1)
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

			B[4 * tId] = Coord(0, 1.0/mass[idx1], 0);
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

			B[4 * tId] = Coord(0, 0, 1.0/mass[idx1]);
			B[4 * tId + 1] = Coord(0);
			B[4 * tId + 2] = Coord(0);
			B[4 * tId + 3] = Coord(0);
		}

		
	}

	template<typename Coord>
	__global__ void calculateDiagonals(
		DArray<Real> D,
		DArray<Coord> J,
		DArray<Coord> B
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= D.size())
			return;

		Real d = J[4 * tId].dot(B[4 * tId]) + J[4 * tId + 1].dot(B[4 * tId + 1]) + J[4 * tId + 2].dot(B[4 * tId + 2]) + J[4 * tId + 3].dot(B[4 * tId + 3]);

		D[tId] = d;
	}

	

	template<typename Coord, typename Constraint, typename Quat>
	__global__ void calculateEta(
		DArray<Real> eta,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> impulse_ext,
		DArray<Coord> J,
		DArray<Real> mass,
		DArray<Coord> pos,
		DArray<Constraint> constraints,
		DArray<Quat> rotation_q,
		Real slop,
		Real dt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= constraints.size())
			return;

		Real invDt = Real(1) / dt;

		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		Real eta_i = Real(0);

		eta_i -= J[4 * tId].dot(velocity[idx1] + impulse_ext[idx1 * 2]);
		eta_i -= J[4 * tId + 1].dot(angular_velocity[idx1] + impulse_ext[idx1 * 2 + 1]);

		if (idx2 != INVALID)
		{
			eta_i -= J[4 * tId + 2].dot(velocity[idx2] + impulse_ext[idx2 * 2]);
			eta_i -= J[4 * tId + 3].dot(angular_velocity[idx2] + impulse_ext[idx2 * 2 + 1]);
		}

		eta[tId] = eta_i;

		if (constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MOTER || constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MOTER)
		{
			Real v_moter = constraints[tId].interpenetration;
			eta[tId] -= v_moter;
		}
	}


	template<typename Real>
	__global__ void calculateDiff(
		DArray<Real> lambda,
		DArray<Real> lambda_old,
		DArray<Real> diff
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= lambda.size())
			return;
		diff[tId] = lambda[tId] - lambda_old[tId];
	}

	template<typename Real>
	Real calculateNorm(
		CArray<Real> diff
	)
	{
		Real norm = 0.0;
		for (int i = 0; i < diff.size(); i++)
		{
			if (abs(diff[i]) > norm)
				norm = abs(diff[i]);
		}
		return norm;
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
		constraints[tId].normal1 = -contacts[tId].normal1;
		constraints[tId].normal2 = -contacts[tId].normal2;
		constraints[tId].interpenetration = -contacts[tId].interpenetration;
		constraints[tId].type = ConstraintType::CN_NONPENETRATION;

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

	template<typename Coord, typename Matrix, typename Contact, typename Constraint>
	__global__ void ICS_setUpContactAndFrictionConstraints(
		DArray<Constraint> constraints,
		DArray<Contact> contactsInLocalFrame,
		DArray<Coord> center,
		DArray<Matrix> rot,
		bool hasFriction
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contactsInLocalFrame.size())
			return;

		int contact_size = contactsInLocalFrame.size();

		int idx1 = contactsInLocalFrame[tId].bodyId1;
		int idx2 = contactsInLocalFrame[tId].bodyId2;

		Coord c1 = center[idx1];
		Matrix rot1 = rot[idx1];

		constraints[tId].bodyId1 = idx1;
		constraints[tId].bodyId2 = idx2;
		constraints[tId].pos1 = rot1 * contactsInLocalFrame[tId].pos1 + c1;
		constraints[tId].normal1 = -rot1 * contactsInLocalFrame[tId].normal1;
		if (idx2 != INVALID)
		{
			Coord c2 = center[idx2];
			Matrix rot2 = rot[idx2];

			constraints[tId].pos2 = rot2 * contactsInLocalFrame[tId].pos2 + c2;
			constraints[tId].normal2 = -rot2 * contactsInLocalFrame[tId].normal2;
		}

		constraints[tId].interpenetration = -contactsInLocalFrame[tId].interpenetration;
		constraints[tId].type = ConstraintType::CN_NONPENETRATION;

		if (hasFriction)
		{
			Vector<Real, 3> n = constraints[tId].normal1;
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

			constraints[tId * 2 + contact_size].bodyId1 = idx1;
			constraints[tId * 2 + contact_size].bodyId2 = idx2;
			constraints[tId * 2 + contact_size].pos1 = constraints[tId].pos1;
			constraints[tId * 2 + contact_size].pos2 = constraints[tId].pos2;
			constraints[tId * 2 + contact_size].normal1 = u1;
			constraints[tId * 2 + contact_size].normal2 = -u1;
			constraints[tId * 2 + contact_size].type = ConstraintType::CN_FRICTION;

			constraints[tId * 2 + 1 + contact_size].bodyId1 = idx1;
			constraints[tId * 2 + 1 + contact_size].bodyId2 = idx2;
			constraints[tId * 2 + 1 + contact_size].pos1 = constraints[tId].pos1;
			constraints[tId * 2 + 1 + contact_size].pos2 = constraints[tId].pos2;
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
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

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

	template<typename Joint, typename Constraint, typename Coord, typename Matrix>
	__global__ void setUpSliderJoint(
		DArray<Joint> joints,
		DArray<Constraint> constraints,
		DArray<Coord> pos,
		DArray<Matrix> rotMat,
		int begin_index
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= joints.size())
			return;

		int constraint_size = 8;
		
		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;

		Coord r1 = rotMat[idx1] * joints[tId].r1;
		Coord r2 = rotMat[idx2] * joints[tId].r2;

		Coord n = rotMat[idx1] * joints[tId].sliderAxis;
		n = n.normalize();
		Coord n1, n2;
		if (abs(n[1]) > EPSILON || abs(n[2]) > EPSILON)
		{
			n1 = Vector<Real, 3>(0, n[2], -n[1]);
			n1 = n1.normalize();
		}
		else if (abs(n[0]) > EPSILON)
		{
			n1 = Vector<Real, 3>(n[2], 0, -n[0]);
			n1 = n1.normalize();
		}
		n2 = n1.cross(n);
		n2 = n2.normalize();

		int baseIndex = constraint_size * tId + begin_index;

		bool useRange = joints[tId].useRange;
		Real C_min = 0.0;
		Real C_max = 0.0;
		if (joints[tId].useRange)
		{
			Coord u = pos[idx2] + r2 - pos[idx1] - r1;
			C_min = u.dot(n) - joints[tId].d_min;
			C_max = joints[tId].d_max - u.dot(n);
			if (C_min < 0)
				constraints[baseIndex + 5].isValid = true;
			else
				constraints[baseIndex + 5].isValid = false;
			if (C_max < 0)
				constraints[baseIndex + 6].isValid = true;
			else
				constraints[baseIndex + 6].isValid = false;
		}

		Real v_moter = 0.0;
		bool useMoter = joints[tId].useMoter;
		if (useMoter)
		{
			v_moter = joints[tId].v_moter;
			constraints[baseIndex + 7].isValid = true;
		}
		else
		{
			constraints[baseIndex + 7].isValid = false;
		}

		for (int i = 0; i < constraint_size; i++)
		{
			auto& constraint = constraints[baseIndex + i];
			constraint.bodyId1 = idx1;
			constraint.bodyId2 = idx2;
			constraint.pos1 = r1;
			constraint.pos2 = r2;
			constraint.normal1 = n1;
			constraint.normal2 = n2;
			constraint.axis = n;
			constraint.interpenetration = v_moter;
			constraint.d_min = C_min;
			constraint.d_max = C_max;
		}
		
		constraints[baseIndex].type = ConstraintType::CN_ANCHOR_TRANS_1;
		constraints[baseIndex + 1].type = ConstraintType::CN_ANCHOR_TRANS_2;
		constraints[baseIndex + 2].type = ConstraintType::CN_BAN_ROT_1;
		constraints[baseIndex + 3].type = ConstraintType::CN_BAN_ROT_2;
		constraints[baseIndex + 4].type = ConstraintType::CN_BAN_ROT_3;
		constraints[baseIndex + 5].type = ConstraintType::CN_JOINT_SLIDER_MIN;
		constraints[baseIndex + 6].type = ConstraintType::CN_JOINT_SLIDER_MAX;
		constraints[baseIndex + 7].type = ConstraintType::CN_JOINT_SLIDER_MOTER;
	}

	template<typename Joint, typename Constraint, typename Coord, typename Matrix, typename Quat>
	__global__ void setUpHingeJoint(
		DArray<Joint> joints,
		DArray<Constraint> constraints,
		DArray<Coord> pos,
		DArray<Matrix> rotMat,
		DArray<Quat> rotation_q,
		int begin_index
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= joints.size())
			return;

		int constraint_size = 8;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;

		Matrix rotMat1 = rotMat[idx1];
		Matrix rotMat2 = rotMat[idx2];


		Coord r1 = rotMat1 * joints[tId].r1;
		Coord r2 = rotMat2 * joints[tId].r2;

		Coord a1 = rotMat1 * joints[tId].hingeAxisBody1;
		Coord a2 = rotMat2 * joints[tId].hingeAxisBody2;



		// two vector orthogonal to the a2
		Coord b2, c2;
		if (abs(a2[1]) > EPSILON || abs(a2[2]) > EPSILON)
		{
			b2 = Coord(0, a2[2], -a2[1]);
			b2 = b2.normalize();
		}
		else if (abs(a2[0]) > EPSILON)
		{
			b2 = Coord(a2[2], 0, -a2[0]);
			b2 = b2.normalize();
		}
		c2 = b2.cross(a2);
		c2 = c2.normalize();

		Real C_min = 0.0;
		Real C_max = 0.0;
		int baseIndex = tId * constraint_size + begin_index;

		if (joints[tId].useRange)
		{
			Real theta = rotation_q[idx2].angle(rotation_q[idx1]);

			Quat q_rot = rotation_q[idx2] * rotation_q[idx1].inverse();

			if (a1.dot(Coord(q_rot.x, q_rot.y, q_rot.z)) < 0)
			{
				theta = -theta;
			}
		
			
			C_min = theta - joints[tId].d_min;
			C_max = joints[tId].d_max - theta;

			if (C_min < 0)
			{
				constraints[baseIndex + 5].isValid = true;
			}
			else
				constraints[baseIndex + 5].isValid = false;

			if (C_max < 0)
			{
				constraints[baseIndex + 6].isValid = true;
			}
			else
				constraints[baseIndex + 6].isValid = false;
		}

		Real v_moter = 0.0;
		if (joints[tId].useMoter)
		{
			v_moter = joints[tId].v_moter;
			constraints[baseIndex + 7].isValid = true;
		}
		else
			constraints[baseIndex + 7].isValid = false;

		for (int i = 0; i < constraint_size; i++)
		{
			constraints[baseIndex + i].bodyId1 = idx1;
			constraints[baseIndex + i].bodyId2 = idx2;
			constraints[baseIndex + i].axis = a1;
			constraints[baseIndex + i].normal1 = r1;
			constraints[baseIndex + i].normal2 = r2;
			constraints[baseIndex + i].pos1 = b2;
			constraints[baseIndex + i].pos2 = c2;
			constraints[baseIndex + i].d_min = C_min > 0 ? 0 : C_min;
			constraints[baseIndex + i].d_max = C_max > 0 ? 0 : C_max;
			constraints[baseIndex + i].interpenetration = v_moter;
		}
		
		constraints[baseIndex].type = ConstraintType::CN_ANCHOR_EQUAL_1;
		constraints[baseIndex + 1].type = ConstraintType::CN_ANCHOR_EQUAL_2;
		constraints[baseIndex + 2].type = ConstraintType::CN_ANCHOR_EQUAL_3;
		constraints[baseIndex + 3].type = ConstraintType::CN_ALLOW_ROT1D_1;
		constraints[baseIndex + 4].type = ConstraintType::CN_ALLOW_ROT1D_2;
		constraints[baseIndex + 5].type = ConstraintType::CN_JOINT_HINGE_MIN;
		constraints[baseIndex + 6].type = ConstraintType::CN_JOINT_HINGE_MAX;
		constraints[baseIndex + 7].type = ConstraintType::CN_JOINT_HINGE_MOTER;

	}

	template<typename Joint, typename Constraint, typename Coord>
	__global__ void setUpPointJoint(
		DArray<Joint> joints,
		DArray<Constraint> constraints,
		DArray<Coord> pos,
		int begin_index
	)
	{
		int tId = threadIdx.x + blockDim.x * blockIdx.x;
		if (tId >= joints.size())
			return;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;

		int baseIndex = 3 * tId + begin_index;

		for (int i = 0; i < 3; i++)
		{
			constraints[baseIndex + i].bodyId1 = idx1;
			constraints[baseIndex + i].bodyId2 = idx2;
			constraints[baseIndex + i].normal1 = pos[idx1] - joints[tId].anchorPoint;
		}

		constraints[baseIndex].type = ConstraintType::CN_JOINT_NO_MOVE_1;
		constraints[baseIndex + 1].type = ConstraintType::CN_JOINT_NO_MOVE_2;
		constraints[baseIndex + 2].type = ConstraintType::CN_JOINT_NO_MOVE_3;

	}

	template<typename Joint, typename Constraint, typename Matrix>
	__global__ void setUpFixedJoint(
		DArray<Joint> joints,
		DArray<Constraint> constraints,
		DArray<Matrix> rotMat,
		int begin_index
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= joints.size())
			return;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;
		Vector<Real, 3> r1 = rotMat[idx1] * joints[tId].r1;
		Vector<Real, 3> r2 = rotMat[idx2] * joints[tId].r2;

		int baseIndex = 6 * tId + begin_index;
		for (int i = 0; i < 6; i++)
		{
			constraints[baseIndex + i].bodyId1 = idx1;
			constraints[baseIndex + i].bodyId2 = idx2;
			constraints[baseIndex + i].normal1 = r1;
			constraints[baseIndex + i].normal2 = r2;
		}

		constraints[baseIndex].type = ConstraintType::CN_ANCHOR_EQUAL_1;
		constraints[baseIndex + 1].type = ConstraintType::CN_ANCHOR_EQUAL_2;
		constraints[baseIndex + 2].type = ConstraintType::CN_ANCHOR_EQUAL_3;
		constraints[baseIndex + 3].type = ConstraintType::CN_BAN_ROT_1;
		constraints[baseIndex + 4].type = ConstraintType::CN_BAN_ROT_2;
		constraints[baseIndex + 5].type = ConstraintType::CN_BAN_ROT_3;
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
		DArray<int> nbq,
		DArray<int> jointNumber,
		Real mu,
		Real g,
		Real dt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= constraints.size())
			return;

		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		double tmp = eta[tId];

		tmp -= J[4 * tId].dot(impulse[idx1 * 2]);
		tmp -= J[4 * tId + 1].dot(impulse[idx1 * 2 + 1]);

		if (idx2 != INVALID)
		{
			tmp -= J[4 * tId + 2].dot(impulse[idx2 * 2]);
			tmp -= J[4 * tId + 3].dot(impulse[idx2 * 2 + 1]);
		}


		if (d[tId] > EPSILON)
		{
			int stepInverse = 0;
			if (constraints[tId].type == ConstraintType::CN_FRICTION || constraints[tId].type == ConstraintType::CN_NONPENETRATION)
			{
				if (idx2 != INVALID)
				{
					stepInverse = nbq[idx1] > nbq[idx2] ? nbq[idx1] : nbq[idx2];
				}
				else
				{
					stepInverse = nbq[idx1];
				}	
			}

			else
			{
				if (idx2 != INVALID)
				{
					stepInverse += 4 * jointNumber[idx2];
				}

				stepInverse += 4 * jointNumber[idx1];
			}

			stepInverse = stepInverse < 1 ? 1 : stepInverse;

			double delta_lambda = (tmp / (d[tId] * stepInverse));

			double lambda_new = lambda[tId] + delta_lambda;

			// Projection to Bound
			if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
			{
				if (lambda_new < 0)
				{
					lambda_new = 0;
					delta_lambda = lambda_new - lambda[tId];
				}
			}

			/*if (constraints[tId].type == ConstraintType::CN_FRICTION)
			{
				Real mass_avl = mass[idx1];
				lambda_new = (abs(lambda_new) > mu *mass_avl * g * dt) ? (lambda_new < 0 ? -mu * mass_avl * g * dt: mu *mass_avl* g * dt) : lambda_new;
				delta_lambda = lambda_new - lambda[tId];
			}*/

			
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
	void IterativeConstraintSolver<TDataType>::initializeJacobian(Real dt)
	{
		int constraint_size = 0;
		int contact_size = this->inContacts()->size();

		auto topo = this->inDiscreteElements()->constDataPtr();

		int ballAndSocketJoint_size = topo->ballAndSocketJoints().size();
		int sliderJoint_size = topo->sliderJoints().size();
		int hingeJoint_size = topo->hingeJoints().size();
		int fixedJoint_size = topo->fixedJoints().size();
		int pointJoint_size = topo->pointJoints().size();

		if (this->varFrictionEnabled()->getData())
		{
			constraint_size += 3 * contact_size;
		}
		else
		{
			constraint_size = contact_size;
		}

		if (ballAndSocketJoint_size != 0)
		{
			constraint_size += 3 * ballAndSocketJoint_size;
		}

		if (sliderJoint_size != 0)
		{
			constraint_size += 8 * sliderJoint_size;
		}

		if (hingeJoint_size != 0)
		{
			constraint_size += 8 * hingeJoint_size;
		}

		if (fixedJoint_size != 0)
		{
			constraint_size += 6 * fixedJoint_size;
		}

		if (pointJoint_size != 0)
		{
			constraint_size += 3 * pointJoint_size;
		}




		if (constraint_size == 0)
		{
			return;
		}

		mVelocityConstraints.resize(constraint_size);


		if (contact_size != 0)
		{
			auto& contacts = this->inContacts()->getData();
			cuExecute(contact_size,
				ICS_setUpContactAndFrictionConstraints,
				mVelocityConstraints,
				mContactsInLocalFrame,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				this->varFrictionEnabled()->getData());
		}

		if (ballAndSocketJoint_size != 0)
		{
			auto& joints = topo->ballAndSocketJoints();
			int begin_index = contact_size;

			if (this->varFrictionEnabled()->getData())
			{
				begin_index += 2 * contact_size;
			}

			cuExecute(ballAndSocketJoint_size,
				setUpBallAndSocketJointConstraints,
				mVelocityConstraints,
				joints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				begin_index);
		}

		if (sliderJoint_size != 0)
		{
			auto& joints = topo->sliderJoints();
			int begin_index = contact_size;

			if (this->varFrictionEnabled()->getData())
			{
				begin_index += 2 * contact_size;
			}
			begin_index += 3 * ballAndSocketJoint_size;

			cuExecute(sliderJoint_size,
				setUpSliderJoint,
				joints,
				mVelocityConstraints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				begin_index);
		}

		if (hingeJoint_size != 0)
		{
			auto& joints = topo->hingeJoints();
			int begin_index = contact_size + 3 * ballAndSocketJoint_size + 8 * sliderJoint_size;
			if (this->varFrictionEnabled()->getData())
			{
				begin_index += 2 * contact_size;
			}
			cuExecute(hingeJoint_size,
				setUpHingeJoint,
				joints,
				mVelocityConstraints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				this->inQuaternion()->getData(),
				begin_index);
		}

		if (fixedJoint_size != 0)
		{
			auto& joints = topo->fixedJoints();
			int begin_index = contact_size + 3 * ballAndSocketJoint_size + 8 * sliderJoint_size + 8 * hingeJoint_size;
			if (this->varFrictionEnabled()->getData())
			{
				begin_index += 2 * contact_size;
			}
			cuExecute(fixedJoint_size,
				setUpFixedJoint,
				joints,
				mVelocityConstraints,
				this->inRotationMatrix()->getData(),
				begin_index);
		}
		if (pointJoint_size != 0)
		{
			auto& joints = topo->pointJoints();
			int begin_index = contact_size + 3 * ballAndSocketJoint_size + 8 * sliderJoint_size + 8 * hingeJoint_size + 6 * fixedJoint_size;
			if (this->varFrictionEnabled()->getData())
			{
				begin_index += 2 * contact_size;
			}
			cuExecute(pointJoint_size,
				setUpPointJoint,
				joints,
				mVelocityConstraints,
				this->inCenter()->getData(),
				begin_index);
		}



		auto sizeOfRigids = this->inCenter()->size();
		mContactNumber.resize(sizeOfRigids);
		mJointNumber.resize(sizeOfRigids);


		mJ.resize(4 * constraint_size);
		mB.resize(4 * constraint_size);
		mD.resize(constraint_size);
		mEta.resize(constraint_size);
		mLambda.resize(constraint_size);
		mLambda_old.resize(constraint_size);

		mJ.reset();
		mB.reset();
		mD.reset();
		mEta.reset();
		mLambda.reset();
		mLambda_old.reset();
		mContactNumber.reset();
		mJointNumber.reset();

		if (contact_size != 0)
		{
			cuExecute(contact_size,
				calculateNbrCons,
				this->inContacts()->getData(),
				mContactNumber);
		}

		if (ballAndSocketJoint_size != 0)
		{
			cuExecute(ballAndSocketJoint_size,
				calculateJoints,
				topo->ballAndSocketJoints(),
				mJointNumber);
		}

		if (sliderJoint_size != 0)
		{
			cuExecute(sliderJoint_size,
				calculateJoints,
				topo->sliderJoints(),
				mJointNumber);
		}

		if (hingeJoint_size != 0)
		{
			cuExecute(hingeJoint_size,
				calculateJoints,
				topo->hingeJoints(),
				mJointNumber);
		}

		if (fixedJoint_size != 0)
		{
			cuExecute(fixedJoint_size,
				calculateJoints,
				topo->fixedJoints(),
				mJointNumber);
		}

		cuExecute(constraint_size,
			calculateJacobianAndB,
			mJ,
			mB,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			mVelocityConstraints,
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
			mImpulseExt,
			mJ,
			this->inMass()->getData(),
			this->inCenter()->getData(),
			mVelocityConstraints,
			this->inQuaternion()->getData(),
			this->varSlop()->getData(),
			dt);
	}

	template<typename TDataType>
	void IterativeConstraintSolver<TDataType>::initializeJacobianForNGS(Real dt)
	{
		int constraint_size = 0;
		int contact_size = this->inContacts()->size();

		auto topo = this->inDiscreteElements()->constDataPtr();

		int ballAndSocketJoint_size = topo->ballAndSocketJoints().size();
		int sliderJoint_size = topo->sliderJoints().size();
		int hingeJoint_size = topo->hingeJoints().size();
		int fixedJoint_size = topo->fixedJoints().size();
		int pointJoint_size = topo->pointJoints().size();

		constraint_size = contact_size;

		if (ballAndSocketJoint_size != 0)
		{
			constraint_size += 3 * ballAndSocketJoint_size;
		}

		if (sliderJoint_size != 0)
		{
			constraint_size += 7 * sliderJoint_size;
		}

		if (hingeJoint_size != 0)
		{
			constraint_size += 7 * hingeJoint_size;
		}

		if (fixedJoint_size != 0)
		{
			constraint_size += 6 * fixedJoint_size;
		}

		if (pointJoint_size != 0)
		{
			constraint_size += 3 * pointJoint_size;
		}

		if (constraint_size == 0)
		{
			return;
		}

		mPositionConstraints.resize(constraint_size);

		if (contact_size != 0)
		{
			auto& contacts = this->inContacts()->getData();
			cuExecute(contact_size,
				NGS_setUpContactConstraints,
				mPositionConstraints,
				mContactsInLocalFrame,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData());
		}

		if (ballAndSocketJoint_size != 0)
		{
			auto& joints = topo->ballAndSocketJoints();
			int begin_index = contact_size;

			cuExecute(ballAndSocketJoint_size,
				NGS_setUpBallAndSocketJointConstraints,
				mPositionConstraints,
				joints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				begin_index);
		}

		if (sliderJoint_size != 0)
		{
			auto& joints = topo->sliderJoints();
			int begin_index = contact_size;

			begin_index += 3 * ballAndSocketJoint_size;

			cuExecute(sliderJoint_size,
				NGS_setUpSliderJoint,
				joints,
				mPositionConstraints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				begin_index);
		}

		if (hingeJoint_size != 0)
		{
			auto& joints = topo->hingeJoints();
			int begin_index = contact_size + 3 * ballAndSocketJoint_size + 7 * sliderJoint_size;

			cuExecute(hingeJoint_size,
				NGS_setUpHingeJoint,
				joints,
				mPositionConstraints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				this->inQuaternion()->getData(),
				begin_index);
		}

		if (fixedJoint_size != 0)
		{
			auto& joints = topo->fixedJoints();
			int begin_index = contact_size + 3 * ballAndSocketJoint_size + 7 * sliderJoint_size + 7 * hingeJoint_size;

			cuExecute(fixedJoint_size,
				NGS_setUpFixedJoint,
				joints,
				mPositionConstraints,
				this->inRotationMatrix()->getData(),
				begin_index);
		}
		if (pointJoint_size != 0)
		{
			auto& joints = topo->pointJoints();
			int begin_index = contact_size + 3 * ballAndSocketJoint_size + 7 * sliderJoint_size + 7 * hingeJoint_size + 6 * fixedJoint_size;

			cuExecute(pointJoint_size,
				NGS_setUpPointJoint,
				joints,
				mPositionConstraints,
				this->inCenter()->getData(),
				begin_index);
		}

		auto sizeOfRigids = this->inCenter()->size();
		mContactNumber.resize(sizeOfRigids);
		mJointNumber.resize(sizeOfRigids);

		mJ_p.resize(4 * constraint_size);
		mB_p.resize(4 * constraint_size);

		mD_p.resize(constraint_size);
		mEta_p.resize(constraint_size);

		mJ_p.reset();
		mB_p.reset();
		mD_p.reset();
		mEta_p.reset();

		mContactNumber.reset();
		mJointNumber.reset();

		if (contact_size != 0)
		{
			cuExecute(contact_size,
				NGS_calculateNbrCons,
				this->inContacts()->getData(),
				mContactNumber);
		}

		if (ballAndSocketJoint_size != 0)
		{
			cuExecute(ballAndSocketJoint_size,
				NGS_calculateJoints,
				topo->ballAndSocketJoints(),
				mJointNumber);
		}

		if (sliderJoint_size != 0)
		{
			cuExecute(sliderJoint_size,
				NGS_calculateJoints,
				topo->sliderJoints(),
				mJointNumber);
		}

		if (hingeJoint_size != 0)
		{
			cuExecute(hingeJoint_size,
				NGS_calculateJoints,
				topo->hingeJoints(),
				mJointNumber);
		}

		if (fixedJoint_size != 0)
		{
			cuExecute(fixedJoint_size,
				NGS_calculateJoints,
				topo->fixedJoints(),
				mJointNumber);
		}

		cuExecute(constraint_size,
			NGS_calculateJacobianAndB,
			mJ_p,
			mB_p,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			mPositionConstraints,
			this->inRotationMatrix()->getData());

		cuExecute(constraint_size,
			NGS_calculateDiagonals,
			mD_p,
			mJ_p,
			mB_p);

		cuExecute(constraint_size,
			NGS_calculateEta,
			mEta_p,
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			mJ_p,
			this->inMass()->getData(),
			this->inCenter()->getData(),
			mPositionConstraints,
			this->inQuaternion()->getData(),
			this->varSlop()->getData(),
			dt);
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

	template<typename Real, typename Coord, typename Quat, typename Matrix>
	__global__ void NGS_updatePositionAndRotation(
		DArray<Coord> center,
		DArray<Quat> rotQuat,
		DArray<Matrix> rotMat,
		DArray<Matrix> inertia,
		DArray<Matrix> inertia_init,
		DArray<Coord> impulse_constrain,
		Real dt)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= center.size())
			return;

		Coord dv = impulse_constrain[2 * tId];


		Coord dw = impulse_constrain[2 * tId + 1];

		center[tId] += dt * dv;
		rotQuat[tId] += dt * 0.5 * Quat(dw.x, dw.y, dw.z, 0) * rotQuat[tId];

		rotQuat[tId] = rotQuat[tId].normalize();

		rotMat[tId] = rotQuat[tId].toMatrix3x3();

		inertia[tId] = rotMat[tId] * inertia_init[tId] * rotMat[tId].inverse();
	}

	template<typename Coord>
	__global__ void NGS_updateVelocity(
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> impulse_ext,
		DArray<Coord> impulse_constrain
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= velocity.size())
			return;
		velocity[tId] += impulse_ext[2 * tId] + impulse_constrain[2 * tId];
		angular_velocity[tId] += impulse_ext[2 * tId + 1] + impulse_constrain[2 * tId + 1];
	}

	template<typename Coord, typename Matrix, typename Quat>
	__global__ void NGS_updateGesture(
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
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= pos.size())
			return;

		velocity[tId] += Coord(0.0f, -9.8f, 0.0f) * dt;

		pos[tId] += velocity[tId] * dt;

		rotQuat[tId] = rotQuat[tId].normalize();

		rotQuat[tId] += dt * 0.5f *
			Quat(angular_velocity[tId][0], angular_velocity[tId][1], angular_velocity[tId][2], 0.0)
			* (rotQuat[tId]);

		rotQuat[tId] = rotQuat[tId].normalize();

		rotMat[tId] = rotQuat[tId].toMatrix3x3();

		inertia[tId] = rotMat[tId] * inertia_init[tId] * rotMat[tId].inverse();
	}

	template<typename ContactPair>
	__global__ void NGS_calculateNbrCons(
		DArray<ContactPair> nbc,
		DArray<int> nbrCnt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nbc.size())
			return;

		int idx1 = nbc[tId].bodyId1;
		int idx2 = nbc[tId].bodyId2;

		atomicAdd(&nbrCnt[idx1], 1);

		if (idx2 != INVALID)
			atomicAdd(&nbrCnt[idx2], 1);
	}

	template<typename Joint>
	__global__ void NGS_calculateJoints(
		DArray<Joint> joints,
		DArray<int> jointCnt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= joints.size())
			return;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;

		atomicAdd(&jointCnt[idx1], 1);
		atomicAdd(&jointCnt[idx2], 1);
	}

	template<typename Coord, typename Matrix, typename Constraint>
	__global__ void NGS_calculateJacobianAndB(
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
			if (constraints[tId].isValid == 1)
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
			if (constraints[tId].isValid == 1)
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

	template<typename Coord>
	__global__ void NGS_calculateDiagonals(
		DArray<Real> D,
		DArray<Coord> J,
		DArray<Coord> B
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= D.size())
			return;

		Real d = J[4 * tId].dot(B[4 * tId]) + J[4 * tId + 1].dot(B[4 * tId + 1]) + J[4 * tId + 2].dot(B[4 * tId + 2]) + J[4 * tId + 3].dot(B[4 * tId + 3]);

		D[tId] = d;
	}



	template<typename Coord, typename Constraint, typename Quat>
	__global__ void NGS_calculateEta(
		DArray<Real> eta,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> J,
		DArray<Real> mass,
		DArray<Coord> pos,
		DArray<Constraint> constraints,
		DArray<Quat> rotation_q,
		Real slop,
		Real dt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= constraints.size())
			return;


		Real invDt = Real(1) / dt;

		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		Real eta_i = Real(0);

		eta[tId] = eta_i;

		if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
		{
			if (constraints[tId].interpenetration < -slop)
			{
				Real beta = Real(1.0);
				Real alpha = 0;
				Real b_error;
				if (constraints[tId].interpenetration + slop < 0)
					b_error = beta * invDt * (constraints[tId].interpenetration + slop);
				else
					b_error = 0;
				eta[tId] -= b_error;// +b_res;
			}
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_1)
		{
			Real beta = 1.0;
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord error = pos[idx2] + r2 - pos[idx1] - r1;
			Real b_trans = beta * invDt * error[0];
			eta[tId] -= b_trans;
		}
		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_2)
		{
			Real beta = 1.0;
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord error = pos[idx2] + r2 - pos[idx1] - r1;
			Real b_trans = beta * invDt * error[1];
			eta[tId] -= b_trans;
		}
		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_3)
		{
			Real beta = 1.0;
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord error = pos[idx2] + r2 - pos[idx1] - r1;
			Real b_trans = beta * invDt * error[2];
			eta[tId] -= b_trans;
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_TRANS_1)
		{
			Real beta = Real(1.0);
			Coord r1 = constraints[tId].pos1;
			Coord r2 = constraints[tId].pos2;
			Coord n1 = constraints[tId].normal1;

			Real b_trans = invDt * beta * (pos[idx2] + r2 - pos[idx1] - r1).dot(n1);
			eta[tId] -= b_trans;
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_TRANS_2)
		{
			Real beta = Real(1.0);
			Coord r1 = constraints[tId].pos1;
			Coord r2 = constraints[tId].pos2;
			Coord n2 = constraints[tId].normal2;
			Real b_trans = invDt * beta * (pos[idx2] + r2 - pos[idx1] - r1).dot(n2);
			eta[tId] -= b_trans;
		}
		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_1)
		{
			Real beta = 1.0;
			Real yaw, pitch, roll_1, roll_2;
			Quat q2 = rotation_q[idx2];
			q2 = q2.normalize();
			Quat q1 = rotation_q[idx1];
			q1 = q1.normalize();

			q2.toEulerAngle(yaw, pitch, roll_2);
			q1.toEulerAngle(yaw, pitch, roll_1);



			Real roll_diff = roll_2 - roll_1;
			if (roll_diff > M_PI)
				roll_diff -= 2 * M_PI;
			else if (roll_diff < -M_PI)
				roll_diff += 2 * M_PI;
			Real b_rot = invDt * beta * roll_diff;
			eta[tId] -= b_rot;
		}
		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_2)
		{
			Real beta = 1.0;
			Real yaw, pitch_1, pitch_2, roll;
			Quat q2 = rotation_q[idx2];
			q2 = q2.normalize();
			Quat q1 = rotation_q[idx1];
			q1 = q1.normalize();

			q2.toEulerAngle(yaw, pitch_2, roll);
			q1.toEulerAngle(yaw, pitch_1, roll);

			Real pitch_diff = pitch_2 - pitch_1;
			if (pitch_diff > M_PI)
				pitch_diff -= 2 * M_PI;
			else if (pitch_diff < -M_PI)
				pitch_diff += 2 * M_PI;

			Real b_rot = invDt * beta * pitch_diff;
			eta[tId] -= b_rot;
		}
		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_3)
		{
			Real beta = 1.0;
			Real yaw_1, yaw_2, pitch, roll;
			Quat q2 = rotation_q[idx2];
			q2 = q2.normalize();
			Quat q1 = rotation_q[idx1];
			q1 = q1.normalize();

			q2.toEulerAngle(yaw_2, pitch, roll);
			q1.toEulerAngle(yaw_1, pitch, roll);
			Real yaw_diff = yaw_2 - yaw_1;
			if (yaw_diff > M_PI)
				yaw_diff -= 2 * M_PI;
			else if (yaw_diff < -M_PI)
				yaw_diff += 2 * M_PI;

			Real b_rot = invDt * beta * yaw_diff;

			eta[tId] -= b_rot;
		}
		if (constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MIN)
		{
			Real beta = 0.3;
			Real b_min = invDt * beta * constraints[tId].d_min;
			eta[tId] -= b_min;
		}
		if (constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MIN)
		{
			Real beta = Real(1) / Real(3);
			Real b_min = invDt * beta * constraints[tId].d_min;
			eta[tId] -= b_min;
		}
		if (constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MAX)
		{
			Real beta = 0.3;
			Real b_max = invDt * beta * constraints[tId].d_max;
			eta[tId] -= b_max;
		}
		if (constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MAX)
		{
			Real beta = Real(1) / Real(3);
			Real b_max = invDt * beta * constraints[tId].d_max;
			eta[tId] -= b_max;
		}

		if (constraints[tId].type == ConstraintType::CN_ALLOW_ROT1D_1)
		{
			Real beta = Real(1.0);
			Coord a1 = constraints[tId].axis;
			Coord b2 = constraints[tId].pos1;
			Real b_rot = invDt * beta * a1.dot(b2);
			eta[tId] -= b_rot;
		}

		if (constraints[tId].type == ConstraintType::CN_ALLOW_ROT1D_2)
		{
			Real beta = Real(1.0);
			Coord a1 = constraints[tId].axis;
			Coord c2 = constraints[tId].pos2;
			Real b_rot = invDt * beta * a1.dot(c2);
			eta[tId] -= b_rot;
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_1)
		{
			Real beta = Real(1.0);
			Coord error = constraints[tId].normal1;
			Real b_error = invDt * beta * error[0];
			eta[tId] -= b_error;
		}
		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_2)
		{
			Real beta = Real(1.0);
			Coord error = constraints[tId].normal1;
			Real b_error = invDt * beta * error[1];
			eta[tId] -= b_error;
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_3)
		{
			Real beta = Real(1.0);
			Coord error = constraints[tId].normal1;
			Real b_error = invDt * beta * error[2];
			eta[tId] -= b_error;
		}
	}


	template<typename Coord, typename Matrix, typename Contact, typename Constraint>
	__global__ void NGS_setUpContactConstraints(
		DArray<Constraint> constraints,
		DArray<Contact> contactsInLocalFrame,
		DArray<Coord> center,
		DArray<Matrix> rot)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contactsInLocalFrame.size())
			return;

		int contact_size = contactsInLocalFrame.size();

		int idx1 = contactsInLocalFrame[tId].bodyId1;
		int idx2 = contactsInLocalFrame[tId].bodyId2;

		Coord c1 = center[idx1];
		Matrix rot1 = rot[idx1];

		constraints[tId].bodyId1 = idx1;
		constraints[tId].bodyId2 = idx2;
		constraints[tId].pos1 = rot1 * contactsInLocalFrame[tId].pos1 + c1;
		constraints[tId].normal1 = -rot1 * contactsInLocalFrame[tId].normal1;
		if (idx2 != INVALID)
		{
			Coord c2 = center[idx2];
			Matrix rot2 = rot[idx2];

			constraints[tId].pos2 = rot2 * contactsInLocalFrame[tId].pos2 + c2;
			constraints[tId].normal2 = -rot2 * contactsInLocalFrame[tId].normal2;
			constraints[tId].interpenetration = (constraints[tId].pos2 - constraints[tId].pos1).dot(constraints[tId].normal1) < 0 ? (constraints[tId].pos2 - constraints[tId].pos1).dot(constraints[tId].normal1) : 0.0f;
		}
		else
		{
			Real dist = (contactsInLocalFrame[tId].pos2 - constraints[tId].pos1).dot(contactsInLocalFrame[tId].normal2);
			constraints[tId].interpenetration = dist < 0 ? dist : 0.0f;
		}

		constraints[tId].type = ConstraintType::CN_NONPENETRATION;
	}

	template<typename Joint, typename Constraint, typename Coord, typename Matrix>
	__global__ void NGS_setUpBallAndSocketJointConstraints(
		DArray<Constraint> constraints,
		DArray<Joint> joints,
		DArray<Coord> pos,
		DArray<Matrix> rotMat,
		int begin_index
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

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

	template<typename Joint, typename Constraint, typename Coord, typename Matrix>
	__global__ void NGS_setUpSliderJoint(
		DArray<Joint> joints,
		DArray<Constraint> constraints,
		DArray<Coord> pos,
		DArray<Matrix> rotMat,
		int begin_index
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= joints.size())
			return;

		int constraint_size = 7;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;

		Coord r1 = rotMat[idx1] * joints[tId].r1;
		Coord r2 = rotMat[idx2] * joints[tId].r2;

		Coord n = rotMat[idx1] * joints[tId].sliderAxis;
		n = n.normalize();
		Coord n1, n2;
		if (abs(n[1]) > EPSILON || abs(n[2]) > EPSILON)
		{
			n1 = Vector<Real, 3>(0, n[2], -n[1]);
			n1 = n1.normalize();
		}
		else if (abs(n[0]) > EPSILON)
		{
			n1 = Vector<Real, 3>(n[2], 0, -n[0]);
			n1 = n1.normalize();
		}
		n2 = n1.cross(n);
		n2 = n2.normalize();

		int baseIndex = constraint_size * tId + begin_index;

		bool useRange = joints[tId].useRange;
		Real C_min = 0.0;
		Real C_max = 0.0;
		if (joints[tId].useRange)
		{
			Coord u = pos[idx2] + r2 - pos[idx1] - r1;
			C_min = u.dot(n) - joints[tId].d_min;
			C_max = joints[tId].d_max - u.dot(n);
			if (C_min < 0)
				constraints[baseIndex + 5].isValid = true;
			else
				constraints[baseIndex + 5].isValid = false;
			if (C_max < 0)
				constraints[baseIndex + 6].isValid = true;
			else
				constraints[baseIndex + 6].isValid = false;
		}


		for (int i = 0; i < constraint_size; i++)
		{
			auto& constraint = constraints[baseIndex + i];
			constraint.bodyId1 = idx1;
			constraint.bodyId2 = idx2;
			constraint.pos1 = r1;
			constraint.pos2 = r2;
			constraint.normal1 = n1;
			constraint.normal2 = n2;
			constraint.axis = n;
			constraint.d_min = C_min;
			constraint.d_max = C_max;
		}

		constraints[baseIndex].type = ConstraintType::CN_ANCHOR_TRANS_1;
		constraints[baseIndex + 1].type = ConstraintType::CN_ANCHOR_TRANS_2;
		constraints[baseIndex + 2].type = ConstraintType::CN_BAN_ROT_1;
		constraints[baseIndex + 3].type = ConstraintType::CN_BAN_ROT_2;
		constraints[baseIndex + 4].type = ConstraintType::CN_BAN_ROT_3;
		constraints[baseIndex + 5].type = ConstraintType::CN_JOINT_SLIDER_MIN;
		constraints[baseIndex + 6].type = ConstraintType::CN_JOINT_SLIDER_MAX;
	}

	template<typename Joint, typename Constraint, typename Coord, typename Matrix, typename Quat>
	__global__ void NGS_setUpHingeJoint(
		DArray<Joint> joints,
		DArray<Constraint> constraints,
		DArray<Coord> pos,
		DArray<Matrix> rotMat,
		DArray<Quat> rotation_q,
		int begin_index
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= joints.size())
			return;

		int constraint_size = 7;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;

		Matrix rotMat1 = rotMat[idx1];
		Matrix rotMat2 = rotMat[idx2];


		Coord r1 = rotMat1 * joints[tId].r1;
		Coord r2 = rotMat2 * joints[tId].r2;

		Coord a1 = rotMat1 * joints[tId].hingeAxisBody1;
		Coord a2 = rotMat2 * joints[tId].hingeAxisBody2;



		// two vector orthogonal to the a2
		Coord b2, c2;
		if (abs(a2[1]) > EPSILON || abs(a2[2]) > EPSILON)
		{
			b2 = Coord(0, a2[2], -a2[1]);
			b2 = b2.normalize();
		}
		else if (abs(a2[0]) > EPSILON)
		{
			b2 = Coord(a2[2], 0, -a2[0]);
			b2 = b2.normalize();
		}
		c2 = b2.cross(a2);
		c2 = c2.normalize();

		Real C_min = 0.0;
		Real C_max = 0.0;
		int baseIndex = tId * constraint_size + begin_index;

		if (joints[tId].useRange)
		{
			Real theta = rotation_q[idx2].angle(rotation_q[idx1]);

			Quat q_rot = rotation_q[idx2] * rotation_q[idx1].inverse();

			if (a1.dot(Coord(q_rot.x, q_rot.y, q_rot.z)) < 0)
			{
				theta = -theta;
			}


			C_min = theta - joints[tId].d_min;
			C_max = joints[tId].d_max - theta;

			if (C_min < 0)
			{
				constraints[baseIndex + 5].isValid = true;
			}
			else
				constraints[baseIndex + 5].isValid = false;

			if (C_max < 0)
			{
				constraints[baseIndex + 6].isValid = true;
			}
			else
				constraints[baseIndex + 6].isValid = false;
		}


		for (int i = 0; i < constraint_size; i++)
		{
			constraints[baseIndex + i].bodyId1 = idx1;
			constraints[baseIndex + i].bodyId2 = idx2;
			constraints[baseIndex + i].axis = a1;
			constraints[baseIndex + i].normal1 = r1;
			constraints[baseIndex + i].normal2 = r2;
			constraints[baseIndex + i].pos1 = b2;
			constraints[baseIndex + i].pos2 = c2;
			constraints[baseIndex + i].d_min = C_min > 0 ? 0 : C_min;
			constraints[baseIndex + i].d_max = C_max > 0 ? 0 : C_max;
		}

		constraints[baseIndex].type = ConstraintType::CN_ANCHOR_EQUAL_1;
		constraints[baseIndex + 1].type = ConstraintType::CN_ANCHOR_EQUAL_2;
		constraints[baseIndex + 2].type = ConstraintType::CN_ANCHOR_EQUAL_3;
		constraints[baseIndex + 3].type = ConstraintType::CN_ALLOW_ROT1D_1;
		constraints[baseIndex + 4].type = ConstraintType::CN_ALLOW_ROT1D_2;
		constraints[baseIndex + 5].type = ConstraintType::CN_JOINT_HINGE_MIN;
		constraints[baseIndex + 6].type = ConstraintType::CN_JOINT_HINGE_MAX;

	}

	template<typename Joint, typename Constraint, typename Coord>
	__global__ void NGS_setUpPointJoint(
		DArray<Joint> joints,
		DArray<Constraint> constraints,
		DArray<Coord> pos,
		int begin_index
	)
	{
		int tId = threadIdx.x + blockDim.x * blockIdx.x;
		if (tId >= joints.size())
			return;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;

		int baseIndex = 3 * tId + begin_index;

		for (int i = 0; i < 3; i++)
		{
			constraints[baseIndex + i].bodyId1 = idx1;
			constraints[baseIndex + i].bodyId2 = idx2;
			constraints[baseIndex + i].normal1 = pos[idx1] - joints[tId].anchorPoint;
		}

		constraints[baseIndex].type = ConstraintType::CN_JOINT_NO_MOVE_1;
		constraints[baseIndex + 1].type = ConstraintType::CN_JOINT_NO_MOVE_2;
		constraints[baseIndex + 2].type = ConstraintType::CN_JOINT_NO_MOVE_3;

	}

	template<typename Joint, typename Constraint, typename Matrix>
	__global__ void NGS_setUpFixedJoint(
		DArray<Joint> joints,
		DArray<Constraint> constraints,
		DArray<Matrix> rotMat,
		int begin_index
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= joints.size())
			return;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;
		Vector<Real, 3> r1 = rotMat[idx1] * joints[tId].r1;
		Vector<Real, 3> r2 = rotMat[idx2] * joints[tId].r2;

		int baseIndex = 6 * tId + begin_index;
		for (int i = 0; i < 6; i++)
		{
			constraints[baseIndex + i].bodyId1 = idx1;
			constraints[baseIndex + i].bodyId2 = idx2;
			constraints[baseIndex + i].normal1 = r1;
			constraints[baseIndex + i].normal2 = r2;
		}

		constraints[baseIndex].type = ConstraintType::CN_ANCHOR_EQUAL_1;
		constraints[baseIndex + 1].type = ConstraintType::CN_ANCHOR_EQUAL_2;
		constraints[baseIndex + 2].type = ConstraintType::CN_ANCHOR_EQUAL_3;
		constraints[baseIndex + 3].type = ConstraintType::CN_BAN_ROT_1;
		constraints[baseIndex + 4].type = ConstraintType::CN_BAN_ROT_2;
		constraints[baseIndex + 5].type = ConstraintType::CN_BAN_ROT_3;
	}

	template<typename Coord, typename Constraint>
	__global__ void NGS_takeOneJacobiIteration(
		DArray<Real> lambda,
		DArray<Coord> impulse,
		DArray<Real> d,
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Real> eta,
		DArray<Real> mass,
		DArray<Constraint> constraints,
		DArray<int> nbq,
		DArray<int> jointNumber,
		Real mu,
		Real g,
		Real dt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= constraints.size())
			return;

		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		double tmp = eta[tId];

		tmp -= J[4 * tId].dot(impulse[idx1 * 2]);
		tmp -= J[4 * tId + 1].dot(impulse[idx1 * 2 + 1]);

		if (idx2 != INVALID)
		{
			tmp -= J[4 * tId + 2].dot(impulse[idx2 * 2]);
			tmp -= J[4 * tId + 3].dot(impulse[idx2 * 2 + 1]);
		}

		if (d[tId] > EPSILON)
		{
			int stepInverse = 0;
			if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
			{
				if (idx2 != INVALID)
				{
					stepInverse = nbq[idx1] > nbq[idx2] ? nbq[idx1] : nbq[idx2];
				}
				else
				{
					stepInverse = nbq[idx1];
				}
			}

			else
			{
				if (idx2 != INVALID)
				{
					stepInverse += 4 * jointNumber[idx2];
				}

				stepInverse += 4 * jointNumber[idx1];
			}

			stepInverse = stepInverse < 1 ? 1 : stepInverse;

			double delta_lambda = (tmp / (d[tId] * stepInverse));
			double lambda_new = lambda[tId] + delta_lambda;

			// Projection to Bound
			if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
			{
				if (lambda_new < 0)
				{
					lambda_new = 0;
					delta_lambda = lambda_new - lambda[tId];
				}
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

		template<typename Contact, typename Coord, typename Matrix>
		__global__ void NGS_setUpContactsInLocalFrame(
			DArray<Contact> contactsInLocalFrame,
			DArray<Contact> contactsInGlobalFrame,
			DArray<Coord> center,
			DArray<Matrix> rot)
		{
			int tId = threadIdx.x + (blockIdx.x * blockDim.x);
			if (tId >= contactsInGlobalFrame.size())
				return;

			Contact globalC = contactsInGlobalFrame[tId];

			int idx1 = globalC.bodyId1;
			int idx2 = globalC.bodyId2;

			Contact localC;
			localC.bodyId1 = globalC.bodyId1;
			localC.bodyId2 = globalC.bodyId2;

			localC.interpenetration = globalC.interpenetration;
			localC.contactType = globalC.contactType;

			Coord c1 = center[idx1];
			Matrix rot1 = rot[idx1];

			if (idx2 != INVALID)
			{
				Coord c2 = center[idx2];
				Matrix rot2 = rot[idx2];
				localC.pos1 = rot1.transpose() * (globalC.pos1 - globalC.normal1 * globalC.interpenetration - c1);
				localC.normal1 = rot1.transpose() * globalC.normal1;
				localC.pos2 = rot2.transpose() * (globalC.pos1 - c2);
				localC.normal2 = -rot2.transpose() * globalC.normal1;
			}
			else
			{
				localC.pos1 = rot1.transpose() * (globalC.pos1 - c1);
				localC.normal1 = rot1.transpose() * globalC.normal1;

				localC.pos2 = globalC.pos1 + globalC.normal1 * globalC.interpenetration;
				localC.normal2 = -globalC.normal1;
			}

			contactsInLocalFrame[tId] = localC;
		}

	template <typename Real, typename Coord, typename Quat>
	__global__ void ICS_UpdateVelocityForPositionSolver(
		DArray<Coord> velocity,
		DArray<Coord> angularVelocity,
		DArray<Coord> center,
		DArray<Coord> p_prev,
		DArray<Quat> quaternion,
		DArray<Quat> q_prev,
		Real damping,
		Real angularDamping,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velocity.size())
			return;

		velocity[pId] += (center[pId] - p_prev[pId]) / dt;
		velocity[pId] *= damping;

		Quat q = quaternion[pId];
		q = q.normalize();
		Quat dq = q * (q_prev[pId].inverse());

		Coord dw = 2 * Coord(dq.x, dq.y, dq.z) / dt;
		dw *= dq.w >= 0 ? 1.0f : -1.0f;

		angularVelocity[pId] += dw;
		angularVelocity[pId] *= 1.0 * angularDamping;
	}

	template<typename TDataType>
	void IterativeConstraintSolver<TDataType>::constrain()
	{
		uint bodyNum = this->inCenter()->size();

		auto topo = this->inDiscreteElements()->constDataPtr();

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

		if (mContactsInLocalFrame.size() != this->inContacts()->size()) {
			mContactsInLocalFrame.resize(this->inContacts()->size());
		}

		cuExecute(this->inContacts()->size(),
			NGS_setUpContactsInLocalFrame,
			mContactsInLocalFrame,
			this->inContacts()->getData(),
			this->inCenter()->getData(),
			this->inRotationMatrix()->getData());

		//Velocity solver
		if (!this->inContacts()->isEmpty() || topo->totalJointSize() > 0)
		{
			initializeJacobian(dt);

			int constraint_size = mVelocityConstraints.size();

			for (int i = 0; i < this->varIterationNumberForVelocitySolver()->getValue(); i++)
			{
				mDiff.resize(constraint_size);
				mDiff.reset();
				cuExecute(constraint_size,
					takeOneJacobiIteration,
					mLambda,
					mImpulseC,
					mD,
					mJ,
					mB,
					mEta,
					this->inMass()->getData(),
					mVelocityConstraints,
					mContactNumber,
					mJointNumber,
					this->varFrictionCoefficient()->getData(),
					this->varGravityValue()->getData(),
					dt);
			}
		}

		cuExecute(bodyNum,
			RB_updateVelocity,
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			mImpulseExt,
			mImpulseC);


		cuExecute(bodyNum,
			RB_updateGesture,
			this->inCenter()->getData(),
			this->inQuaternion()->getData(),
			this->inRotationMatrix()->getData(),
			this->inInertia()->getData(),
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			this->inInitialInertia()->getData(),
			dt);

		//Position solver
		if (!this->inContacts()->isEmpty() || topo->totalJointSize() > 0)
		{
			mCenterPre.assign(this->inCenter()->constData());
			mQuatPre.assign(this->inQuaternion()->constData());

			if (mContactsInLocalFrame.size() != this->inContacts()->size()) {
				mContactsInLocalFrame.resize(this->inContacts()->size());
			}

			cuExecute(this->inContacts()->size(),
				NGS_setUpContactsInLocalFrame,
				mContactsInLocalFrame,
				this->inContacts()->getData(),
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData());
			for (size_t ngs = 0; ngs < this->varIterationNumberForPositionSolver()->getData(); ngs++)
			{
				mImpulseC.reset();
				mLambda.reset();
				initializeJacobianForNGS(dt);
				int constraint_size = mPositionConstraints.size();
				for (int j = 0; j < 1; j++)
				{
					cuExecute(constraint_size,
						NGS_takeOneJacobiIteration,
						mLambda,
						mImpulseC,
						mD_p,
						mJ_p,
						mB_p,
						mEta_p,
						this->inMass()->getData(),
						mPositionConstraints,
						mContactNumber,
						mJointNumber,
						this->varFrictionCoefficient()->getData(),
						this->varGravityValue()->getData(),
						dt);
				}
				cuExecute(bodyNum,
					NGS_updatePositionAndRotation,
					this->inCenter()->getData(),
					this->inQuaternion()->getData(),
					this->inRotationMatrix()->getData(),
					this->inInertia()->getData(),
					this->inInitialInertia()->getData(),
					mImpulseC,
					dt);
			}

			cuExecute(bodyNum,
				ICS_UpdateVelocityForPositionSolver,
				this->inVelocity()->getData(),
				this->inAngularVelocity()->getData(),
				this->inCenter()->getData(),
				mCenterPre,
				this->inQuaternion()->getData(),
				mQuatPre,
				Real(0.99),
				Real(0.99),
				dt);
		}
	}

	DEFINE_CLASS(IterativeConstraintSolver);
}