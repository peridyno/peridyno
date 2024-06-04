#include "SharedFuncsForRigidBody.h"

namespace dyno
{
	  __global__ void SF_ApplyTransform(
		DArrayList<Transform3f> instanceTransform,
		const DArray<Vec3f> offset,
		const DArray<Vec3f> translate,
		const DArray<Mat3f> rotation,
		const DArray<Mat3f> rotationInit,
		const DArray<Pair<uint, uint>> binding,
		const DArray<int> bindingtag)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= rotation.size())
			return;
		if (bindingtag[tId] == 0)
			return;

		Pair<uint, uint> pair = binding[tId];

		Mat3f rot = rotation[tId] * rotationInit[tId].transpose();

		Transform3f ti = Transform3f(translate[tId] - rot * offset[tId], rot);

		instanceTransform[pair.first][pair.second] = ti;
	}

	void ApplyTransform(
		DArrayList<Transform3f>& instanceTransform, 
		const DArray<Vec3f>& offset,
		const DArray<Vec3f>& translate,
		const DArray<Mat3f>& rotation,
		const DArray<Mat3f>& rotationInit,
		const DArray<Pair<uint, uint>>& binding,
		const DArray<int>& bindingtag)
	{
		cuExecute(rotation.size(),
			SF_ApplyTransform,
			instanceTransform,
			offset,
			translate,
			rotation,
			rotationInit,
			binding,
			bindingtag);

	}

	/**
	* Update Velocity Function
	*
	* @param velocity			velocity of rigids
	* @param angular_velocity	angular velocity of rigids
	* @param impulse			impulse exert on rigids
	* @param linearDamping		damping ratio of linear velocity
	* @param angularDamping		damping ratio of angular velocity
	* @param dt					time step
	* This function update the velocity of rigids based on impulse
	*/
	__global__ void SF_updateVelocity(
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Vec3f> impulse,
		float linearDamping,
		float angularDamping,
		float dt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= velocity.size())
			return;

		velocity[tId] += impulse[2 * tId];
		angular_velocity[tId] += impulse[2 * tId + 1];

		//Damping
		velocity[tId] *= 1.0f / (1.0f + dt * linearDamping);
		angular_velocity[tId] *= 1.0f / (1.0f + dt * angularDamping);
	}

	void updateVelocity(
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Vec3f> impulse,
		float linearDamping,
		float angularDamping,
		float dt
	)
	{
		cuExecute(velocity.size(),
			SF_updateVelocity,
			velocity,
			angular_velocity,
			impulse,
			linearDamping,
			angularDamping,
			dt);
	}


	/**
	* Update Gesture Function
	*
	* @param pos				position of rigids
	* @param rotQuat			quaterion of rigids
	* @param rotMat				rotation matrix of rigids
	* @param inertia			inertia matrix of rigids
	* @param velocity			velocity of rigids
	* @param angular_velocity	angular velocity of rigids
	* @param inertia_init		initial inertial matrix of rigids
	* @param dt					time step
	* This function update the gesture of rigids based on velocity and timeStep
	*/
	__global__ void SF_updateGesture(
		DArray<Vec3f> pos,
		DArray<Quat1f> rotQuat,
		DArray<Mat3f> rotMat,
		DArray<Mat3f> inertia,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Mat3f> inertia_init,
		float dt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= pos.size())
			return;

		pos[tId] += velocity[tId] * dt;

		rotQuat[tId] = rotQuat[tId].normalize();

		rotQuat[tId] += dt * 0.5f *
			Quat1f(angular_velocity[tId][0], angular_velocity[tId][1], angular_velocity[tId][2], 0.0)
			* (rotQuat[tId]);

		rotQuat[tId] = rotQuat[tId].normalize();

		rotMat[tId] = rotQuat[tId].toMatrix3x3();

		inertia[tId] = rotMat[tId] * inertia_init[tId] * rotMat[tId].inverse();
	}

	void updateGesture(
		DArray<Vec3f> pos,
		DArray<Quat1f> rotQuat,
		DArray<Mat3f> rotMat,
		DArray<Mat3f> inertia,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Mat3f> inertia_init,
		float dt
	)
	{
		cuExecute(pos.size(),
			SF_updateGesture,
			pos,
			rotQuat,
			rotMat,
			inertia,
			velocity,
			angular_velocity,
			inertia_init,
			dt);
	}

	/**
	* update the position and rotation of rigids
	*
	* @param pos				position of rigids
	* @param rotQuat			quaterion of rigids
	* @param rotMat				rotation matrix of rigids
	* @param inertia			inertia matrix of rigids
	* @param inertia_init		initial inertia matrix of rigids
	* @param impulse_constrain  impulse to update position and rotation
	* This function update the position of rigids use delta position
	*/
	__global__ void SF_updatePositionAndRotation(
		DArray<Vec3f> pos,
		DArray<Quat1f> rotQuat,
		DArray<Mat3f> rotMat,
		DArray<Mat3f> inertia,
		DArray<Mat3f> inertia_init,
		DArray<Vec3f> impulse_constrain
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= pos.size())
			return;
		Vec3f dx = impulse_constrain[2 * tId];
		Vec3f dq = impulse_constrain[2 * tId + 1];

		pos[tId] += impulse_constrain[2 * tId];
		rotQuat[tId] += 0.5 * Quat1f(dq.x, dq.y, dq.z, 0.0f) * rotQuat[tId];

		rotQuat[tId] = rotQuat[tId].normalize();
		rotMat[tId] = rotQuat[tId].toMatrix3x3();
		inertia[tId] = rotMat[tId] * inertia_init[tId] * rotMat[tId].inverse();
	}

	void updatePositionAndRotation(
		DArray<Vec3f> pos,
		DArray<Quat1f> rotQuat,
		DArray<Mat3f> rotMat,
		DArray<Mat3f> inertia,
		DArray<Mat3f> inertia_init,
		DArray<Vec3f> impulse_constrain
	)
	{
		cuExecute(pos.size(),
			SF_updatePositionAndRotation,
			pos,
			rotQuat,
			rotMat,
			inertia,
			inertia_init,
			impulse_constrain);
	}

	/**
	* calculate contact point num function
	*
	* @param contacts					contacts
	* @param contactCnt					contact num of each rigids
	* This function calculate the contact num of each rigids
	*/
	template<typename ContactPair>
	__global__ void SF_calculateContactPoints(
		DArray<ContactPair> contacts,
		DArray<int> contactCnt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contacts.size())
			return;

		int idx1 = contacts[tId].bodyId1;
		int idx2 = contacts[tId].bodyId2;

		atomicAdd(&contactCnt[idx1], 1);

		if (idx2 != INVALID)
			atomicAdd(&contactCnt[idx2], 1);
	}

	void calculateContactPoints(
		DArray<TContactPair<float>> contacts,
		DArray<int> contactCnt
	)
	{
		cuExecute(contacts.size(),
			SF_calculateContactPoints,
			contacts,
			contactCnt);
	}

	
	/**
	* calculate Jacobian Matrix function
	*
	* @param J				Jacobian Matrix
	* @param B				M^-1J Matrix
	* @param pos			postion of rigids
	* @param inertia		inertia matrix of rigids
	* @param mass			mass of rigids
	* @param rotMat			rotation Matrix of rigids
	* @param constraints	constraints data
	* This function calculate the Jacobian Matrix of constraints
	*/
	template<typename Coord, typename Matrix, typename Constraint>
	__global__ void SF_calculateJacobianMatrix(
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Coord> pos,
		DArray<Matrix> inertia,
		DArray<Real> mass,
		DArray<Matrix> rotMat,
		DArray<Constraint> constraints
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

	template<typename Coord, typename Matrix, typename Constraint>
	__global__ void SF_calculateJacobianMatrixForNJS(
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Coord> pos,
		DArray<Matrix> inertia,
		DArray<Real> mass,
		DArray<Matrix> rotMat,
		DArray<Constraint> constraints
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= constraints.size())
			return;

		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
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

	void calculateJacobianMatrix(
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<Vec3f> pos,
		DArray<Mat3f> inertia,
		DArray<float> mass,
		DArray<Mat3f> rotMat,
		DArray<TConstraintPair<float>> constraints
	)
	{
		cuExecute(constraints.size(),
			SF_calculateJacobianMatrix,
			J,
			B,
			pos,
			inertia,
			mass,
			rotMat,
			constraints);
	}

	void calculateJacobianMatrixForNJS(
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<Vec3f> pos,
		DArray<Mat3f> inertia,
		DArray<float> mass,
		DArray<Mat3f> rotMat,
		DArray<TConstraintPair<float>> constraints
	)
	{
		cuExecute(constraints.size(),
			SF_calculateJacobianMatrixForNJS,
			J,
			B,
			pos,
			inertia,
			mass,
			rotMat,
			constraints);
	}


	/**
	* calculate eta vector for PJS
	*
	* @param eta				eta vector
	* @param J					Jacobian Matrix
	* @param velocity			linear velocity of rigids
	* @param angular_velocity	angular velocity of rigids
	* @param constraints		constraints data
	* This function calculate the diagonal Matrix of JB
	*/
	template<typename Coord, typename Constraint, typename Real>
	__global__ void SF_calculateEtaVectorForPJS(
		DArray<Real> eta,
		DArray<Coord> J,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Constraint> constraints
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
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

		if (constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MOTER || constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MOTER)
		{
			Real v_moter = constraints[tId].interpenetration;
			eta[tId] -= v_moter;
		}
	}

	void calculateEtaVectorForPJS(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<TConstraintPair<float>> constraints
	)
	{
		cuExecute(constraints.size(),
			SF_calculateEtaVectorForPJS,
			eta,
			J,
			velocity,
			angular_velocity,
			constraints);
	}

	/**
	* calculate eta vector for PJS Baumgarte stabilization
	*
	* @param eta				eta vector
	* @param J					Jacobian Matrix
	* @param velocity			linear velocity of rigids
	* @param angular_velocity	angular velocity of rigids
	* @param pos				position of rigids
	* @param rotation_q			quarterion of rigids
	* @param constraints		constraints data
	* @param slop				interpenetration slop
	* @param beta				Baumgarte bias
	* @param dt					time step
	* This function calculate the diagonal Matrix of JB
	*/
	template<typename Coord, typename Constraint, typename Real, typename Quat>
	__global__ void SF_calculateEtaVectorForPJSBaumgarte(
		DArray<Real> eta,
		DArray<Coord> J,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> pos,
		DArray<Quat> rotation_q,
		DArray<Constraint> constraints,
		Real slop,
		Real beta,
		Real dt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
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

		Real invDt = Real(1) / dt;
		Real error = 0;

		if (constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MOTER || constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MOTER)
		{
			Real v_moter = constraints[tId].interpenetration;
			eta[tId] -= v_moter;
		}

		if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
		{
			error = minimum(constraints[tId].interpenetration + slop, 0.0f);
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_1)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			error = errorVec[0];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_2)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			error = errorVec[1];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_3)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			error = errorVec[2];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_TRANS_1)
		{
			Coord r1 = constraints[tId].pos1;
			Coord r2 = constraints[tId].pos2;
			Coord n1 = constraints[tId].normal1;

			error = (pos[idx2] + r2 - pos[idx1] - r1).dot(n1);
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_TRANS_2)
		{
			Coord r1 = constraints[tId].pos1;
			Coord r2 = constraints[tId].pos2;
			Coord n2 = constraints[tId].normal2;

			error = (pos[idx2] + r2 - pos[idx1] - r1).dot(n2);
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_1)
		{
			Real yaw, pitch, roll_1, roll_2;
			Quat q2 = rotation_q[idx2];
			q2 = q2.normalize();
			Quat q1 = rotation_q[idx1];
			q1 = q1.normalize();

			/*q2.toEulerAngle(yaw, pitch, roll_2);
			q1.toEulerAngle(yaw, pitch, roll_1);

			Real roll_diff = roll_2 - roll_1;
			if (roll_diff > M_PI)
				roll_diff -= 2 * M_PI;
			else if (roll_diff < -M_PI)
				roll_diff += 2 * M_PI;
			error = roll_diff;*/
			Quat q_error = q2 * q1.inverse();
			q_error = q_error.normalize();
			Real theta = 2.0 * acos(q_error.w);
			Real s = sqrt(1.0 - q_error.w * q_error.w);
			Vec3f u = Vec3f(q_error.x, q_error.y, q_error.z);
			if (s < 1e-6)
			{
				u = Vec3f(1.0, 0, 0);
			}
			else
			{
				u = u / s;
			}
			
			error = theta * u.x;
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_2)
		{
			Real yaw, pitch_1, pitch_2, roll;
			Quat q2 = rotation_q[idx2];
			q2 = q2.normalize();
			Quat q1 = rotation_q[idx1];
			q1 = q1.normalize();

			/*q2.toEulerAngle(yaw, pitch_2, roll);
			q1.toEulerAngle(yaw, pitch_1, roll);

			Real pitch_diff = pitch_2 - pitch_1;
			if (pitch_diff > M_PI)
				pitch_diff -= 2 * M_PI;
			else if (pitch_diff < -M_PI)
				pitch_diff += 2 * M_PI;

			error = pitch_diff;*/
			Quat q_error = q2 * q1.inverse();
			q_error = q_error.normalize();
			Real theta = 2.0 * acos(q_error.w);
			Real s = sqrt(1.0 - q_error.w * q_error.w);
			Vec3f u = Vec3f(q_error.x, q_error.y, q_error.z);
			if (s < 1e-6)
			{
				u = Vec3f(1.0, 0, 0);
			}
			else
			{
				u = u / s;
			}
			error = theta * u.y;
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_3)
		{
			Real yaw_1, yaw_2, pitch, roll;
			Quat q2 = rotation_q[idx2];
			q2 = q2.normalize();
			Quat q1 = rotation_q[idx1];
			q1 = q1.normalize();

			/*q2.toEulerAngle(yaw_2, pitch, roll);
			q1.toEulerAngle(yaw_1, pitch, roll);
			Real yaw_diff = yaw_2 - yaw_1;
			if (yaw_diff > M_PI)
				yaw_diff -= 2 * M_PI;
			else if (yaw_diff < -M_PI)
				yaw_diff += 2 * M_PI;

			error = yaw_diff;*/
			Quat q_error = q2 * q1.inverse();
			q_error = q_error.normalize();
			Real theta = 2.0 * acos(q_error.w);
			Real s = sqrt(1.0 - q_error.w * q_error.w);
			Vec3f u = Vec3f(q_error.x, q_error.y, q_error.z);
			if (s < 1e-6)
			{
				u = Vec3f(1.0, 0, 0);
			}
			else
			{
				u = u / s;
			}
			error = theta * u.z;
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MIN || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MIN)
		{
			error = constraints[tId].d_min;
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MAX || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MAX)
		{
			error = constraints[tId].d_max;
		}

		if (constraints[tId].type == ConstraintType::CN_ALLOW_ROT1D_1)
		{
			Coord a1 = constraints[tId].axis;
			Coord b2 = constraints[tId].pos1;
			error = a1.dot(b2);
		}

		if (constraints[tId].type == ConstraintType::CN_ALLOW_ROT1D_2)
		{
			Coord a1 = constraints[tId].axis;
			Coord c2 = constraints[tId].pos2;
			error = a1.dot(c2);
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_1)
		{
			Coord errorVec = constraints[tId].normal1;
			error = errorVec[0];
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_2)
		{
			Coord errorVec = constraints[tId].normal1;
			error = errorVec[1];
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_3)
		{
			Coord errorVec = constraints[tId].normal1;
			error = errorVec[2];
		}

		eta[tId] -= beta * invDt * error;
	}

	void calculateEtaVectorForPJSBaumgarte(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Vec3f> pos,
		DArray<Quat1f> rotation_q,
		DArray <TConstraintPair<float>> constraints,
		float slop,
		float beta,
		float dt
	)
	{
		cuExecute(constraints.size(),
			SF_calculateEtaVectorForPJSBaumgarte,
			eta,
			J,
			velocity,
			angular_velocity,
			pos,
			rotation_q,
			constraints,
			slop,
			beta,
			dt);
	}

	template<typename Coord, typename Constraint, typename Real, typename Quat>
	__global__ void SF_calculateEtaVectorForPJSoft(
		DArray<Real> eta,
		DArray<Coord> J,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> pos,
		DArray<Quat> rotation_q,
		DArray<Constraint> constraints,
		Real slop,
		Real zeta,
		Real hertz,
		Real dt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
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

		Real invDt = Real(1) / dt;
		Real error = 0;

		if (constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MOTER || constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MOTER)
		{
			Real v_moter = constraints[tId].interpenetration;
			eta[tId] -= v_moter;
		}

		if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
		{
			error = minimum(constraints[tId].interpenetration + slop, 0.0f);
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_1)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			error = errorVec[0];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_2)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			error = errorVec[1];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_3)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			error = errorVec[2];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_TRANS_1)
		{
			Coord r1 = constraints[tId].pos1;
			Coord r2 = constraints[tId].pos2;
			Coord n1 = constraints[tId].normal1;

			error = (pos[idx2] + r2 - pos[idx1] - r1).dot(n1);
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_TRANS_2)
		{
			Coord r1 = constraints[tId].pos1;
			Coord r2 = constraints[tId].pos2;
			Coord n2 = constraints[tId].normal2;

			error = (pos[idx2] + r2 - pos[idx1] - r1).dot(n2);
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_1)
		{
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
			error = roll_diff;
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_2)
		{
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

			error = pitch_diff;
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_3)
		{
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

			error = yaw_diff;
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MIN || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MIN)
		{
			error = constraints[tId].d_min;
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MAX || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MAX)
		{
			error = constraints[tId].d_max;
		}

		if (constraints[tId].type == ConstraintType::CN_ALLOW_ROT1D_1)
		{
			Coord a1 = constraints[tId].axis;
			Coord b2 = constraints[tId].pos1;
			error = a1.dot(b2);
		}

		if (constraints[tId].type == ConstraintType::CN_ALLOW_ROT1D_2)
		{
			Coord a1 = constraints[tId].axis;
			Coord c2 = constraints[tId].pos2;
			error = a1.dot(c2);
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_1)
		{
			Coord errorVec = constraints[tId].normal1;
			error = errorVec[0];
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_2)
		{
			Coord errorVec = constraints[tId].normal1;
			error = errorVec[1];
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_3)
		{
			Coord errorVec = constraints[tId].normal1;
			error = errorVec[2];
		}

		Real omega = 2.0 * M_PI * hertz;
		Real a1 = 2.0 * zeta + omega * dt;
		Real biasRate = omega / a1;
		eta[tId] -= biasRate * error;
	}

	void calculateEtaVectorForPJSoft(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Vec3f> pos,
		DArray<Quat1f> rotation_q,
		DArray <TConstraintPair<float>> constraints,
		float slop,
		float zeta,
		float hertz,
		float dt
	)
	{
		cuExecute(constraints.size(),
			SF_calculateEtaVectorForPJSoft,
			eta,
			J,
			velocity,
			angular_velocity,
			pos,
			rotation_q,
			constraints,
			slop,
			zeta,
			hertz,
			dt);
	}


	/**
	* calculate eta vector for NJS
	*
	* @param eta			eta vector
	* @param J				Jacobian Matrix
	* @param pos			position of rigids
	* @param rotation_q		quaterion of rigids
	* @param constraints	constraints data
	* @param linear slop	linear slop
	* @param angular slop	angular slop
	* @param beta			bias ratio
	* This function calculate the diagonal Matrix of JB
	*/
	template<typename Coord, typename Constraint, typename Real, typename Quat>
	__global__ void SF_calculateEtaVectorForNJS(
		DArray<Real> eta,
		DArray<Coord> J,
		DArray<Coord> pos,
		DArray<Quat> rotation_q,
		DArray<Constraint> constraints,
		Real slop,
		Real beta
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= constraints.size())
			return;

		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		Real eta_i = Real(0);

		Real error = 0.0f;

		if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
		{
			error = minimum(constraints[tId].interpenetration + slop, 0.0f);
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_1)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			error = errorVec[0];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_2)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			error = errorVec[1];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_3)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			error = errorVec[2];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_TRANS_1)
		{
			Coord r1 = constraints[tId].pos1;
			Coord r2 = constraints[tId].pos2;
			Coord n1 = constraints[tId].normal1;

			error = (pos[idx2] + r2 - pos[idx1] - r1).dot(n1);
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_TRANS_2)
		{
			Coord r1 = constraints[tId].pos1;
			Coord r2 = constraints[tId].pos2;
			Coord n2 = constraints[tId].normal2;

			error = (pos[idx2] + r2 - pos[idx1] - r1).dot(n2);
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_1)
		{
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
			error = roll_diff;
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_2)
		{
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

			error = pitch_diff;
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_3)
		{
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

			error = yaw_diff;
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MIN || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MIN)
		{
			error = constraints[tId].d_min;
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MAX || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MAX)
		{
			error = constraints[tId].d_max;
		}

		if (constraints[tId].type == ConstraintType::CN_ALLOW_ROT1D_1)
		{
			Coord a1 = constraints[tId].axis;
			Coord b2 = constraints[tId].pos1;
			error = a1.dot(b2);
		}

		if (constraints[tId].type == ConstraintType::CN_ALLOW_ROT1D_2)
		{
			Coord a1 = constraints[tId].axis;
			Coord c2 = constraints[tId].pos2;
			error = a1.dot(c2);
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_1)
		{
			Coord errorVec = constraints[tId].normal1;
			error = errorVec[0];
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_2)
		{
			Coord errorVec = constraints[tId].normal1;
			error = errorVec[1];
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_3)
		{
			Coord errorVec = constraints[tId].normal1;
			error = errorVec[2];
		}

		eta[tId] = eta_i - beta * error;
	}

	void calculateEtaVectorForNJS(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> pos,
		DArray<Quat1f> rotation_q,
		DArray <TConstraintPair<float>> constraints,
		float slop,
		float beta
	)
	{
		cuExecute(constraints.size(),
			SF_calculateEtaVectorForNJS,
			eta,
			J,
			pos,
			rotation_q,
			constraints,
			slop,
			beta);
	}

	/**
	* Store the contacts in local coordinates.
	*
	* @param contactsInLocalFrame		contacts in local coordinates
	* @param contactsInGlobalFrame		contacts in global coordinates
	* @param pos						position of rigids
	* @param rotMat						rotation matrix of rigids
	* This function store the contacts in local coordinates.
	*/
	template<typename Contact, typename Coord, typename Matrix>
	__global__ void SF_setUpContactsInLocalFrame(
		DArray<Contact> contactsInLocalFrame,
		DArray<Contact> contactsInGlobalFrame,
		DArray<Coord> pos,
		DArray<Matrix> rotMat
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contactsInGlobalFrame.size())
			return;

		Contact globalC = contactsInGlobalFrame[tId];
		int idx1 = globalC.bodyId1;
		int idx2 = globalC.bodyId2;

		Contact localC;
		localC.bodyId1 = idx1;
		localC.bodyId2 = idx2;

		localC.interpenetration = -globalC.interpenetration;
		localC.contactType = globalC.contactType;

		Coord c1 = pos[idx1];
		Matrix rot1 = rotMat[idx1];

		if (idx2 != INVALID)
		{
			Coord c2 = pos[idx2];
			Matrix rot2 = rotMat[idx2];
			localC.pos1 = rot1.transpose() * (globalC.pos1 - c1);
			localC.normal1 = -globalC.normal1;
			localC.pos2 = rot2.transpose() * (globalC.pos1 - c2);
			localC.normal2 = globalC.normal1;
		}
		else
		{
			localC.pos1 = rot1.transpose() * (globalC.pos1 - c1);
			localC.normal1 = - globalC.normal1;
			localC.pos2 = globalC.pos1;
			localC.normal2 = globalC.normal1;
		}
		contactsInLocalFrame[tId] = localC;
	}

	void setUpContactsInLocalFrame(
		DArray<TContactPair<float>> contactsInLocalFrame,
		DArray<TContactPair<float>> contactsInGlobalFrame,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat
	)
	{
		cuExecute(contactsInGlobalFrame.size(),
			SF_setUpContactsInLocalFrame,
			contactsInLocalFrame,
			contactsInGlobalFrame,
			pos,
			rotMat);
	}

	/**
	* Set up the contact and friction constraints
	*
	* @param constraints				constraints data
	* @param contactsInLocalFrame		contacts in local coordinates
	* @param pos						position of rigids
	* @param rotMat						rotation matrix of rigids
	* @param hasFriction				friction choice
	* This function set up the contact and friction constraints
	*/
	template<typename Coord, typename Matrix, typename Contact, typename Constraint>
	__global__ void SF_setUpContactAndFrictionConstraints(
		DArray<Constraint> constraints,
		DArray<Contact> contactsInLocalFrame,
		DArray<Coord> pos,
		DArray<Matrix> rotMat,
		bool hasFriction
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contactsInLocalFrame.size())
			return;

		int contact_size = contactsInLocalFrame.size();

		int idx1 = contactsInLocalFrame[tId].bodyId1;
		int idx2 = contactsInLocalFrame[tId].bodyId2;

		Coord c1 = pos[idx1];
		Matrix rot1 = rotMat[idx1];

		constraints[tId].bodyId1 = idx1;
		constraints[tId].bodyId2 = idx2;
		constraints[tId].pos1 = rot1 * contactsInLocalFrame[tId].pos1 + c1;
		constraints[tId].normal1 = contactsInLocalFrame[tId].normal1;

		if (idx2 != INVALID)
		{
			Coord c2 = pos[idx2];
			Matrix rot2 = rotMat[idx2];
			constraints[tId].pos2 = rot2 * contactsInLocalFrame[tId].pos2 + c2;
			constraints[tId].normal2 = contactsInLocalFrame[tId].normal2;
		}
		else
		{
			constraints[tId].pos2 = contactsInLocalFrame[tId].pos2;
			constraints[tId].normal2 = contactsInLocalFrame[tId].normal2;
		}

		constraints[tId].interpenetration = minimum(contactsInLocalFrame[tId].interpenetration + (constraints[tId].pos2 - constraints[tId].pos1).dot(contactsInLocalFrame[tId].normal1), 0.0f);
		constraints[tId].type = ConstraintType::CN_NONPENETRATION;

		if (hasFriction)
		{
			Coord n = constraints[tId].normal1;
			n = n.normalize();

			Coord u1, u2;

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

	void setUpContactAndFrictionConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<TContactPair<float>> contactsInLocalFrame,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat,
		bool hasFriction
	)
	{
		cuExecute(constraints.size(),
			SF_setUpContactAndFrictionConstraints,
			constraints,
			contactsInLocalFrame,
			pos,
			rotMat,
			hasFriction);
	}

	/**
	* Set up the contact constraints
	*
	* @param constraints				constraints data
	* @param contactsInLocalFrame		contacts in local coordinates
	* @param pos						position of rigids
	* @param rotMat						rotation matrix of rigids
	* This function set up the contact constraints
	*/
	template<typename Coord, typename Matrix, typename Contact, typename Constraint>
	__global__ void SF_setUpContactConstraints(
		DArray<Constraint> constraints,
		DArray<Contact> contactsInLocalFrame,
		DArray<Coord> pos,
		DArray<Matrix> rotMat
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contactsInLocalFrame.size())
			return;

		int contact_size = contactsInLocalFrame.size();

		int idx1 = contactsInLocalFrame[tId].bodyId1;
		int idx2 = contactsInLocalFrame[tId].bodyId2;

		Coord c1 = pos[idx1];
		Matrix rot1 = rotMat[idx1];

		constraints[tId].bodyId1 = idx1;
		constraints[tId].bodyId2 = idx2;
		constraints[tId].pos1 = rot1 * contactsInLocalFrame[tId].pos1 + c1;
		constraints[tId].normal1 = contactsInLocalFrame[tId].normal1;

		if (idx2 != INVALID)
		{
			Coord c2 = pos[idx2];
			Matrix rot2 = rotMat[idx2];

			constraints[tId].pos2 = rot2 * contactsInLocalFrame[tId].pos2 + c2;
			constraints[tId].normal2 = contactsInLocalFrame[tId].normal2;
			constraints[tId].interpenetration = (constraints[tId].pos2 - constraints[tId].pos1).dot(constraints[tId].normal1) + contactsInLocalFrame[tId].interpenetration;
		}
		else
		{
			Real dist = (contactsInLocalFrame[tId].pos2 - constraints[tId].pos1).dot(constraints[tId].normal1) + contactsInLocalFrame[tId].interpenetration;
			constraints[tId].interpenetration = dist;
		}

		constraints[tId].type = ConstraintType::CN_NONPENETRATION;
	}

	void setUpContactConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<TContactPair<float>> contactsInLocalFrame,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat
	)
	{
		cuExecute(constraints.size(),
			SF_setUpContactConstraints,
			constraints,
			contactsInLocalFrame,
			pos,
			rotMat);
	}

	/**
	* Set up the ball and socket constraints
	*
	* @param constraints				constraints data
	* @param joints						joints data
	* @param pos						position of rigids
	* @param rotMat						rotation matrix of rigids
	* @param begin_index				begin index of ball and socket joints constraints in array
	* This function set up the ball and socket joint constraints
	*/
	template<typename Joint, typename Constraint, typename Coord, typename Matrix>
	__global__ void SF_setUpBallAndSocketJointConstraints(
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

	void setUpBallAndSocketJointConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<BallAndSocketJoint<float>> joints,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat,
		int begin_index
	)
	{
		cuExecute(constraints.size(),
			SF_setUpBallAndSocketJointConstraints,
			constraints,
			joints,
			pos,
			rotMat,
			begin_index);
	}

	/**
	* Set up the slider constraints
	*
	* @param constraints				constraints data
	* @param joints						joints data
	* @param pos						position of rigids
	* @param rotMat						rotation matrix of rigids
	* @param begin_index				begin index of slider constraints in array
	* This function set up the slider joint constraints
	*/
	template<typename Joint, typename Constraint, typename Coord, typename Matrix>
	__global__ void SF_setUpSliderJointConstraints(
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
			n1 = Coord(0, n[2], -n[1]);
			n1 = n1.normalize();
		}
		else if (abs(n[0]) > EPSILON)
		{
			n1 = Coord(n[2], 0, -n[0]);
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

	void setUpSliderJointConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<SliderJoint<float>> joints,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat,
		int begin_index
	)
	{
		cuExecute(constraints.size(),
			SF_setUpSliderJointConstraints,
			constraints,
			joints,
			pos,
			rotMat,
			begin_index);
	}

	/**
	* Set up the hinge constraints
	*
	* @param constraints				constraints data
	* @param joints						joints data
	* @param pos						position of rigids
	* @param rotMat						rotation matrix of rigids
	* @oaran rotation_q					quaterion of rigids
	* @param begin_index				begin index of hinge constraints in array
	* This function set up the hinge joint constraints
	*/
	template<typename Joint, typename Constraint, typename Coord, typename Matrix, typename Quat>
	__global__ void SF_setUpHingeJointConstraints(
		DArray<Constraint> constraints,
		DArray<Joint> joints,
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

	void setUpHingeJointConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<HingeJoint<float>> joints,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat,
		DArray<Quat1f> rotation_q,
		int begin_index
	)
	{
		cuExecute(constraints.size(),
			SF_setUpHingeJointConstraints,
			constraints,
			joints,
			pos,
			rotMat,
			rotation_q,
			begin_index);
	}

	/**
	* Set up the fixed joint constraints
	*
	* @param constraints				constraints data
	* @param joints						joints data
	* @param rotMat						rotation matrix of rigids
	* @param begin_index				begin index of fixed constraints in array
	* This function set up the fixed joint constraints
	*/
	template<typename Joint, typename Constraint, typename Matrix>
	__global__ void SF_setUpFixedJointConstraints(
		DArray<Constraint> constraints,
		DArray<Joint> joints,
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

	void setUpFixedJointConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<FixedJoint<float>> joints,
		DArray<Mat3f> rotMat,
		int begin_index
	)
	{
		cuExecute(constraints.size(),
			SF_setUpFixedJointConstraints,
			constraints,
			joints,
			rotMat,
			begin_index);
	}

	/**
	* Set up the point joint constraints
	*
	* @param constraints				constraints data
	* @param joints						joints data
	* @param begin_index				begin index of fixed constraints in array
	* This function set up the point joint constraints
	*/
	template<typename Joint, typename Constraint, typename Coord>
	__global__ void SF_setUpPointJointConstraints(
		DArray<Constraint> constraints,
		DArray<Joint> joints,
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

	void setUpPointJointConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<PointJoint<float>> joints,
		DArray<Vec3f> pos,
		int begin_index
	)
	{
		cuExecute(constraints.size(),
			SF_setUpPointJointConstraints,
			constraints,
			joints,
			pos,
			begin_index);
	}


	template<typename Coord, typename Constraint, typename Matrix>
	__global__ void SF_calculateK(
		DArray<Constraint> constraints,
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Coord> pos,
		DArray<Matrix> inertia,
		DArray<Real> mass,
		DArray<Real> K_1,
		DArray<Mat2f> K_2,
		DArray<Matrix> K_3
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= constraints.size())
			return;

		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_1)
		{
			Matrix E(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Matrix r1x(0.0f, -r1[2], r1[1], r1[2], 0, -r1[0], -r1[1], r1[0], 0);
			Matrix r2x(0.0f, -r2[2], r2[1], r2[2], 0, -r2[0], -r2[1], r2[0], 0);
			Matrix K = (1 / mass[idx1]) * E + (r1x * inertia[idx1].inverse()) * r1x.transpose() + (1 / mass[idx2]) * E + (r2x * inertia[idx2].inverse()) * r2x.transpose();
			K_3[tId] = K;
			K_3[tId + 1] = K;
			K_3[tId + 2] = K;
		}

		else if (constraints[tId].type == ConstraintType::CN_ALLOW_ROT1D_1)
		{
			Coord a1 = constraints[tId].axis;
			Coord b2 = constraints[tId].pos1;
			Coord c2 = constraints[tId].pos2;
			Coord b2_c_a1 = b2.cross(a1);
			Coord c2_c_a1 = c2.cross(a1);
			Real a = b2_c_a1.dot(inertia[idx1].inverse() * b2_c_a1) + b2_c_a1.dot(inertia[idx2].inverse() * b2_c_a1);
			Real b = b2_c_a1.dot(inertia[idx1].inverse() * c2_c_a1) + b2_c_a1.dot(inertia[idx2].inverse() * c2_c_a1);
			Real c = c2_c_a1.dot(inertia[idx1].inverse() * b2_c_a1) + c2_c_a1.dot(inertia[idx2].inverse() * b2_c_a1);
			Real d = c2_c_a1.dot(inertia[idx1].inverse() * c2_c_a1) + c2_c_a1.dot(inertia[idx2].inverse() * c2_c_a1);
			Mat2f K(a, b, c, d);
			K_2[tId] = K;
			K_2[tId + 1] = K;
		}

		else if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_1)
		{
			Matrix K = (1 / mass[idx1]) * Matrix(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
			K_3[tId] = K;
			K_3[tId + 1] = K;
			K_3[tId + 2] = K;
		}

		else if (constraints[tId].type == ConstraintType::CN_ANCHOR_TRANS_1)
		{
			Coord r1 = constraints[tId].pos1;
			Coord r2 = constraints[tId].pos2;
			Coord n1 = constraints[tId].normal1;
			Coord n2 = constraints[tId].normal2;
			Coord u = pos[idx2] + r2 - pos[idx1] - r1;
			Coord r1u_c_n1 = (r1 + u).cross(n1);
			Coord r1u_c_n2 = (r1 + u).cross(n2);
			Coord r2_c_n1 = r2.cross(n1);
			Coord r2_c_n2 = r2.cross(n2);
			Real a = 1 / mass[idx1] + 1 / mass[idx2] + r1u_c_n1.dot(inertia[idx1].inverse() * r1u_c_n1) + r2_c_n1.dot(inertia[idx2].inverse() * r2_c_n1);
			Real b = r1u_c_n1.dot(inertia[idx1].inverse() * r1u_c_n2) + r2_c_n1.dot(inertia[idx2].inverse() * r2_c_n2);
			Real c = r1u_c_n2.dot(inertia[idx1].inverse() * r1u_c_n1) + r2_c_n2.dot(inertia[idx2].inverse() * r2_c_n1);
			Real d = 1 / mass[idx1] + 1 / mass[idx2] + r1u_c_n2.dot(inertia[idx1].inverse() * r1u_c_n2) + r2_c_n2.dot(inertia[idx2].inverse() * r2_c_n2);
			Mat2f K(a, b, c, d);
			K_2[tId] = K;
			K_2[tId + 1] = K;
		}

		else if (constraints[tId].type == ConstraintType::CN_BAN_ROT_1)
		{
			Matrix K = inertia[idx1].inverse() + inertia[idx2].inverse();
			K_3[tId] = K;
			K_3[tId + 1] = K;
			K_3[tId + 2] = K;
		}

		else
		{
			Real K = J[4 * tId].dot(B[4 * tId]) + J[4 * tId + 1].dot(B[4 * tId + 1]) + J[4 * tId + 2].dot(B[4 * tId + 2]) + J[4 * tId + 3].dot(B[4 * tId + 3]);
			K_1[tId] = K;
		}

	}

	void calculateK(
		DArray<TConstraintPair<float>> constraints,
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<Vec3f> pos,
		DArray<Mat3f> inertia,
		DArray<float> mass,
		DArray<float> K_1,
		DArray<Mat2f> K_2,
		DArray<Mat3f> K_3
	)
	{
		cuExecute(constraints.size(),
			SF_calculateK,
			constraints,
			J,
			B,
			pos,
			inertia,
			mass,
			K_1,
			K_2,
			K_3);
	}

	/**
	* take one Jacobi Iteration
	* @param lambda			
	* @param impulse				
	* @param J			
	* @param B		
	* @param eta			
	* @param constraints			
	* @param nbq	
	* @param K_1	
	* @param K_2
	* @param K_3 
	* @param mass
	* @param mu
	* @param g
	* @param dt
	* This function take one Jacobi Iteration to calculate constrain impulse
	*/
	template<typename Real, typename Coord, typename Constraint, typename Matrix3, typename Matrix2>
	__global__ void SF_JacobiIteration(
		DArray<Real> lambda,
		DArray<Coord> impulse,
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Real> eta,
		DArray<Constraint> constraints,
		DArray<int> nbq,
		DArray<Real> K_1,
		DArray<Matrix2> K_2,
		DArray<Matrix3> K_3,
		DArray<Real> mass,
		Real mu,
		Real g,
		Real dt
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= constraints.size())
			return;


		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		int stepInverse = 0;
		if (constraints[tId].type == ConstraintType::CN_FRICTION || constraints[tId].type == ConstraintType::CN_NONPENETRATION)
		{
			if (idx2 != INVALID)
			{
				stepInverse = nbq[idx1] + nbq[idx2];
			}
			else
			{
				stepInverse = nbq[idx1];
			}
		}
		else
		{
			stepInverse = 5;
		}

		Real omega = Real(1) / stepInverse;

		if (constraints[tId].type == ConstraintType::CN_FRICTION || constraints[tId].type == ConstraintType::CN_NONPENETRATION)
		{
			Real tmp = eta[tId];
			tmp -= J[4 * tId].dot(impulse[idx1 * 2]);
			tmp -= J[4 * tId + 1].dot(impulse[idx1 * 2 + 1]);
			if (idx2 != INVALID)
			{
				tmp -= J[4 * tId + 2].dot(impulse[idx2 * 2]);
				tmp -= J[4 * tId + 3].dot(impulse[idx2 * 2 + 1]);
			}
			Real delta_lambda = 0;
			if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
			{
				Real lambda_new = maximum(0.0f, lambda[tId] + (tmp / (K_1[tId] * stepInverse)));
				delta_lambda = lambda_new - lambda[tId];
			}
			if (constraints[tId].type == ConstraintType::CN_FRICTION)
			{
				Real mass_avl = mass[idx1];
				Real lambda_new = minimum(maximum(lambda[tId] + (tmp / (K_1[tId] * stepInverse)), -mu * mass_avl * g * dt), mu * mass_avl * g * dt);
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

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_1 || constraints[tId].type == ConstraintType::CN_BAN_ROT_1)
		{
			Coord tmp(eta[tId], eta[tId + 1], eta[tId + 2]);
			for (int i = 0; i < 3; i++)
			{
				tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]) + J[4 * (tId + i) + 2].dot(impulse[idx2 * 2]);
				tmp[i] -= J[4 * (tId + i) + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * (tId + i) + 3].dot(impulse[idx2 * 2 + 1]);
			}

			Coord delta_lambda = omega * (K_3[tId].inverse() * tmp);

			for (int i = 0; i < 3; i++)
			{
				atomicAdd(&impulse[idx1 * 2][0], B[4 * (tId + i)][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][1], B[4 * (tId + i)][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][2], B[4 * (tId + i)][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx1 * 2 + 1][0], B[4 * (tId + i) + 1][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][1], B[4 * (tId + i) + 1][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][2], B[4 * (tId + i) + 1][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx2 * 2][0], B[4 * (tId + i) + 2][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2][1], B[4 * (tId + i) + 2][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2][2], B[4 * (tId + i) + 2][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx2 * 2 + 1][0], B[4 * (tId + i) + 3][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2 + 1][1], B[4 * (tId + i) + 3][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2 + 1][2], B[4 * (tId + i) + 3][2] * delta_lambda[i]);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_ALLOW_ROT1D_1 || constraints[tId].type == ConstraintType::CN_ANCHOR_TRANS_1)
		{
			Vec2f tmp(eta[tId], eta[tId + 1]);

			for (int i = 0; i < 2; i++)
			{
				tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]) + J[4 * (tId + i) + 2].dot(impulse[idx2 * 2]);
				tmp[i] -= J[4 * (tId + i) + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * (tId + i) + 3].dot(impulse[idx2 * 2 + 1]);
			}

			Vec2f delta_lambda = omega * (K_2[tId].inverse() * tmp);

			for (int i = 0; i < 2; i++)
			{
				atomicAdd(&impulse[idx1 * 2][0], B[4 * (tId + i)][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][1], B[4 * (tId + i)][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][2], B[4 * (tId + i)][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx1 * 2 + 1][0], B[4 * (tId + i) + 1][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][1], B[4 * (tId + i) + 1][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][2], B[4 * (tId + i) + 1][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx2 * 2][0], B[4 * (tId + i) + 2][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2][1], B[4 * (tId + i) + 2][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2][2], B[4 * (tId + i) + 2][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx2 * 2 + 1][0], B[4 * (tId + i) + 3][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2 + 1][1], B[4 * (tId + i) + 3][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2 + 1][2], B[4 * (tId + i) + 3][2] * delta_lambda[i]);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MIN || constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MAX || constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MOTER || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MIN || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MAX || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MOTER)
		{
			Real tmp = eta[tId];
			tmp -= J[4 * tId].dot(impulse[idx1 * 2]) + J[4 * tId + 2].dot(impulse[idx2 * 2]);
			tmp -= J[4 * tId + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * tId + 3].dot(impulse[idx2 * 2 + 1]);
			if (K_1[tId] > 0)
			{
				Real delta_lambda = tmp / (K_1[tId] * stepInverse);
				lambda[tId] += delta_lambda;
				atomicAdd(&impulse[idx1 * 2][0], B[4 * tId][0] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2][1], B[4 * tId][1] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2][2], B[4 * tId][2] * delta_lambda);

				atomicAdd(&impulse[idx1 * 2 + 1][0], B[4 * tId + 1][0] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2 + 1][1], B[4 * tId + 1][1] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2 + 1][2], B[4 * tId + 1][2] * delta_lambda);

				atomicAdd(&impulse[idx2 * 2][0], B[4 * tId + 2][0] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2][1], B[4 * tId + 2][1] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2][2], B[4 * tId + 2][2] * delta_lambda);

				atomicAdd(&impulse[idx2 * 2 + 1][0], B[4 * tId + 3][0] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2 + 1][1], B[4 * tId + 3][1] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2 + 1][2], B[4 * tId + 3][2] * delta_lambda);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_1)
		{
			Coord tmp(eta[tId], eta[tId + 1], eta[tId + 2]);
			for (int i = 0; i < 3; i++)
			{
				tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]);
			}

			Coord delta_lambda = omega * (K_3[tId].inverse() * tmp);
			for (int i = 0; i < 3; i++)
			{
				atomicAdd(&impulse[idx1 * 2][0], B[4 * (tId + i)][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][1], B[4 * (tId + i)][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][2], B[4 * (tId + i)][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx1 * 2 + 1][0], B[4 * (tId + i) + 1][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][1], B[4 * (tId + i) + 1][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][2], B[4 * (tId + i) + 1][2] * delta_lambda[i]);
			}
		}
	}

	template<typename Real, typename Coord, typename Constraint, typename Matrix3, typename Matrix2>
	__global__ void SF_JacobiIterationForNJS(
		DArray<Real> lambda,
		DArray<Coord> impulse,
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Real> eta,
		DArray<Constraint> constraints,
		DArray<int> nbq,
		DArray<Real> K_1,
		DArray<Matrix2> K_2,
		DArray<Matrix3> K_3
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= constraints.size())
			return;


		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

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
			stepInverse = 3;
		}

		Real omega = Real(1) / stepInverse;

		if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
		{
			Real tmp = eta[tId];
			tmp -= J[4 * tId].dot(impulse[idx1 * 2]);
			tmp -= J[4 * tId + 1].dot(impulse[idx1 * 2 + 1]);
			if (idx2 != INVALID)
			{
				tmp -= J[4 * tId + 2].dot(impulse[idx2 * 2]);
				tmp -= J[4 * tId + 3].dot(impulse[idx2 * 2 + 1]);
			}
			Real delta_lambda = 0;
			Real lambda_new = maximum(0.0f, lambda[tId] + (tmp / (K_1[tId] * stepInverse)));
			delta_lambda = lambda_new - lambda[tId];
			

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

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_1 || constraints[tId].type == ConstraintType::CN_BAN_ROT_1)
		{
			Coord tmp(eta[tId], eta[tId + 1], eta[tId + 2]);
			for (int i = 0; i < 3; i++)
			{
				tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]) + J[4 * (tId + i) + 2].dot(impulse[idx2 * 2]);
				tmp[i] -= J[4 * (tId + i) + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * (tId + i) + 3].dot(impulse[idx2 * 2 + 1]);
			}

			Coord delta_lambda = omega * (K_3[tId].inverse() * tmp);

			for (int i = 0; i < 3; i++)
			{
				atomicAdd(&impulse[idx1 * 2][0], B[4 * (tId + i)][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][1], B[4 * (tId + i)][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][2], B[4 * (tId + i)][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx1 * 2 + 1][0], B[4 * (tId + i) + 1][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][1], B[4 * (tId + i) + 1][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][2], B[4 * (tId + i) + 1][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx2 * 2][0], B[4 * (tId + i) + 2][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2][1], B[4 * (tId + i) + 2][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2][2], B[4 * (tId + i) + 2][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx2 * 2 + 1][0], B[4 * (tId + i) + 3][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2 + 1][1], B[4 * (tId + i) + 3][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2 + 1][2], B[4 * (tId + i) + 3][2] * delta_lambda[i]);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_ALLOW_ROT1D_1 || constraints[tId].type == ConstraintType::CN_ANCHOR_TRANS_1)
		{
			Vec2f tmp(eta[tId], eta[tId + 1]);

			for (int i = 0; i < 2; i++)
			{
				tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]) + J[4 * (tId + i) + 2].dot(impulse[idx2 * 2]);
				tmp[i] -= J[4 * (tId + i) + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * (tId + i) + 3].dot(impulse[idx2 * 2 + 1]);
			}

			Vec2f delta_lambda = omega * (K_2[tId].inverse() * tmp);

			for (int i = 0; i < 2; i++)
			{
				atomicAdd(&impulse[idx1 * 2][0], B[4 * (tId + i)][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][1], B[4 * (tId + i)][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][2], B[4 * (tId + i)][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx1 * 2 + 1][0], B[4 * (tId + i) + 1][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][1], B[4 * (tId + i) + 1][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][2], B[4 * (tId + i) + 1][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx2 * 2][0], B[4 * (tId + i) + 2][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2][1], B[4 * (tId + i) + 2][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2][2], B[4 * (tId + i) + 2][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx2 * 2 + 1][0], B[4 * (tId + i) + 3][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2 + 1][1], B[4 * (tId + i) + 3][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2 + 1][2], B[4 * (tId + i) + 3][2] * delta_lambda[i]);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MIN || constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MAX || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MIN || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MAX)
		{
			Real tmp = eta[tId];
			tmp -= J[4 * tId].dot(impulse[idx1 * 2]) + J[4 * tId + 2].dot(impulse[idx2 * 2]);
			tmp -= J[4 * tId + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * tId + 3].dot(impulse[idx2 * 2 + 1]);
			if (K_1[tId] > 0)
			{
				Real delta_lambda = tmp / (K_1[tId] * stepInverse);
				lambda[tId] += delta_lambda;
				atomicAdd(&impulse[idx1 * 2][0], B[4 * tId][0] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2][1], B[4 * tId][1] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2][2], B[4 * tId][2] * delta_lambda);

				atomicAdd(&impulse[idx1 * 2 + 1][0], B[4 * tId + 1][0] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2 + 1][1], B[4 * tId + 1][1] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2 + 1][2], B[4 * tId + 1][2] * delta_lambda);

				atomicAdd(&impulse[idx2 * 2][0], B[4 * tId + 2][0] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2][1], B[4 * tId + 2][1] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2][2], B[4 * tId + 2][2] * delta_lambda);

				atomicAdd(&impulse[idx2 * 2 + 1][0], B[4 * tId + 3][0] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2 + 1][1], B[4 * tId + 3][1] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2 + 1][2], B[4 * tId + 3][2] * delta_lambda);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_1)
		{
			Coord tmp(eta[tId], eta[tId + 1], eta[tId + 2]);
			for (int i = 0; i < 3; i++)
			{
				tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]);
			}

			Coord delta_lambda = omega * (K_3[tId].inverse() * tmp);
			for (int i = 0; i < 3; i++)
			{
				atomicAdd(&impulse[idx1 * 2][0], B[4 * (tId + i)][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][1], B[4 * (tId + i)][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][2], B[4 * (tId + i)][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx1 * 2 + 1][0], B[4 * (tId + i) + 1][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][1], B[4 * (tId + i) + 1][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][2], B[4 * (tId + i) + 1][2] * delta_lambda[i]);
			}
		}
	}

	template<typename Real, typename Coord, typename Constraint, typename Matrix3, typename Matrix2>
	__global__ void SF_JacobiIterationForSoft(
		DArray<Real> lambda,
		DArray<Coord> impulse,
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Real> eta,
		DArray<Constraint> constraints,
		DArray<int> nbq,
		DArray<Real> K_1,
		DArray<Matrix2> K_2,
		DArray<Matrix3> K_3,
		DArray<Real> mass,
		Real mu,
		Real g,
		Real dt,
		Real zeta,
		Real hertz
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= constraints.size())
			return;


		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		Real ome = 2.0f * M_PI * hertz;
		Real a1 = 2.0 * zeta + ome * dt;
		Real a2 = dt * ome * a1;
		Real a3 = 1.0f / (1.0f + a2);
		Real massCoeff = a2 * a3;
		Real impulseCoeff = a3;

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
			stepInverse = 5;
		}

		Real omega = Real(1) / stepInverse;

		if (constraints[tId].type == ConstraintType::CN_FRICTION || constraints[tId].type == ConstraintType::CN_NONPENETRATION)
		{
			Real tmp = eta[tId];
			tmp -= J[4 * tId].dot(impulse[idx1 * 2]);
			tmp -= J[4 * tId + 1].dot(impulse[idx1 * 2 + 1]);
			if (idx2 != INVALID)
			{
				tmp -= J[4 * tId + 2].dot(impulse[idx2 * 2]);
				tmp -= J[4 * tId + 3].dot(impulse[idx2 * 2 + 1]);
			}
			Real delta_lambda = 0;
			if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
			{
				Real lambda_new = maximum(0.0f, lambda[tId] + ((massCoeff * tmp) / (K_1[tId] * stepInverse) - impulseCoeff * lambda[tId]));
				delta_lambda = lambda_new - lambda[tId];
			}
			if (constraints[tId].type == ConstraintType::CN_FRICTION)
			{
				Real mass_avl = mass[idx1];
				Real lambda_new = minimum(maximum(lambda[tId] + (tmp / (K_1[tId] * stepInverse)), -mu * mass_avl * g * dt), mu * mass_avl * g * dt);
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

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_1 || constraints[tId].type == ConstraintType::CN_BAN_ROT_1)
		{
			Coord tmp(eta[tId], eta[tId + 1], eta[tId + 2]);
			for (int i = 0; i < 3; i++)
			{
				tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]) + J[4 * (tId + i) + 2].dot(impulse[idx2 * 2]);
				tmp[i] -= J[4 * (tId + i) + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * (tId + i) + 3].dot(impulse[idx2 * 2 + 1]);
			}
			Coord oldLambda(lambda[tId], lambda[tId + 1], lambda[tId + 2]);
			Coord delta_lambda = massCoeff * omega * (K_3[tId].inverse() * tmp) - impulseCoeff * oldLambda;


			for (int i = 0; i < 3; i++)
			{
				lambda[tId + i] += delta_lambda[i];

				atomicAdd(&impulse[idx1 * 2][0], B[4 * (tId + i)][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][1], B[4 * (tId + i)][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][2], B[4 * (tId + i)][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx1 * 2 + 1][0], B[4 * (tId + i) + 1][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][1], B[4 * (tId + i) + 1][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][2], B[4 * (tId + i) + 1][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx2 * 2][0], B[4 * (tId + i) + 2][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2][1], B[4 * (tId + i) + 2][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2][2], B[4 * (tId + i) + 2][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx2 * 2 + 1][0], B[4 * (tId + i) + 3][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2 + 1][1], B[4 * (tId + i) + 3][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2 + 1][2], B[4 * (tId + i) + 3][2] * delta_lambda[i]);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_ALLOW_ROT1D_1 || constraints[tId].type == ConstraintType::CN_ANCHOR_TRANS_1)
		{
			Vec2f tmp(eta[tId], eta[tId + 1]);

			for (int i = 0; i < 2; i++)
			{
				tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]) + J[4 * (tId + i) + 2].dot(impulse[idx2 * 2]);
				tmp[i] -= J[4 * (tId + i) + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * (tId + i) + 3].dot(impulse[idx2 * 2 + 1]);
			}
			Vec2f oldLambda(lambda[tId], lambda[tId + 1]);
			Vec2f delta_lambda = massCoeff * omega * (K_2[tId].inverse() * tmp) - impulseCoeff * oldLambda;


			for (int i = 0; i < 2; i++)
			{
				lambda[tId + i] += delta_lambda[i];
				atomicAdd(&impulse[idx1 * 2][0], B[4 * (tId + i)][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][1], B[4 * (tId + i)][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][2], B[4 * (tId + i)][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx1 * 2 + 1][0], B[4 * (tId + i) + 1][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][1], B[4 * (tId + i) + 1][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][2], B[4 * (tId + i) + 1][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx2 * 2][0], B[4 * (tId + i) + 2][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2][1], B[4 * (tId + i) + 2][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2][2], B[4 * (tId + i) + 2][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx2 * 2 + 1][0], B[4 * (tId + i) + 3][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2 + 1][1], B[4 * (tId + i) + 3][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx2 * 2 + 1][2], B[4 * (tId + i) + 3][2] * delta_lambda[i]);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MIN || constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MAX || constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MOTER || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MIN || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MAX || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MOTER)
		{
			Real tmp = eta[tId];
			tmp -= J[4 * tId].dot(impulse[idx1 * 2]) + J[4 * tId + 2].dot(impulse[idx2 * 2]);
			tmp -= J[4 * tId + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * tId + 3].dot(impulse[idx2 * 2 + 1]);
			if (K_1[tId] > 0)
			{
				Real delta_lambda = massCoeff * tmp / (K_1[tId] * stepInverse) - impulseCoeff * lambda[tId];
				lambda[tId] += delta_lambda;
				atomicAdd(&impulse[idx1 * 2][0], B[4 * tId][0] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2][1], B[4 * tId][1] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2][2], B[4 * tId][2] * delta_lambda);

				atomicAdd(&impulse[idx1 * 2 + 1][0], B[4 * tId + 1][0] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2 + 1][1], B[4 * tId + 1][1] * delta_lambda);
				atomicAdd(&impulse[idx1 * 2 + 1][2], B[4 * tId + 1][2] * delta_lambda);

				atomicAdd(&impulse[idx2 * 2][0], B[4 * tId + 2][0] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2][1], B[4 * tId + 2][1] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2][2], B[4 * tId + 2][2] * delta_lambda);

				atomicAdd(&impulse[idx2 * 2 + 1][0], B[4 * tId + 3][0] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2 + 1][1], B[4 * tId + 3][1] * delta_lambda);
				atomicAdd(&impulse[idx2 * 2 + 1][2], B[4 * tId + 3][2] * delta_lambda);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_1)
		{
			Coord tmp(eta[tId], eta[tId + 1], eta[tId + 2]);
			for (int i = 0; i < 3; i++)
			{
				tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]);
			}
			Coord oldLambda(lambda[tId], lambda[tId + 1], lambda[tId + 2]);
			Coord delta_lambda = massCoeff * omega * (K_3[tId].inverse() * tmp) - impulseCoeff * oldLambda;
			for (int i = 0; i < 3; i++)
			{
				lambda[tId + i] += delta_lambda[i];
				atomicAdd(&impulse[idx1 * 2][0], B[4 * (tId + i)][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][1], B[4 * (tId + i)][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][2], B[4 * (tId + i)][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx1 * 2 + 1][0], B[4 * (tId + i) + 1][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][1], B[4 * (tId + i) + 1][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][2], B[4 * (tId + i) + 1][2] * delta_lambda[i]);
			}
		}
	}

	void JacobiIteration(
		DArray<float> lambda,
		DArray<Vec3f> impulse,
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<float> eta,
		DArray<TConstraintPair<float>> constraints,
		DArray<int> nbq,
		DArray<float> K_1,
		DArray<Mat2f> K_2,
		DArray<Mat3f> K_3,
		DArray<float> mass,
		float mu,
		float g,
		float dt
	)
	{
		cuExecute(constraints.size(),
			SF_JacobiIteration,
			lambda,
			impulse,
			J,
			B,
			eta,
			constraints,
			nbq,
			K_1,
			K_2,
			K_3,
			mass,
			mu,
			g,
			dt);
	}

	void JacobiIterationForSoft(
		DArray<float> lambda,
		DArray<Vec3f> impulse,
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<float> eta,
		DArray<TConstraintPair<float>> constraints,
		DArray<int> nbq,
		DArray<float> K_1,
		DArray<Mat2f> K_2,
		DArray<Mat3f> K_3,
		DArray<float> mass,
		float mu,
		float g,
		float dt,
		float zeta,
		float hertz
	)
	{
		cuExecute(constraints.size(),
			SF_JacobiIterationForSoft,
			lambda,
			impulse,
			J,
			B,
			eta,
			constraints,
			nbq,
			K_1,
			K_2,
			K_3,
			mass,
			mu,
			g,
			dt,
			zeta,
			hertz);
	}

	void JacobiIterationForNJS(
		DArray<float> lambda,
		DArray<Vec3f> impulse,
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<float> eta,
		DArray<TConstraintPair<float>> constraints,
		DArray<int> nbq,
		DArray<float> K_1,
		DArray<Mat2f> K_2,
		DArray<Mat3f> K_3
	)
	{
		cuExecute(constraints.size(),
			SF_JacobiIterationForNJS,
			lambda,
			impulse,
			J,
			B,
			eta,
			constraints,
			nbq,
			K_1,
			K_2,
			K_3);
	}

	/**
	* Set up Gravity
	* @param impulse_ext
	* @param g
	* @param dt
	* This function set up gravity
	*/
	template<typename Coord>
	__global__ void SF_setUpGravity(
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

	void setUpGravity(
		DArray<Vec3f> impulse_ext,
		float g,
		float dt
	)
	{
		cuExecute(impulse_ext.size() / 2,
			SF_setUpGravity,
			impulse_ext,
			g,
			dt);
	}


	
	
}