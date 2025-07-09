#include "SharedFuncsForRigidBody.h"

namespace dyno
{
	  __global__ void SF_ApplyTransform(
		DArrayList<Transform3f> instanceTransform,
		const DArray<Vec3f> translate,
		const DArray<Mat3f> rotation,
		const DArray<Pair<uint, uint>> binding,
		const DArray<int> bindingtag)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= rotation.size())
			return;
		if (bindingtag[tId] == 0)
			return;

		Pair<uint, uint> pair = binding[tId];

		Mat3f rot = rotation[tId];

		Transform3f ti = Transform3f(translate[tId], rot);

		instanceTransform[pair.first][pair.second] = ti;
	}

	void ApplyTransform(
		DArrayList<Transform3f>& instanceTransform, 
		const DArray<Vec3f>& translate,
		const DArray<Mat3f>& rotation,
		const DArray<Pair<uint, uint>>& binding,
		const DArray<int>& bindingtag)
	{
		cuExecute(rotation.size(),
			SF_ApplyTransform,
			instanceTransform,
			translate,
			rotation,
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
		DArray<Attribute> attribute,
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

		if (attribute[tId].isDynamic())
		{
			velocity[tId] += impulse[2 * tId];
			angular_velocity[tId] += impulse[2 * tId + 1];
			//Damping
			velocity[tId] *= 1.0f / (1.0f + dt * linearDamping);
			angular_velocity[tId] *= 1.0f / (1.0f + dt * angularDamping);
		}
	}

	void updateVelocity(
		DArray<Attribute> attribute,
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
			attribute,
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
	__global__ void SF_updateGestureNoSelf(
		DArray<Attribute> attribute,
		DArray<Vec3f> initPos,
		DArray<Vec3f> pos,
		DArray<Quat1f> rotQuat,
		DArray<Quat1f> initRotQuat,
		DArray<Mat3f> rotMat,
		DArray<Mat3f> initRotMat,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		float dt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= pos.size())
			return;

		if (!attribute[tId].isFixed())
		{
			pos[tId] = initPos[tId] + velocity[tId] * dt;

			initRotQuat[tId].normalize();
			rotQuat[tId] = initRotQuat[tId] + dt * 0.5f *
				Quat1f(angular_velocity[tId][0], angular_velocity[tId][1], angular_velocity[tId][2], 0.0)
				* (initRotQuat[tId]);
			rotQuat[tId].normalize();

			rotMat[tId] = rotQuat[tId].toMatrix3x3();
		}
	}



	__global__ void SF_updateGesture(
		DArray<Attribute> attribute,
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

		if (!attribute[tId].isFixed())
		{
			pos[tId] += velocity[tId] * dt;

			rotQuat[tId] = rotQuat[tId].normalize();

			rotQuat[tId] += dt * 0.5f *
				Quat1f(angular_velocity[tId][0], angular_velocity[tId][1], angular_velocity[tId][2], 0.0)
				* (rotQuat[tId]);

			rotQuat[tId] = rotQuat[tId].normalize();

			rotMat[tId] = rotQuat[tId].toMatrix3x3();

			inertia[tId] = rotMat[tId] * inertia_init[tId] * rotMat[tId].inverse();
		}
	}

	void updateGesture(
		DArray<Attribute> attribute,
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
			attribute,
			pos,
			rotQuat,
			rotMat,
			inertia,
			velocity,
			angular_velocity,
			inertia_init,
			dt);
	}

	void updateGestureNoSelf(
		DArray<Attribute> attribute,
		DArray<Vec3f> initPos,
		DArray<Vec3f> pos,
		DArray<Quat1f> initRotQuat,
		DArray<Quat1f> rotQuat,
		DArray<Mat3f> initRotMat,
		DArray<Mat3f> rotMat,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		float dt
	)
	{
		cuExecute(pos.size(),
			SF_updateGestureNoSelf,
			attribute,
			initPos,
			pos,
			initRotQuat,
			rotQuat,
			initRotMat,
			rotMat,
			velocity,
			angular_velocity,
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
			if (idx2 != INVALID)
			{
				J[4 * tId + 2] = Coord(1, 0, 0);
				J[4 * tId + 3] = Coord(0, r2[2], -r2[1]);
			}

			B[4 * tId] = Coord(-1, 0, 0) / mass[idx1];
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(0, -r1[2], r1[1]);
			if (idx2 != INVALID)
			{
				B[4 * tId + 2] = Coord(1, 0, 0) / mass[idx2];
				B[4 * tId + 3] = inertia[idx2].inverse() * Coord(0, r2[2], -r2[1]);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_2)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;

			J[4 * tId] = Coord(0, -1, 0);
			J[4 * tId + 1] = Coord(r1[2], 0, -r1[0]);
			if (idx2 != INVALID)
			{
				J[4 * tId + 2] = Coord(0, 1, 0);
				J[4 * tId + 3] = Coord(-r2[2], 0, r2[0]);
			}

			B[4 * tId] = Coord(0, -1, 0) / mass[idx1];
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(r1[2], 0, -r1[0]);
			if (idx2 != INVALID)
			{
				B[4 * tId + 2] = Coord(0, 1, 0) / mass[idx2];
				B[4 * tId + 3] = inertia[idx2].inverse() * Coord(-r2[2], 0, r2[0]);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_3)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;

			J[4 * tId] = Coord(0, 0, -1);
			J[4 * tId + 1] = Coord(-r1[1], r1[0], 0);
			if (idx2 != INVALID)
			{
				J[4 * tId + 2] = Coord(0, 0, 1);
				J[4 * tId + 3] = Coord(r2[1], -r2[0], 0);
			}

			B[4 * tId] = Coord(0, 0, -1) / mass[idx1];
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(-r1[1], r1[0], 0);
			if (idx2 != INVALID)
			{
				B[4 * tId + 2] = Coord(0, 0, 1) / mass[idx2];
				B[4 * tId + 3] = inertia[idx2].inverse() * Coord(r2[1], -r2[0], 0);
			}	
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
			if (idx2 != INVALID)
			{
				J[4 * tId + 2] = Coord(0);
				J[4 * tId + 3] = Coord(1, 0, 0);
			}

			B[4 * tId] = Coord(0);
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(-1, 0, 0);
			if (idx2 != INVALID)
			{
				B[4 * tId + 2] = Coord(0);
				B[4 * tId + 3] = inertia[idx2].inverse() * Coord(1, 0, 0);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_2)
		{
			J[4 * tId] = Coord(0);
			J[4 * tId + 1] = Coord(0, -1, 0);
			if (idx2 != INVALID)
			{
				J[4 * tId + 2] = Coord(0);
				J[4 * tId + 3] = Coord(0, 1, 0);
			}

			B[4 * tId] = Coord(0);
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(0, -1, 0);
			if (idx2 != INVALID)
			{
				B[4 * tId + 2] = Coord(0);
				B[4 * tId + 3] = inertia[idx2].inverse() * Coord(0, 1, 0);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_3)
		{
			J[4 * tId] = Coord(0);
			J[4 * tId + 1] = Coord(0, 0, -1);
			if (idx2 != INVALID)
			{
				J[4 * tId + 2] = Coord(0);
				J[4 * tId + 3] = Coord(0, 0, 1);
			}

			B[4 * tId] = Coord(0);
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(0, 0, -1);
			if (idx2 != INVALID)
			{
				B[4 * tId + 2] = Coord(0);
				B[4 * tId + 3] = inertia[idx2].inverse() * Coord(0, 0, 1);
			}
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
			if (idx2 != INVALID)
			{
				J[4 * tId + 2] = Coord(1, 0, 0);
				J[4 * tId + 3] = Coord(0, r2[2], -r2[1]);
			}

			B[4 * tId] = Coord(-1, 0, 0) / mass[idx1];
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(0, -r1[2], r1[1]);
			if (idx2 != INVALID)
			{
				B[4 * tId + 2] = Coord(1, 0, 0) / mass[idx2];
				B[4 * tId + 3] = inertia[idx2].inverse() * Coord(0, r2[2], -r2[1]);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_2)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;

			J[4 * tId] = Coord(0, -1, 0);
			J[4 * tId + 1] = Coord(r1[2], 0, -r1[0]);
			if (idx2 != INVALID)
			{
				J[4 * tId + 2] = Coord(0, 1, 0);
				J[4 * tId + 3] = Coord(-r2[2], 0, r2[0]);
			}

			B[4 * tId] = Coord(0, -1, 0) / mass[idx1];
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(r1[2], 0, -r1[0]);
			if (idx2 != INVALID)
			{
				B[4 * tId + 2] = Coord(0, 1, 0) / mass[idx2];
				B[4 * tId + 3] = inertia[idx2].inverse() * Coord(-r2[2], 0, r2[0]);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_3)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;

			J[4 * tId] = Coord(0, 0, -1);
			J[4 * tId + 1] = Coord(-r1[1], r1[0], 0);
			if (idx2 != INVALID)
			{
				J[4 * tId + 2] = Coord(0, 0, 1);
				J[4 * tId + 3] = Coord(r2[1], -r2[0], 0);
			}

			B[4 * tId] = Coord(0, 0, -1) / mass[idx1];
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(-r1[1], r1[0], 0);
			if (idx2 != INVALID)
			{
				B[4 * tId + 2] = Coord(0, 0, 1) / mass[idx2];
				B[4 * tId + 3] = inertia[idx2].inverse() * Coord(r2[1], -r2[0], 0);
			}
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
			if (idx2 != INVALID)
			{
				J[4 * tId + 2] = Coord(0);
				J[4 * tId + 3] = Coord(1, 0, 0);
			}

			B[4 * tId] = Coord(0);
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(-1, 0, 0);
			if (idx2 != INVALID)
			{
				B[4 * tId + 2] = Coord(0);
				B[4 * tId + 3] = inertia[idx2].inverse() * Coord(1, 0, 0);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_2)
		{
			J[4 * tId] = Coord(0);
			J[4 * tId + 1] = Coord(0, -1, 0);
			if (idx2 != INVALID)
			{
				J[4 * tId + 2] = Coord(0);
				J[4 * tId + 3] = Coord(0, 1, 0);
			}

			B[4 * tId] = Coord(0);
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(0, -1, 0);
			if (idx2 != INVALID)
			{
				B[4 * tId + 2] = Coord(0);
				B[4 * tId + 3] = inertia[idx2].inverse() * Coord(0, 1, 0);
			}
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_3)
		{
			J[4 * tId] = Coord(0);
			J[4 * tId + 1] = Coord(0, 0, -1);
			if (idx2 != INVALID)
			{
				J[4 * tId + 2] = Coord(0);
				J[4 * tId + 3] = Coord(0, 0, 1);
			}

			B[4 * tId] = Coord(0);
			B[4 * tId + 1] = inertia[idx1].inverse() * Coord(0, 0, -1);
			if (idx2 != INVALID)
			{
				B[4 * tId + 2] = Coord(0);
				B[4 * tId + 3] = inertia[idx2].inverse() * Coord(0, 0, 1);
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
		DArray<Real> errors,
		Real slop,
		Real beta,
		uint substepping,
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
			error = minimum(constraints[tId].interpenetration + slop, 0.0f) / substepping;
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_1)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord pos1 = constraints[tId].pos1;
			Coord errorVec;
			if(idx2 != INVALID)
				errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			else
				errorVec = pos1 - pos[idx1] - r1;
			error = errorVec[0];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_2)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord pos1 = constraints[tId].pos1;
			Coord errorVec;
			if (idx2 != INVALID)
				errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			else
				errorVec = pos1 - pos[idx1] - r1;
			error = errorVec[1];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_3)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord pos1 = constraints[tId].pos1;
			Coord errorVec;
			if (idx2 != INVALID)
				errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			else
				errorVec = pos1 - pos[idx1] - r1;
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
			Quat q1 = rotation_q[idx1];
			if (idx2 != INVALID)
			{
				Quat q2 = rotation_q[idx2];
				Quat q_init = constraints[tId].rotQuat;
				Quat q_error = q2 * q_init * q1.inverse();

				error = q_error.x * 2;
			}
			else
			{
				Quat diff = rotation_q[idx1] * constraints[tId].rotQuat.inverse();
				error = -diff.x * 2;
			}
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_2)
		{
			Quat q1 = rotation_q[idx1];
			if (idx2 != INVALID)
			{
				Quat q2 = rotation_q[idx2];
				Quat q_init = constraints[tId].rotQuat;
				Quat q_error = q2 * q_init * q1.inverse();

				error = q_error.y * 2;
			}
			else
			{
				Quat diff = rotation_q[idx1] * constraints[tId].rotQuat.inverse();
				error = -diff.y * 2;
			}
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_3)
		{
			Quat q1 = rotation_q[idx1];
			if (idx2 != INVALID)
			{
				Quat q2 = rotation_q[idx2];
				Quat q_init = constraints[tId].rotQuat;
				Quat q_error = q2 * q_init * q1.inverse();

				error = q_error.z * 2;
			}
			else
			{
				Quat diff = rotation_q[idx1] * constraints[tId].rotQuat.inverse();
				error = -diff.z * 2;
			}
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
		errors[tId] = error;
	}

	template<typename Coord, typename Constraint, typename Real, typename Quat>
	__global__ void SF_calculateEtaVectorWithERP(
		DArray<Real> eta,
		DArray<Coord> J,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> pos,
		DArray<Quat> rotation_q,
		DArray<Constraint> constraints,
		DArray<Real> ERP,
		Real slop,
		Real dt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= constraints.size())
			return;

		if (!constraints[tId].isValid)
			return;

		Real invDt = Real(1) / dt;
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
			Coord pos1 = constraints[tId].pos1;
			Coord errorVec;
			if (idx2 != INVALID)
				errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			else
				errorVec = pos1 - pos[idx1] - r1;
			error = errorVec[0];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_2)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord pos1 = constraints[tId].pos1;
			Coord errorVec;
			if (idx2 != INVALID)
				errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			else
				errorVec = pos1 - pos[idx1] - r1;
			error = errorVec[1];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_3)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord pos1 = constraints[tId].pos1;
			Coord errorVec;
			if (idx2 != INVALID)
				errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			else
				errorVec = pos1 - pos[idx1] - r1;
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
			Quat q1 = rotation_q[idx1];
			q1 = q1.normalize();
			if (idx2 != INVALID)
			{
				Quat q2 = rotation_q[idx2];
				q2 = q2.normalize();

				Quat q_error = q2 * q1.inverse();
				q_error = q_error.normalize();

				Real theta = 2.0 * acos(q_error.w);
				if (theta > 1e-6)
				{
					Vec3f v = (1 / sin(theta / 2.0)) * Vec3f(q_error.x, q_error.y, q_error.z);
					error = theta * v.x;
				}
				else
				{
					error = theta;
				}
			}
			else
			{
				Real theta = 2.0 * acos(q1.w);
				if (theta > 1e-6)
				{
					Vec3f v = (1 / sin(theta / 2.0)) * Vec3f(q1.x, q1.y, q1.z);
					error = theta * v.x;
				}
				else
				{
					error = theta;
				}
			}
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_2)
		{
			Quat q1 = rotation_q[idx1];
			q1 = q1.normalize();
			if (idx2 != INVALID)
			{
				Quat q2 = rotation_q[idx2];
				q2 = q2.normalize();
				Quat q_error = q2 * q1.inverse();
				q_error = q_error.normalize();
				Real theta = 2.0 * acos(q_error.w);
				if (theta > 1e-6)
				{
					Vec3f v = (1 / sin(theta / 2.0)) * Vec3f(q_error.x, q_error.y, q_error.z);
					error = theta * v.y;
				}
				else
				{
					error = theta;
				}
			}
			else
			{
				Real theta = 2.0 * acos(q1.w);
				if (theta > 1e-6)
				{
					Vec3f v = (1 / sin(theta / 2.0)) * Vec3f(q1.x, q1.y, q1.z);
					error = theta * v.y;
				}
				else
				{
					error = theta;
				}
			}
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_3)
		{
			Quat q1 = rotation_q[idx1];
			q1 = q1.normalize();
			if (idx2 != INVALID)
			{
				Quat q2 = rotation_q[idx2];
				q2 = q2.normalize();
				Quat q_error = q2 * q1.inverse();
				q_error = q_error.normalize();
				Real theta = 2.0 * acos(q_error.w);
				if (theta > 1e-6)
				{
					Vec3f v = (1 / sin(theta / 2.0)) * Vec3f(q_error.x, q_error.y, q_error.z);
					error = theta * v.z;
				}
				else
				{
					error = theta;
				}
			}
			else
			{
				Real theta = 2.0 * acos(q1.w);
				if (theta > 1e-6)
				{
					Vec3f v = (1 / sin(theta / 2.0)) * Vec3f(q1.x, q1.y, q1.z);
					error = theta * v.z;
				}
				else
				{
					error = theta;
				}
			}
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

		eta[tId] -= ERP[tId] * invDt * error;
	}

	void calculateEtaVectorForPJSBaumgarte(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Vec3f> pos,
		DArray<Quat1f> rotation_q,
		DArray<TConstraintPair<float>> constraints,
		DArray<float> errors,
		float slop,
		float beta,
		uint substepping,
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
			errors,
			slop,
			beta,
			substepping,
			dt);
	}

	void calculateEtaVectorWithERP(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Vec3f> pos,
		DArray<Quat1f> rotation_q,
		DArray<TConstraintPair<float>> constraints,
		DArray<float> ERP,
		float slop,
		float dt
	)
	{
		cuExecute(constraints.size(),
			SF_calculateEtaVectorWithERP,
			eta,
			J,
			velocity,
			angular_velocity,
			pos,
			rotation_q,
			constraints,
			ERP,
			slop,
			dt);
	}


	template<typename Coord, typename Constraint, typename Real>
	__global__ void SF_calculateEtaVectorForRelaxation(
		DArray<Real> eta,
		DArray<Coord> J,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Constraint> constraints
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


	void calculateEtaVectorForRelaxation(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray <TConstraintPair<float>> constraints
	)
	{
		cuExecute(constraints.size(),
			SF_calculateEtaVectorForRelaxation,
			eta,
			J,
			velocity,
			angular_velocity,
			constraints);
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
		Real substepping,
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
			error = minimum(constraints[tId].interpenetration + slop, 0.0f) / substepping;
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_1)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord pos1 = constraints[tId].pos1;
			Coord errorVec;
			if (idx2 != INVALID)
				errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			else
				errorVec = pos1 - pos[idx1] - r1;
			error = errorVec[0];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_2)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord pos1 = constraints[tId].pos1;
			Coord errorVec;
			if (idx2 != INVALID)
				errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			else
				errorVec = pos1 - pos[idx1] - r1;
			error = errorVec[1];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_3)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord pos1 = constraints[tId].pos1;
			Coord errorVec;
			if (idx2 != INVALID)
				errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			else
				errorVec = pos1 - pos[idx1] - r1;
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
			Quat q1 = rotation_q[idx1];
			if (idx2 != INVALID)
			{
				Quat q2 = rotation_q[idx2];
				Quat q_init = constraints[tId].rotQuat;
				Quat q_error = q2 * q_init * q1.inverse();

				error = q_error.x * 2;
			}
			else
			{
				Quat diff = rotation_q[idx1] * constraints[tId].rotQuat.inverse();
				error = -diff.x * 2;
			}
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_2)
		{
			Quat q1 = rotation_q[idx1];
			if (idx2 != INVALID)
			{
				Quat q2 = rotation_q[idx2];
				Quat q_init = constraints[tId].rotQuat;
				Quat q_error = q2 * q_init * q1.inverse();

				error = q_error.y * 2;
			}
			else
			{
				Quat diff = rotation_q[idx1] * constraints[tId].rotQuat.inverse();
				error = -diff.y * 2;
			}
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_3)
		{
			Quat q1 = rotation_q[idx1];
			if (idx2 != INVALID)
			{
				Quat q2 = rotation_q[idx2];
				Quat q_init = constraints[tId].rotQuat;
				Quat q_error = q2 * q_init * q1.inverse();

				error = q_error.z * 2;
			}
			else
			{
				Quat diff = rotation_q[idx1] * constraints[tId].rotQuat.inverse();
				error = -diff.z * 2;
			}
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
		float substepping,
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
			substepping,
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
			Coord pos1 = constraints[tId].pos1;
			Coord errorVec;
			if (idx2 != INVALID)
				errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			else
				errorVec = pos1 - pos[idx1] - r1;
			error = errorVec[0];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_2)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord pos1 = constraints[tId].pos1;
			Coord errorVec;
			if (idx2 != INVALID)
				errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			else
				errorVec = pos1 - pos[idx1] - r1;
			error = errorVec[1];
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_3)
		{
			Coord r1 = constraints[tId].normal1;
			Coord r2 = constraints[tId].normal2;
			Coord pos1 = constraints[tId].pos1;
			Coord errorVec;
			if (idx2 != INVALID)
				errorVec = pos[idx2] + r2 - pos[idx1] - r1;
			else
				errorVec = pos1 - pos[idx1] - r1;
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
			Quat q1 = rotation_q[idx1];
			q1 = q1.normalize();
			if (idx2 != INVALID)
			{
				Quat q2 = rotation_q[idx2];
				q2 = q2.normalize();

				Quat q_error = q2 * q1.inverse();
				q_error = q_error.normalize();

				Real theta = 2.0 * acos(q_error.w);
				if (theta > 1e-6)
				{
					Vec3f v = (1 / sin(theta / 2.0)) * Vec3f(q_error.x, q_error.y, q_error.z);
					error = theta * v.x;
				}
				else
				{
					error = theta;
				}
			}
			else
			{
				Real theta = 2.0 * acos(q1.w);
				if (theta > 1e-6)
				{
					Vec3f v = (1 / sin(theta / 2.0)) * Vec3f(q1.x, q1.y, q1.z);
					error = theta * v.x;
				}
				else
				{
					error = theta;
				}
			}
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_2)
		{
			Quat q1 = rotation_q[idx1];
			q1 = q1.normalize();
			if (idx2 != INVALID)
			{
				Quat q2 = rotation_q[idx2];
				q2 = q2.normalize();
				Quat q_error = q2 * q1.inverse();
				q_error = q_error.normalize();
				Real theta = 2.0 * acos(q_error.w);
				if (theta > 1e-6)
				{
					Vec3f v = (1 / sin(theta / 2.0)) * Vec3f(q_error.x, q_error.y, q_error.z);
					error = theta * v.y;
				}
				else
				{
					error = theta;
				}
			}
			else
			{
				Real theta = 2.0 * acos(q1.w);
				if (theta > 1e-6)
				{
					Vec3f v = (1 / sin(theta / 2.0)) * Vec3f(q1.x, q1.y, q1.z);
					error = theta * v.y;
				}
				else
				{
					error = theta;
				}
			}
		}

		if (constraints[tId].type == ConstraintType::CN_BAN_ROT_3)
		{
			Quat q1 = rotation_q[idx1];
			q1 = q1.normalize();
			if (idx2 != INVALID)
			{
				Quat q2 = rotation_q[idx2];
				q2 = q2.normalize();
				Quat q_error = q2 * q1.inverse();
				q_error = q_error.normalize();
				Real theta = 2.0 * acos(q_error.w);
				if (theta > 1e-6)
				{
					Vec3f v = (1 / sin(theta / 2.0)) * Vec3f(q_error.x, q_error.y, q_error.z);
					error = theta * v.z;
				}
				else
				{
					error = theta;
				}
			}
			else
			{
				Real theta = 2.0 * acos(q1.w);
				if (theta > 1e-6)
				{
					Vec3f v = (1 / sin(theta / 2.0)) * Vec3f(q1.x, q1.y, q1.z);
					error = theta * v.z;
				}
				else
				{
					error = theta;
				}
			}
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
		constraints[tId].isValid = true;

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
			constraints[tId * 2 + contact_size].isValid = true;

			constraints[tId * 2 + 1 + contact_size].bodyId1 = idx1;
			constraints[tId * 2 + 1 + contact_size].bodyId2 = idx2;
			constraints[tId * 2 + 1 + contact_size].pos1 = constraints[tId].pos1;
			constraints[tId * 2 + 1 + contact_size].pos2 = constraints[tId].pos2;
			constraints[tId * 2 + 1 + contact_size].normal1 = u2;
			constraints[tId * 2 + 1 + contact_size].normal2 = -u2;
			constraints[tId * 2 + 1 + contact_size].type = ConstraintType::CN_FRICTION;
			constraints[tId * 2 + 1 + contact_size].isValid = true;
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
		constraints[baseIndex].isValid = true;

		constraints[baseIndex + 1].bodyId1 = idx1;
		constraints[baseIndex + 1].bodyId2 = idx2;
		constraints[baseIndex + 1].normal1 = r1;
		constraints[baseIndex + 1].normal2 = r2;
		constraints[baseIndex + 1].type = ConstraintType::CN_ANCHOR_EQUAL_2;
		constraints[baseIndex + 1].isValid = true;

		constraints[baseIndex + 2].bodyId1 = idx1;
		constraints[baseIndex + 2].bodyId2 = idx2;
		constraints[baseIndex + 2].normal1 = r1;
		constraints[baseIndex + 2].normal2 = r2;
		constraints[baseIndex + 2].type = ConstraintType::CN_ANCHOR_EQUAL_3;
		constraints[baseIndex + 2].isValid = true;
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
	template<typename Joint, typename Constraint, typename Coord, typename Matrix, typename Quat>
	__global__ void SF_setUpSliderJointConstraints(
		DArray<Constraint> constraints,
		DArray<Joint> joints,
		DArray<Coord> pos,
		DArray<Matrix> rotMat,
		DArray<Quat> rotQuat,
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

		for (int i = 0; i < 5; i++)
		{
			constraints[baseIndex + i].isValid = true;
		}


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
		else
		{
			constraints[baseIndex + 5].isValid = false;
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
			constraint.rotQuat = joints[tId].q_init;
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
		DArray<Quat1f> rotQuat,
		int begin_index
	)
	{
		cuExecute(constraints.size(),
			SF_setUpSliderJointConstraints,
			constraints,
			joints,
			pos,
			rotMat,
			rotQuat,
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

		else
		{
			constraints[baseIndex + 5].isValid = false;
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

		for (int i = 0; i < 5; i++)
		{
			constraints[baseIndex + i].isValid = true;
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
	template<typename Joint, typename Constraint, typename Matrix, typename Quat>
	__global__ void SF_setUpFixedJointConstraints(
		DArray<Constraint> constraints,
		DArray<Joint> joints,
		DArray<Matrix> rotMat,
		DArray<Quat> rotQuat,
		int begin_index
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= joints.size())
			return;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;
		Vector<Real, 3> r1 = rotMat[idx1] * joints[tId].r1;
		Vector<Real, 3> r2;
		if (idx2 != INVALID)
		{
			r2 = rotMat[idx2] * joints[tId].r2;
		}

		int baseIndex = 6 * tId + begin_index;
		for (int i = 0; i < 6; i++)
		{
			constraints[baseIndex + i].bodyId1 = idx1;
			constraints[baseIndex + i].bodyId2 = idx2;
			constraints[baseIndex + i].normal1 = r1;
			constraints[baseIndex + i].normal2 = r2;
			constraints[baseIndex + i].pos1 = joints[tId].w;
			constraints[baseIndex + i].rotQuat = joints[tId].q_init;
			constraints[baseIndex + i].isValid = true;
		}

		constraints[baseIndex].type = ConstraintType::CN_ANCHOR_EQUAL_1;
		constraints[baseIndex + 1].type = ConstraintType::CN_ANCHOR_EQUAL_2;
		constraints[baseIndex + 2].type = ConstraintType::CN_ANCHOR_EQUAL_3;
		constraints[baseIndex + 3].type = ConstraintType::CN_BAN_ROT_1;
		constraints[baseIndex + 4].type = ConstraintType::CN_BAN_ROT_2;
		constraints[baseIndex + 5].type = ConstraintType::CN_BAN_ROT_3;
	}

	void setUpFixedJointConstraints(
		DArray<TConstraintPair<float>> &constraints,
		DArray<FixedJoint<float>> &joints,
		DArray<Mat3f> &rotMat,
		DArray<Quat1f> &rotQuat,
		int begin_index
	)
	{
		cuExecute(constraints.size(),
			SF_setUpFixedJointConstraints,
			constraints,
			joints,
			rotMat,
			rotQuat,
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
			constraints[baseIndex + i].isValid = true;
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
			Matrix r1x(0.0f, -r1[2], r1[1], r1[2], 0, -r1[0], -r1[1], r1[0], 0);
			if (idx2 != INVALID)
			{
				Coord r2 = constraints[tId].normal2;
				Matrix r2x(0.0f, -r2[2], r2[1], r2[2], 0, -r2[0], -r2[1], r2[0], 0);
				Matrix K = (1 / mass[idx1]) * E + (r1x * inertia[idx1].inverse()) * r1x.transpose() + (1 / mass[idx2]) * E + (r2x * inertia[idx2].inverse()) * r2x.transpose();
				K_3[tId] = K.inverse();
			}
			else
			{
				Matrix K = (1 / mass[idx1]) * E + (r1x * inertia[idx1].inverse()) * r1x.transpose();
				K_3[tId] = K.inverse();
			}

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
			K_2[tId] = K.inverse();
		}

		else if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_1)
		{
			Matrix K = (1 / mass[idx1]) * Matrix(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
			K_3[tId] = K.inverse();
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
			K_2[tId] = K.inverse();

		}

		else if (constraints[tId].type == ConstraintType::CN_BAN_ROT_1)
		{
			if (idx2 != INVALID)
			{
				Matrix K = inertia[idx1].inverse() + inertia[idx2].inverse();
				K_3[tId] = K.inverse();
			}
			else
			{
				Matrix K = inertia[idx1].inverse();
				K_3[tId] = K.inverse();
			}
		}

		else
		{
			if (constraints[tId].isValid)
			{
				Real K = J[4 * tId].dot(B[4 * tId]) + J[4 * tId + 1].dot(B[4 * tId + 1]) + J[4 * tId + 2].dot(B[4 * tId + 2]) + J[4 * tId + 3].dot(B[4 * tId + 3]);
				K_1[tId] = 1 / K;
			}
		}
	}

	template<typename Coord, typename Constraint, typename Matrix>
	__global__ void SF_calculateKWithCFM(
		DArray<Constraint> constraints,
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Coord> pos,
		DArray<Matrix> inertia,
		DArray<Real> mass,
		DArray<Real> K_1,
		DArray<Mat2f> K_2,
		DArray<Matrix> K_3,
		DArray<float> CFM
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
			Matrix r1x(0.0f, -r1[2], r1[1], r1[2], 0, -r1[0], -r1[1], r1[0], 0);
			Matrix CFM_3(CFM[tId], 0, 0, 0, CFM[tId], 0, 0, 0, CFM[tId]);

			if (idx2 != INVALID)
			{
				Coord r2 = constraints[tId].normal2;
				Matrix r2x(0.0f, -r2[2], r2[1], r2[2], 0, -r2[0], -r2[1], r2[0], 0);
				Matrix K = (1 / mass[idx1]) * E + (r1x * inertia[idx1].inverse()) * r1x.transpose() + (1 / mass[idx2]) * E + (r2x * inertia[idx2].inverse()) * r2x.transpose();
				K_3[tId] = (K + CFM_3).inverse();
			}
			else
			{
				Matrix K = (1 / mass[idx1]) * E + (r1x * inertia[idx1].inverse()) * r1x.transpose();
				K_3[tId] = (K + CFM_3).inverse();
			}

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
			Mat2f CFM_2(CFM[tId], 0, 0, CFM[tId]);
			K_2[tId] = (K + CFM_2).inverse();
		}

		else if (constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_1)
		{
			Matrix K = (1 / mass[idx1]) * Matrix(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
			Matrix CFM_3(CFM[tId], 0, 0, 0, CFM[tId], 0, 0, 0, CFM[tId]);
			K_3[tId] = (K + CFM_3).inverse();
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
			Mat2f CFM_2(CFM[tId], 0, 0, CFM[tId]);
			K_2[tId] = (K + CFM_2).inverse();
		}

		else if (constraints[tId].type == ConstraintType::CN_BAN_ROT_1)
		{
			Matrix CFM_3(CFM[tId], 0, 0, 0, CFM[tId], 0, 0, 0, CFM[tId]);
			if (idx2 != INVALID)
			{
				Matrix K = inertia[idx1].inverse() + inertia[idx2].inverse();
				K_3[tId] = (K + CFM_3).inverse();
			}
			else
			{
				Matrix K = inertia[idx1].inverse();
				K_3[tId] = (K + CFM_3).inverse();
			}
		}

		else
		{
			if (constraints[tId].isValid)
			{
				Real K = J[4 * tId].dot(B[4 * tId]) + J[4 * tId + 1].dot(B[4 * tId + 1]) + J[4 * tId + 2].dot(B[4 * tId + 2]) + J[4 * tId + 3].dot(B[4 * tId + 3]);
				K_1[tId] = 1 / (K + CFM[tId]);
			}
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

	void calculateKWithCFM(
		DArray<TConstraintPair<float>> constraints,
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<Vec3f> pos,
		DArray<Mat3f> inertia,
		DArray<float> mass,
		DArray<float> K_1,
		DArray<Mat2f> K_2,
		DArray<Mat3f> K_3,
		DArray<float> CFM
	)
	{
		cuExecute(constraints.size(),
			SF_calculateKWithCFM,
			constraints,
			J,
			B,
			pos,
			inertia,
			mass,
			K_1,
			K_2,
			K_3,
			CFM);
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
		DArray<Real> fricCoeffs,
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
				Real lambda_new = maximum(0.0f, lambda[tId] + (tmp * K_1[tId] * omega));
				delta_lambda = lambda_new - lambda[tId];
			}
			if (constraints[tId].type == ConstraintType::CN_FRICTION)
			{
				Real mass_avl = mass[idx1];
				Real mu_i = fricCoeffs[idx1];
				if (idx2 != INVALID)
				{
					mass_avl = (mass_avl + mass[idx2]) / 2;
					mu_i = (mu_i + fricCoeffs[idx2]) / 2;
				}
				Real lambda_new = minimum(maximum(lambda[tId] + (tmp * K_1[tId] * omega), -mu_i * mass_avl * g * dt), mu_i * mass_avl * g * dt);
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
			if (idx2 != INVALID)
			{
				for (int i = 0; i < 3; i++)
				{
					tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]) + J[4 * (tId + i) + 2].dot(impulse[idx2 * 2]);
					tmp[i] -= J[4 * (tId + i) + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * (tId + i) + 3].dot(impulse[idx2 * 2 + 1]);
				}
			}
			else
			{
				for (int i = 0; i < 3; i++)
				{
					tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]);
					tmp[i] -= J[4 * (tId + i) + 1].dot(impulse[idx1 * 2 + 1]);
				}
			}

			Coord delta_lambda = omega * (K_3[tId] * tmp);

			for (int i = 0; i < 3; i++)
			{
				atomicAdd(&impulse[idx1 * 2][0], B[4 * (tId + i)][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][1], B[4 * (tId + i)][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][2], B[4 * (tId + i)][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx1 * 2 + 1][0], B[4 * (tId + i) + 1][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][1], B[4 * (tId + i) + 1][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][2], B[4 * (tId + i) + 1][2] * delta_lambda[i]);

				if (idx2 != INVALID)
				{
					atomicAdd(&impulse[idx2 * 2][0], B[4 * (tId + i) + 2][0] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2][1], B[4 * (tId + i) + 2][1] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2][2], B[4 * (tId + i) + 2][2] * delta_lambda[i]);

					atomicAdd(&impulse[idx2 * 2 + 1][0], B[4 * (tId + i) + 3][0] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2 + 1][1], B[4 * (tId + i) + 3][1] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2 + 1][2], B[4 * (tId + i) + 3][2] * delta_lambda[i]);
				}
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

			Vec2f delta_lambda = omega * (K_2[tId] * tmp);
			
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
				Real delta_lambda = tmp * K_1[tId] * omega;
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

			Coord delta_lambda = omega * (K_3[tId] * tmp);
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
	__global__ void SF_JacobiIterationForCFM(
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
		DArray<Real> CFM,
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

			tmp -= lambda[tId] * CFM[tId];

			Real delta_lambda = 0;
			if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
			{
				Real lambda_new = maximum(0.0f, lambda[tId] + (tmp * K_1[tId] * omega));
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
			if (idx2 != INVALID)
			{
				for (int i = 0; i < 3; i++)
				{
					tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]) + J[4 * (tId + i) + 2].dot(impulse[idx2 * 2]);
					tmp[i] -= J[4 * (tId + i) + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * (tId + i) + 3].dot(impulse[idx2 * 2 + 1]);
					tmp[i] -= lambda[tId + i] * CFM[tId + i];
				}
			}
			else
			{
				for (int i = 0; i < 3; i++)
				{
					tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]);
					tmp[i] -= J[4 * (tId + i) + 1].dot(impulse[idx1 * 2 + 1]);
					tmp[i] -= lambda[tId + i] * CFM[tId + i];
				}
			}

			Coord delta_lambda = omega * (K_3[tId] * tmp);

			for (int i = 0; i < 3; i++)
			{
				lambda[tId + i] += delta_lambda[i];
				atomicAdd(&impulse[idx1 * 2][0], B[4 * (tId + i)][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][1], B[4 * (tId + i)][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][2], B[4 * (tId + i)][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx1 * 2 + 1][0], B[4 * (tId + i) + 1][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][1], B[4 * (tId + i) + 1][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][2], B[4 * (tId + i) + 1][2] * delta_lambda[i]);

				if (idx2 != INVALID)
				{
					atomicAdd(&impulse[idx2 * 2][0], B[4 * (tId + i) + 2][0] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2][1], B[4 * (tId + i) + 2][1] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2][2], B[4 * (tId + i) + 2][2] * delta_lambda[i]);

					atomicAdd(&impulse[idx2 * 2 + 1][0], B[4 * (tId + i) + 3][0] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2 + 1][1], B[4 * (tId + i) + 3][1] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2 + 1][2], B[4 * (tId + i) + 3][2] * delta_lambda[i]);
				}
			}
		}

		if (constraints[tId].type == ConstraintType::CN_ALLOW_ROT1D_1 || constraints[tId].type == ConstraintType::CN_ANCHOR_TRANS_1)
		{
			Vec2f tmp(eta[tId], eta[tId + 1]);

			for (int i = 0; i < 2; i++)
			{
				tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]) + J[4 * (tId + i) + 2].dot(impulse[idx2 * 2]);
				tmp[i] -= J[4 * (tId + i) + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * (tId + i) + 3].dot(impulse[idx2 * 2 + 1]);
				tmp[i] -= lambda[tId + i] * CFM[tId + i];
			}

			Vec2f delta_lambda = omega * (K_2[tId] * tmp);

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
			tmp -= lambda[tId] * CFM[tId];
			if (K_1[tId] > 0)
			{
				Real delta_lambda = tmp * (K_1[tId] * omega);
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
				tmp[i] -= lambda[tId + i] * CFM[tId + i];
			}

			Coord delta_lambda = omega * (K_3[tId] * tmp);
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
			if (idx2 != INVALID)
			{
				for (int i = 0; i < 3; i++)
				{
					tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]) + J[4 * (tId + i) + 2].dot(impulse[idx2 * 2]);
					tmp[i] -= J[4 * (tId + i) + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * (tId + i) + 3].dot(impulse[idx2 * 2 + 1]);
				}
			}
			else
			{
				for (int i = 0; i < 3; i++)
				{
					tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]);
					tmp[i] -= J[4 * (tId + i) + 1].dot(impulse[idx1 * 2 + 1]);
				}
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

				if (idx2 != INVALID)
				{
					atomicAdd(&impulse[idx2 * 2][0], B[4 * (tId + i) + 2][0] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2][1], B[4 * (tId + i) + 2][1] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2][2], B[4 * (tId + i) + 2][2] * delta_lambda[i]);

					atomicAdd(&impulse[idx2 * 2 + 1][0], B[4 * (tId + i) + 3][0] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2 + 1][1], B[4 * (tId + i) + 3][1] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2 + 1][2], B[4 * (tId + i) + 3][2] * delta_lambda[i]);
				}
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
		DArray<Real> mu,
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
				Real lambda_new = maximum(0.0f, lambda[tId] + omega * (massCoeff * tmp * K_1[tId] - impulseCoeff * lambda[tId]));
				delta_lambda = lambda_new - lambda[tId];
			}
			if (constraints[tId].type == ConstraintType::CN_FRICTION)
			{
				Real mass_avl = mass[idx1];
				Real mu_i = mu[idx1];
				if (idx2 != INVALID)
				{
					mass_avl = (mass_avl + mass[idx2]) / 2;
					mu_i = (mu_i + mu[idx2]) / 2;
				}
				Real lambda_new = minimum(maximum(lambda[tId] + omega * (massCoeff * tmp * K_1[tId] - impulseCoeff * lambda[tId]), -mu_i * mass_avl * g * dt), mu_i * mass_avl * g * dt);
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
			if (idx2 != INVALID)
			{
				for (int i = 0; i < 3; i++)
				{
					tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]) + J[4 * (tId + i) + 2].dot(impulse[idx2 * 2]);
					tmp[i] -= J[4 * (tId + i) + 1].dot(impulse[idx1 * 2 + 1]) + J[4 * (tId + i) + 3].dot(impulse[idx2 * 2 + 1]);
				}
			}
			else
			{
				for (int i = 0; i < 3; i++)
				{
					tmp[i] -= J[4 * (tId + i)].dot(impulse[idx1 * 2]);
					tmp[i] -= J[4 * (tId + i) + 1].dot(impulse[idx1 * 2 + 1]);
				}
			}
			Coord tmp_lambda(lambda[tId], lambda[tId + 1], lambda[tId + 2]);
			Coord delta_lambda = omega * (massCoeff * K_3[tId] * tmp - impulseCoeff * tmp_lambda);

			for (int i = 0; i < 3; i++)
			{
				atomicAdd(&impulse[idx1 * 2][0], B[4 * (tId + i)][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][1], B[4 * (tId + i)][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2][2], B[4 * (tId + i)][2] * delta_lambda[i]);

				atomicAdd(&impulse[idx1 * 2 + 1][0], B[4 * (tId + i) + 1][0] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][1], B[4 * (tId + i) + 1][1] * delta_lambda[i]);
				atomicAdd(&impulse[idx1 * 2 + 1][2], B[4 * (tId + i) + 1][2] * delta_lambda[i]);

				if (idx2 != INVALID)
				{
					atomicAdd(&impulse[idx2 * 2][0], B[4 * (tId + i) + 2][0] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2][1], B[4 * (tId + i) + 2][1] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2][2], B[4 * (tId + i) + 2][2] * delta_lambda[i]);

					atomicAdd(&impulse[idx2 * 2 + 1][0], B[4 * (tId + i) + 3][0] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2 + 1][1], B[4 * (tId + i) + 3][1] * delta_lambda[i]);
					atomicAdd(&impulse[idx2 * 2 + 1][2], B[4 * (tId + i) + 3][2] * delta_lambda[i]);
				}
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
			Vec2f delta_lambda = omega * (massCoeff * K_2[tId] * tmp - impulseCoeff * oldLambda);


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
				Real delta_lambda = omega * (massCoeff * K_1[tId] * tmp - impulseCoeff * lambda[tId]);
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
			Coord delta_lambda = omega * (massCoeff * K_3[tId] * tmp - impulseCoeff * oldLambda);
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

	template<typename Real, typename Coord, typename Constraint>
	__global__ void SF_JacobiIterationStrict(
		DArray<Real> lambda,
		DArray<Coord> impulse,
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Real> eta,
		DArray<Constraint> constraints,
		DArray<int> nbq,
		DArray<Real> d,
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

		Real tmp = eta[tId];

		tmp -= J[4 * tId].dot(impulse[idx1 * 2]);
		tmp -= J[4 * tId + 1].dot(impulse[idx1 * 2 + 1]);

		if (idx2 != INVALID)
		{
			tmp -= J[4 * tId + 2].dot(impulse[idx2 * 2]);
			tmp -= J[4 * tId + 3].dot(impulse[idx2 * 2 + 1]);
		}

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

		if (d[tId] > EPSILON)
		{
			Real delta_lambda = (tmp / d[tId]) * omega;
			if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
			{
				Real lambda_new = maximum(0.0f, lambda[tId] + (tmp / (d[tId] * stepInverse)));
				delta_lambda = lambda_new - lambda[tId];
			}
			if (constraints[tId].type == ConstraintType::CN_FRICTION)
			{
				Real mass_avl = mass[idx1];
				Real lambda_new = minimum(maximum(lambda[tId] + (tmp / (d[tId] * stepInverse)), -mu * mass_avl * g * dt), mu * mass_avl * g * dt);
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
		DArray<float> fricCoeffs,
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
			fricCoeffs,
			mu,
			g,
			dt);
	}

	void JacobiIterationForCFM(
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
		DArray<float> CFM,
		float mu,
		float g,
		float dt
	)
	{
		cuExecute(constraints.size(),
			SF_JacobiIterationForCFM,
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
			CFM,
			mu,
			g,
			dt);
	}

	void JacobiIterationStrict(
		DArray<float> lambda,
		DArray<Vec3f> impulse,
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<float> eta,
		DArray<TConstraintPair<float>> constraints,
		DArray<int> nbq,
		DArray<float> d,
		DArray<float> mass,
		float mu,
		float g,
		float dt
	)
	{
		cuExecute(constraints.size(),
			SF_JacobiIterationStrict,
			lambda,
			impulse,
			J,
			B,
			eta,
			constraints,
			nbq,
			d,
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
		DArray<float> mu,
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

	template<typename Coord, typename Real>
	__global__ void SF_calculateDiagnals(
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

	void calculateDiagnals(
		DArray<float> d,
		DArray<Vec3f> J,
		DArray<Vec3f> B
	)
	{
		cuExecute(d.size(),
			SF_calculateDiagnals,
			d,
			J,
			B);
	}

	template<typename Coord, typename Real>
	__global__ void SF_preConditionJ(
		DArray<Coord> J,
		DArray<Real> d,
		DArray<Real> eta
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= d.size())
			return;

		if (d[tId] > EPSILON)
		{
			Real d_inv = 1 / d[tId];
			J[4 * tId] = d_inv * J[4 * tId];
			J[4 * tId + 1] = d_inv * J[4 * tId + 1];
			J[4 * tId + 2] = d_inv * J[4 * tId + 2];
			J[4 * tId + 3] = d_inv * J[4 * tId + 3];

			eta[tId] = d_inv * eta[tId];
		}
	}

	void preConditionJ(
		DArray<Vec3f> J,
		DArray<float> d,
		DArray<float> eta
	)
	{
		cuExecute(d.size(),
			SF_preConditionJ,
			J,
			d,
			eta);
	}

	template<typename Coord, typename Real, typename Constraint>
	__global__ void SF_checkOutError(
		DArray<Coord> J,
		DArray<Coord> mImpulse,
		DArray<Constraint> constraints,
		DArray<Real> eta,
		DArray<Real> error
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= constraints.size())
			return;

		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		Real tmp = 0;
		tmp += J[4 * tId].dot(mImpulse[idx1 * 2]) + J[4 * tId + 1].dot(mImpulse[idx1 * 2 + 1]);
		if (idx2 != INVALID)
			tmp += J[4 * tId + 2].dot(mImpulse[idx2 * 2]) + J[4 * tId + 3].dot(mImpulse[idx2 * 2 + 1]);

		Real e = tmp - eta[tId];
		error[tId] = e * e;
	}



	Real checkOutError(
		DArray<Vec3f> J,
		DArray<Vec3f> mImpulse,
		DArray<TConstraintPair<float>> constraints,
		DArray<float> eta
	)
	{
		DArray<float> error;
		error.resize(eta.size());
		error.reset();

		cuExecute(eta.size(),
			SF_checkOutError,
			J,
			mImpulse,
			constraints,
			eta,
			error);

		CArray<float> errorHost;
		errorHost.assign(error);

		Real tmp = 0.0f;
		int num = errorHost.size();
		for (int i = 0; i < num; i++)
		{
			tmp += errorHost[i];
		}
		error.clear();
		errorHost.clear();
		return sqrt(tmp);
	}

	bool saveVectorToFile(
		const std::vector<float>& vec,
		const std::string& filename
	)
	{
		std::ofstream file(filename);
		if (!file.is_open()) {
			std::cerr << "Failed to open file." << std::endl;
			return false; 
		}

		for (float f : vec)
		{
			file << f << " ";
		}

		file.close();
		return true;
	}


	bool saveMatrixToFile(
		DArray<float> &Matrix,
		int n,
		const std::string& filename
	)
	{
		CArray<float> matrix;
		matrix.assign(Matrix);
		std::ofstream file(filename);
		if (!file.is_open()) {
			std::cerr << "Failed to open file." << std::endl;
			return false;
		}

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				file << matrix[i * n + j] << " ";
			}
			file << "\n";
		}

		file.close();
		return true;
	}

	bool saveVectorToFile(
		DArray<float>& vec,
		const std::string& filename
	)
	{
		CArray<float> v;
		v.assign(vec);
		std::ofstream file(filename);
		if (!file.is_open()) {
			std::cerr << "Failed to open file." << std::endl;
			return false;
		}
		printf("%d\n", v.size());
		for (int i = 0; i < v.size(); i++)
		{
			file << v[i] << " ";
		}

		file.close();
		return true;
	}


	double checkOutErrors(
		DArray<float> errors
	)
	{
		CArray<float> merrors;
		merrors.assign(errors);
		double tmp = 0.0;
		for (int i = 0; i < merrors.size(); i++)
		{
			tmp += merrors[i] * merrors[i];
		}

		return sqrt(tmp);
	}

	template<typename Coord, typename Real, typename Constraint>
	__global__ void SF_calculateMatrixA(
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Real> A,
		DArray<Constraint> constraints,
		Real k
	)
	{
		int n = constraints.size();
		int tId = threadIdx.x + blockDim.x * blockIdx.x;
		
		int i = tId / n;
		int j = tId % n;

		if (i >= constraints.size() || j >= constraints.size())
			return;

		if (i > j)
			return;


		int row_idx1 = constraints[i].bodyId1;
		int row_idx2 = constraints[i].bodyId2;

		int col_idx1 = constraints[j].bodyId1;
		int col_idx2 = constraints[j].bodyId2;

		

		Real tmp = 0.0f;

		if (row_idx1 == col_idx1)
			tmp += J[4 * i].dot(B[4 * j]) + J[4 * i + 1].dot(B[4 * j + 1]);

		if (row_idx1 == col_idx2)
			tmp += J[4 * i].dot(B[4 * j + 2]) + J[4 * i + 1].dot(B[4 * j + 3]);

		if (row_idx2 == col_idx1)
			tmp += J[4 * i + 2].dot(B[4 * j]) + J[4 * i + 3].dot(B[4 * j + 1]);

		if (row_idx2 == col_idx2)
			tmp += J[4 * i + 2].dot(B[4 * j + 2]) + J[4 * i + 3].dot(B[4 * j + 3]);
		
		if (i == j && constraints[tId].isValid == true)
			tmp += k;

		A[i * n + j] = tmp;
		A[j * n + i] = tmp;

	}


	void calculateMatrixA(
		DArray<Vec3f> &J,
		DArray<Vec3f> &B,
		DArray<float> &A,
		DArray<TConstraintPair<float>> &constraints,
		float k
	)
	{
		int n = constraints.size();

		cuExecute(n * n,
			SF_calculateMatrixA,
			J,
			B,
			A,
			constraints,
			k);
	}

	template<typename Real, typename Constraint>
	__global__ void SF_vectorSub(
		DArray<Real> ans,
		DArray<Real> subtranhend,
		DArray<Real> minuend,
		DArray<Constraint> constraints
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (!constraints[tId].isValid)
			return;
		if (tId >= ans.size())
			return;
		ans[tId] = minuend[tId] - subtranhend[tId];

		
	}

	void vectorSub(
		DArray<float> &ans,
		DArray<float> &subtranhend,
		DArray<float> &minuend,
		DArray<TConstraintPair<float>> &constraints
	)
	{
		cuExecute(ans.size(),
			SF_vectorSub,
			ans,
			subtranhend,
			minuend,
			constraints);
	}

	
	template<typename Real, typename Constraint>
	__global__ void SF_vectorAdd(
		DArray<Real> ans,
		DArray<Real> v1,
		DArray<Real> v2,
		DArray<Constraint> constraints
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (!constraints[tId].isValid)
			return;
		if (tId >= ans.size())
			return;
		ans[tId] = v1[tId] + v2[tId];
	}

	void vectorAdd(
		DArray<float> &ans,
		DArray<float> &v1,
		DArray<float> &v2,
		DArray<TConstraintPair<float>> &constraints
	)
	{
		cuExecute(ans.size(),
			SF_vectorAdd,
			ans,
			v1,
			v2,
			constraints);
	}
	
	template<typename Real, typename Coord, typename Constraint>
	__global__ void SF_matrixMultiplyVecBuildImpulse(
		DArray<Coord> B,
		DArray<Real> lambda,
		DArray<Coord> impulse,
		DArray<Constraint> constraints
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= lambda.size())
			return;
		if (!constraints[tId].isValid)
			return;

		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		Real lambda_i = lambda[tId];

		atomicAdd(&impulse[2 * idx1][0], B[4 * tId][0] * lambda_i);
		atomicAdd(&impulse[2 * idx1][1], B[4 * tId][1] * lambda_i);
		atomicAdd(&impulse[2 * idx1][2], B[4 * tId][2] * lambda_i);
		atomicAdd(&impulse[2 * idx1 + 1][0], B[4 * tId + 1][0] * lambda_i);
		atomicAdd(&impulse[2 * idx1 + 1][1], B[4 * tId + 1][1] * lambda_i);
		atomicAdd(&impulse[2 * idx1 + 1][2], B[4 * tId + 1][2] * lambda_i);

		if (idx2 != INVALID)
		{
			atomicAdd(&impulse[2 * idx2][0], B[4 * tId + 2][0] * lambda_i);
			atomicAdd(&impulse[2 * idx2][1], B[4 * tId + 2][1] * lambda_i);
			atomicAdd(&impulse[2 * idx2][2], B[4 * tId + 2][2] * lambda_i);
			atomicAdd(&impulse[2 * idx2 + 1][0], B[4 * tId + 3][0] * lambda_i);
			atomicAdd(&impulse[2 * idx2 + 1][1], B[4 * tId + 3][1] * lambda_i);
			atomicAdd(&impulse[2 * idx2 + 1][2], B[4 * tId + 3][2] * lambda_i);
		}
	}

	template<typename Real, typename Coord, typename Constraint>
	__global__ void SF_matrixMultiplyVecUseImpulse(
		DArray<Coord> J,
		DArray<Coord> impulse,
		DArray<Constraint> constraints,
		DArray<Real> ans
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= ans.size())
			return;
		if (!constraints[tId].isValid)
			return;

		Real tmp = 0.0;
		
		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		tmp += J[4 * tId].dot(impulse[idx1 * 2]) + J[4 * tId + 1].dot(impulse[idx1 * 2 + 1]);

		if (idx2 != INVALID)
		{
			tmp += J[4 * tId + 2].dot(impulse[idx2 * 2]) + J[4 * tId + 3].dot(impulse[idx2 * 2 + 1]);
		}

		ans[tId] = tmp;
		
	}


	void matrixMultiplyVec(
		DArray<Vec3f> &J,
		DArray<Vec3f> &B,
		DArray<float> &lambda,
		DArray<float> &ans,
		DArray<TConstraintPair<float>> &constraints,
		int bodyNum
	)
	{
		DArray<Vec3f> impulse;
		impulse.resize(2 * bodyNum);
		impulse.reset();

		cuExecute(constraints.size(),
			SF_matrixMultiplyVecBuildImpulse,
			B,
			lambda,
			impulse,
			constraints);

		cuExecute(constraints.size(),
			SF_matrixMultiplyVecUseImpulse,
			J,
			impulse,
			constraints,
			ans);
		impulse.clear();
	}


	template<typename Real>
	__global__ void SF_vectorInnerProduct(
		DArray<Real> v1,
		DArray<Real> v2,
		DArray<Real> result
	)
	{
		int index = threadIdx.x + blockIdx.x * blockDim.x;

		int N = v1.size();

		if (index >= N)
			return;

		result[index] = v1[index] * v2[index];
	}




	float vectorNorm(
		DArray<float> &a,
		DArray<float> &b
	)
	{
		DArray<float> c;
		c.resize(a.size());
		c.reset();

		cuExecute(a.size(),
			SF_vectorInnerProduct,
			a,
			b,
			c);

		Reduction<float> reduction;
		return reduction.accumulate(c.begin(), c.size());
	}

	template<typename Real, typename Constraint>
	__global__ void SF_vectorMultiplyScale(
		DArray<Real> ans,
		DArray<Real> initialVec,
		DArray<Constraint> constraints,
		Real scale
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (!constraints[tId].isValid)
			return;
		if (tId >= ans.size())
			return;
		ans[tId] = initialVec[tId] * scale;
	}

	void vectorMultiplyScale(
		DArray<float> &ans,
		DArray<float> &initialVec,
		float scale,
		DArray<TConstraintPair<float>>& constraints
	)
	{
		cuExecute(ans.size(),
			SF_vectorMultiplyScale,
			ans,
			initialVec,
			constraints,
			scale);
	}

	template<typename Real, typename Constraint>
	__global__ void SF_vectorClampSupport(
		DArray<Real> v,
		DArray<Constraint> constraints
	)
	{
		int tId = threadIdx.x + blockDim.x * blockIdx.x;
		if (tId >= v.size())
			return;

		if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
		{
			if (v[tId] < 0)
				v[tId] = 0;
		}	
	}

	void vectorClampSupport(
		DArray<float> v,
		DArray<TConstraintPair<float>> constraints
	)
	{
		cuExecute(v.size(),
			SF_vectorClampSupport,
			v,
			constraints);
	}
	template<typename Real, typename Constraint>
	__global__ void SF_vectorClampFriction(
		DArray<Real> v,
		DArray<Constraint> constraints,
		Real mu,
		int contact_size
	)
	{
		int tId = threadIdx.x + blockDim.x * blockIdx.x;
		if (tId >= v.size())
			return;
		if (constraints[tId].type == ConstraintType::CN_FRICTION)
		{
			Real support = abs(v[tId % contact_size]);
			v[tId] = minimum(maximum(v[tId], -mu * support), mu * support);
		}
	}


	void vectorClampFriction(
		DArray<float> v,
		DArray<TConstraintPair<float>> constraints,
		int contact_size,
		float mu
	)
	{
		cuExecute(v.size(),
			SF_vectorClampFriction,
			v,
			constraints,
			mu,
			contact_size);
	}
	

	void calculateImpulseByLambda(
		DArray<float> lambda,
		DArray<TConstraintPair<float>> constraints,
		DArray<Vec3f> impulse,
		DArray<Vec3f> B
	)
	{
		cuExecute(lambda.size(),
			SF_matrixMultiplyVecBuildImpulse,
			B,
			lambda,
			impulse,
			constraints);
	}

	template<typename Real, typename Constraint>
	__global__ void SF_vectorMultiplyVector(
		DArray<Real> v1,
		DArray<Real> v2,
		DArray<Real> ans,
		DArray<Constraint> constraints
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (!constraints[tId].isValid)
			return;

		if (tId >= v1.size())
			return;

		ans[tId] = v1[tId] * v2[tId];
	}

	void vectorMultiplyVector(
		DArray<float>& v1,
		DArray<float>& v2,
		DArray<float>& ans,
		DArray<TConstraintPair<float>>& constraints
	)
	{
		cuExecute(v1.size(),
			SF_vectorMultiplyVector,
			v1,
			v2,
			ans,
			constraints);
	}

	template<typename Real, typename Matrix2x2, typename Matrix3x3, typename Constraint>
	__global__ void SF_preconditionedResidual(
		DArray<Real> residual,
		DArray<Real> ans,
		DArray<Real> k_1,
		DArray<Matrix2x2> k_2,
		DArray<Matrix3x3> k_3,
		DArray<Constraint> constraints
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= residual.size())
			return;

		if (!constraints[tId].isValid)
		{
			return;
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_1 || constraints[tId].type == ConstraintType::CN_BAN_ROT_1 || constraints[tId].type == ConstraintType::CN_JOINT_NO_MOVE_1)
		{
			Vec3f tmp(residual[tId], residual[tId + 1], residual[tId + 2]);
			Vec3f delta = k_3[tId] * tmp;
			for (int i = 0; i < 3; i++)
			{
				ans[tId + i] = delta[i];
			}
		}

		if (constraints[tId].type == ConstraintType::CN_ALLOW_ROT1D_1 || constraints[tId].type == ConstraintType::CN_ANCHOR_TRANS_1)
		{
			Vec2f tmp(residual[tId], residual[tId + 1]);
			Vec2f delta = k_2[tId] * tmp;
			for (int i = 0; i < 2; i++)
			{
				ans[tId + i] = delta[i];
			}
		}

		if (constraints[tId].type == ConstraintType::CN_NONPENETRATION || constraints[tId].type == ConstraintType::CN_FRICTION)
		{
			ans[tId] = residual[tId] * k_1[tId];
		}

		if (constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MIN || constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MAX || constraints[tId].type == ConstraintType::CN_JOINT_HINGE_MOTER || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MIN || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MAX || constraints[tId].type == ConstraintType::CN_JOINT_SLIDER_MOTER)
		{
			if (constraints[tId].isValid)
			{
				ans[tId] = residual[tId] * k_1[tId];
			}
		}
	}

	void preconditionedResidual(
		DArray<float> &residual,
		DArray<float> &ans,
		DArray<float> &k_1,
		DArray<Mat2f> &k_2,
		DArray<Mat3f> &k_3,
		DArray<TConstraintPair<float>> &constraints
	)
	{
		cuExecute(residual.size(),
			SF_preconditionedResidual,
			residual,
			ans,
			k_1,
			k_2,
			k_3,
			constraints);
	}

	template<typename Coord, typename Constraint, typename Real>
	__global__ void SF_buildCFMAndERP(
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Constraint> constraints,
		DArray<Real> CFM,
		DArray<Real> ERP,
		Real hertz,
		Real zeta,
		Real dt
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;

		if (tId >= constraints.size())
			return;

		if (!constraints[tId].isValid)
			return;


		Real d = 0.0;
		int idx2 = constraints[tId].bodyId2;
		if (idx2 != INVALID)
		{
			d += J[4 * tId].dot(B[4 * tId]) + J[4 * tId + 1].dot(B[4 * tId + 1]) + J[4 * tId + 2].dot(B[4 * tId + 2]) + J[4 * tId + 3].dot(B[4 * tId + 3]);
		}
		else
		{
			d += J[4 * tId].dot(B[4 * tId]) + J[4 * tId + 1].dot(B[4 * tId + 1]);
		}
		
		Real m_eff = 1 / d;
		Real omega = 2 * M_PI * hertz;
		Real k = m_eff * omega * omega;
		Real c = 2 * m_eff * zeta * omega;

		CFM[tId] = 1 / (c * dt + dt * dt * k);
		ERP[tId] = dt * k / (c + dt * k);
		
		//printf("%d : CFM(%lf), ERP(%lf)\n", tId, CFM[tId], ERP[tId]);

	}



	void buildCFMAndERP(
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<TConstraintPair<float>> constraints,
		DArray<float> CFM,
		DArray<float> ERP,
		float hertz,
		float zeta,
		float dt
	)
	{
		cuExecute(constraints.size(),
			SF_buildCFMAndERP,
			J,
			B,
			constraints,
			CFM,
			ERP,
			hertz,
			zeta,
			dt);
	}

	template<typename Real, typename Coord, typename Constraint>
	__global__ void SF_calculateLinearSystemLHSImpulse(
		DArray<Coord> B,
		DArray<Coord> impulse,
		DArray<Real> lambda,
		DArray<Constraint> constraints
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;

		if (tId >= constraints.size())
			return;

		if (!constraints[tId].isValid)
			return;


		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		Real lambda_i = lambda[tId];

		atomicAdd(&impulse[2 * idx1][0], B[4 * tId][0] * lambda_i);
		atomicAdd(&impulse[2 * idx1][1], B[4 * tId][1] * lambda_i);
		atomicAdd(&impulse[2 * idx1][2], B[4 * tId][2] * lambda_i);
		atomicAdd(&impulse[2 * idx1 + 1][0], B[4 * tId + 1][0] * lambda_i);
		atomicAdd(&impulse[2 * idx1 + 1][1], B[4 * tId + 1][1] * lambda_i);
		atomicAdd(&impulse[2 * idx1 + 1][2], B[4 * tId + 1][2] * lambda_i);

		if (idx2 != INVALID)
		{
			atomicAdd(&impulse[2 * idx2][0], B[4 * tId + 2][0] * lambda_i);
			atomicAdd(&impulse[2 * idx2][1], B[4 * tId + 2][1] * lambda_i);
			atomicAdd(&impulse[2 * idx2][2], B[4 * tId + 2][2] * lambda_i);
			atomicAdd(&impulse[2 * idx2 + 1][0], B[4 * tId + 3][0] * lambda_i);
			atomicAdd(&impulse[2 * idx2 + 1][1], B[4 * tId + 3][1] * lambda_i);
			atomicAdd(&impulse[2 * idx2 + 1][2], B[4 * tId + 3][2] * lambda_i);
		}
	}

	template<typename Real, typename Coord, typename Constraint>
	__global__ void SF_calculateLinearSystemLHSResult(
		DArray<Coord> impulse,
		DArray<Coord> J,
		DArray<Real> CFM,
		DArray<Real> ans,
		DArray<Real> lambda,
		DArray<Constraint> constraints
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;

		if (tId >= constraints.size())
			return;

		if (!constraints[tId].isValid)
			return;

		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		Real tmp = 0.0;

		tmp += J[4 * tId].dot(impulse[2 * idx1]) + J[4 * tId + 1].dot(impulse[2 * idx1 + 1]);

		if (idx2 != INVALID)
			tmp += J[4 * tId + 2].dot(impulse[2 * idx2]) + J[4 * tId + 3].dot(impulse[2 * idx2 + 1]);


		tmp += CFM[tId] * lambda[tId];


		ans[tId] = tmp;
	}



	void calculateLinearSystemLHS(
		DArray<Vec3f>& J,
		DArray<Vec3f>& B,
		DArray<Vec3f>& impulse,
		DArray<float>& lambda,
		DArray<float>& ans,
		DArray<float>& CFM,
		DArray<TConstraintPair<float>>& constraints
	)
	{
		int n = constraints.size();

		cuExecute(n,
			SF_calculateLinearSystemLHSImpulse,
			B,
			impulse,
			lambda,
			constraints);

		cuExecute(n,
			SF_calculateLinearSystemLHSResult,
			impulse,
			J,
			CFM,
			ans,
			lambda,
			constraints);
	}


	std::vector<int> DynamicGraphColoring:: orderedGreedyColoring(const std::vector<std::vector<int>>& graph) {
		const int num_vertices = graph.size();
		if(num_vertices == 0) {
			return {};
		}

		std::vector<int> colors(num_vertices, -1); // -1 means uncolored
		std::vector<int> degree(num_vertices);
		std::vector<std::list<int>::iterator> vertex_iterators(num_vertices);
		std::vector<std::list<int>> color_buckets(num_vertices);

		int max_degree = 0;

		for (int i = 0; i < num_vertices; ++i) {
			degree[i] = graph[i].size();
			color_buckets[degree[i]].push_front(i);
			vertex_iterators[i] = color_buckets[degree[i]].begin();
			if(degree[i] > max_degree) {
				max_degree = degree[i];
			}
		}

		std::vector<int> processing_order;
		processing_order.reserve(num_vertices);
		int min_degree_idx = 0;

		for (int i = 0; i < num_vertices; ++i) {
			while(min_degree_idx <= max_degree && color_buckets[min_degree_idx].empty()) {
				++min_degree_idx;
			}

			if(min_degree_idx > max_degree) {
				break; // No more vertices to process
			}

			int current_vertex = color_buckets[min_degree_idx].front();
			color_buckets[min_degree_idx].pop_front();
			processing_order.push_back(current_vertex);
			degree[current_vertex] = -1; // Mark as processed

			for (int neighbor : graph[current_vertex]) {
				if(degree[neighbor] != -1) {
					int old_degree = degree[neighbor];
					color_buckets[old_degree].erase(vertex_iterators[neighbor]);
					degree[neighbor]--;
					int new_degree = degree[neighbor];
					color_buckets[new_degree].push_front(neighbor);
					vertex_iterators[neighbor] = color_buckets[new_degree].begin();
					if(new_degree < min_degree_idx) {
						min_degree_idx = new_degree;
					}
				}
			}
		}

		int max_color_used = 0;
		std::reverse(processing_order.begin(), processing_order.end());
		for (int node_to_color : processing_order) {
			std::vector<bool> used_colors(max_color_used + 2, false);
			for (int neighbor : graph[node_to_color]) {
				if (colors[neighbor] != -1) used_colors[colors[neighbor]] = true;
			}
			int node_color = 0;
			while (node_color < used_colors.size() && used_colors[node_color]) node_color++;
			colors[node_to_color] = node_color;
			max_color_used = std::max(max_color_used, node_color);
		}

		return colors;
	}


	void DynamicGraphColoring::balanceColoring(const std::vector<std::vector<int>>& graph, std::vector<int>& colors, float goal_ratio, int maxAttempts) {
		if(colors.empty()) {
			return; // No colors to balance
		}
		int num_colors = *std::max_element(colors.begin(), colors.end()) + 1;
		if(num_colors <= 1) {
			return; // No need to balance if there's only one color
		}

		std::vector<std::vector<int>> categories(num_colors);

		for (int i = 0; i < colors.size(); ++i) {
			if(colors[i] >= 0)
				categories[colors[i]].push_back(i);
		}

		std::mt19937 gen(std::random_device{}());
		for (int attempt = 0; attempt < maxAttempts; ++attempt) {
			int biggest_idx = -1, smallest_idx = -1;
			size_t max_size = 0, min_size = colors.size() + 1;
			for (int i = 0; i < num_colors; ++i) {
				if (categories[i].size() > max_size) {
					max_size = categories[i].size();
					biggest_idx = i;
				}
				if(!categories[i].empty() && categories[i].size() < min_size) {
					min_size = categories[i].size();
					smallest_idx = i;
				}
			}

			if(biggest_idx == -1 || smallest_idx == -1 || biggest_idx == smallest_idx) {
				break; // No more balancing needed
			}

			if(min_size > 0 && (static_cast<float>(max_size) / min_size) <= goal_ratio) {
				break; // Already balanced
			}

			auto& biggestCatNodes = categories[biggest_idx];
			if (biggestCatNodes.empty()) continue;

			std::uniform_int_distribution<> dis(0, biggestCatNodes.size() - 1);
			int node_index_in_category = dis(gen);
			int node_to_move = biggestCatNodes[node_index_in_category];

			bool is_changeable = true;
			for (int n : graph[node_to_move]) {
				if (colors[n] == smallest_idx) {
					is_changeable = false;
					break;
				}
			}

			if (is_changeable) {
				colors[node_to_move] = smallest_idx;
				categories[smallest_idx].push_back(node_to_move);
				std::swap(biggestCatNodes[node_index_in_category], biggestCatNodes.back());
				biggestCatNodes.pop_back();
			}
		}
	}



	DynamicGraphColoring::DynamicGraphColoring() : num_vertices(0), num_colors(0) {}

	void DynamicGraphColoring::initializeGraph(int num_v, const std::vector<std::pair<int, int>>& initial_edges) {
		this->num_vertices = num_v;
		graph.assign(num_vertices, std::vector<int>());

		for (const auto& edge : initial_edges) {
			if (edge.first < num_vertices && edge.second < num_vertices && edge.first >= 0 && edge.second >= 0) {
				graph[edge.first].push_back(edge.second);
				graph[edge.second].push_back(edge.first);
			}
		}
		this->graph_initialized_ = true;
	}

	void DynamicGraphColoring::performInitialColoring() {
		if (num_vertices == 0) return;

		// Perform initial coloring
		colors = orderedGreedyColoring(graph);
		this->num_colors = colors.empty() ? 0 : *std::max_element(colors.begin(), colors.end()) + 1;

		// Balance the coloring
		balanceColoring(graph, colors);
		this->num_colors = colors.empty() ? 0 : *std::max_element(colors.begin(), colors.end()) + 1;

		// Rebuild internal data structures based on the new coloring
		rebuildCategories();
	}

	void DynamicGraphColoring::applyBatchUpdate(const std::vector<std::pair<int, int>>& add_edges, const std::vector<std::pair<int, int>>& delete_edges) {
		// Step 1: Update graph structure
		for (const auto& edge : add_edges) {
			int u = edge.first;
			int v = edge.second;

			if (u >= num_vertices || v >= num_vertices || u < 0 || v < 0 || u == v) continue;

			if (std::find(graph[u].begin(), graph[u].end(), v) == graph[u].end()) {
				graph[u].push_back(v);
				graph[v].push_back(u);
			}
		}

		for (const auto& edge : delete_edges) {
			int u = edge.first;
			int v = edge.second;

			if (u >= num_vertices || v >= num_vertices || u < 0 || v < 0 || u == v) continue;

			graph[u].erase(std::remove(graph[u].begin(), graph[u].end(), v), graph[u].end());
			graph[v].erase(std::remove(graph[v].begin(), graph[v].end(), u), graph[v].end());
		}

		// Step 2: Conflict Detection
		std::set<int> conflict_vertices;
		for (const auto& edge : add_edges) {
			if (colors[edge.first] == colors[edge.second]) {
				// Add the vertex with the smaller degree to the conflict set for potential recoloring
				if (graph[edge.first].size() < graph[edge.second].size()) {
					conflict_vertices.insert(edge.first);
				}
				else {
					conflict_vertices.insert(edge.second);
				}
			}
		}

		// Step 3: Conflict Resolution
		for (int vertex : conflict_vertices) {
			if (isConflictNode(vertex)) {
				int old_color = colors[vertex];
				std::optional<int> new_color_opt = findNewColorFor(vertex);
				if (new_color_opt.has_value()) {
					updateNodeColor(vertex, old_color, new_color_opt.value());
				}
			}
		}

		// Optionally, re-balance the coloring after updates
		balanceColoring(graph, colors);
		this->num_colors = colors.empty() ? 0 : *std::max_element(colors.begin(), colors.end()) + 1;
		rebuildCategories(); // Rebuild categories after potential color changes
	}

	void DynamicGraphColoring::rebuildCategories() {
		if (num_colors == 0) {
			categories.clear();
			return;
		}
		categories.assign(num_colors, std::vector<int>());
		for (int i = 0; i < num_vertices; ++i) {
			if (colors[i] >= 0) {
				if (colors[i] >= categories.size()) {
					categories.resize(colors[i] + 1);
				}
				categories[colors[i]].push_back(i);
			}
		}
	}

	std::optional<int> DynamicGraphColoring::findNewColorFor(int node) {
		std::vector<bool> used_colors(num_colors, false);
		for (int neighbor : graph[node]) {
			if (colors[neighbor] != -1) {
				used_colors[colors[neighbor]] = true;
			}
		}
		for (int c = 0; c < num_colors; ++c) {
			if (!used_colors[c]) {
				return c;
			}
		}
		// If no existing color is available, create a new one.
		num_colors++;
		return num_colors - 1;
	}

	bool DynamicGraphColoring::isConflictNode(int node) {
		for (int neighbor : graph[node]) {
			if (colors[node] == colors[neighbor]) {
				return true;
			}
		}
		return false;
	}

	void DynamicGraphColoring::updateNodeColor(int node, int old_color, int new_color) {
		colors[node] = new_color;

		if (new_color >= categories.size()) {
			categories.resize(new_color + 1);
		}

		// Remove node from its old color category
		if (old_color != -1 && old_color < categories.size()) {
			auto& old_cat_nodes = categories[old_color];
			old_cat_nodes.erase(std::remove(old_cat_nodes.begin(), old_cat_nodes.end(), node), old_cat_nodes.end());
		}

		// Add node to its new color category
		categories[new_color].push_back(node);
	}


	void constraintsMappingToEdges(DArray<TConstraintPair<float>> constraints, std::vector<std::pair<int, int>>& edges) {
		CArray<TConstraintPair<float>> c_constraints;
		c_constraints.assign(constraints);

		std::set<std::pair<int, int>> unique_edges;

		for (int i = 0; i < c_constraints.size(); ++i) {
			auto constraint = c_constraints[i];
			if (constraint.bodyId2 == INVALID || constraint.bodyId1 == constraint.bodyId2) {
				continue;
			}

			int u = std::min(constraint.bodyId1, constraint.bodyId2);
			int v = std::max(constraint.bodyId1, constraint.bodyId2);

			unique_edges.insert({ u, v });
		}
		edges.assign(unique_edges.begin(), unique_edges.end());

		c_constraints.clear();
	}

}