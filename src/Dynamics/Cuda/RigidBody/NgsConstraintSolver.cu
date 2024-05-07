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
		DArray<Coord> impulse,
		Real dt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= velocity.size())
			return;

		velocity[tId] += impulse[2 * tId] * dt;
		angular_velocity[tId] += impulse[2 * tId + 1] * dt;
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

	template<typename Coord, typename Matrix, typename Quat>
	__global__ void CorrectPosition(
		DArray<Coord> impulse,
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

		pos[tId] += impulse[2 * tId] * dt * dt;
		rotQuat[tId] += dt * dt * 0.5 * Quat(impulse[2 * tId + 1][0], impulse[2 * tId + 1][1], impulse[tId][2], 0.0) * (rotQuat[tId]);

		rotQuat[tId] = rotQuat[tId].normalize();

		rotMat[tId] = rotQuat[tId].toMatrix3x3();

		inertia[tId] = rotMat[tId] * inertia_init[tId] * rotMat[tId].inverse();
	}

	template<typename Coord, typename Matrix, typename Constraint>
	__global__ void NGScalculateJacobianAndB(
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
			Coord r1 = constraints[tId].pos1;
			Coord rcn_1 = r1.cross(n);

			J[4 * tId] = -n;
			J[4 * tId + 1] = -rcn_1;
			B[4 * tId] = -n / mass[idx1];
			B[4 * tId + 1] = -inertia[idx1].inverse() * rcn_1;

			if (idx2 != INVALID)
			{
				Coord r2 = constraints[tId].pos2;
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
	__global__ void NGScalculateDiagonals(
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
	__global__ void NGScalculateEta(
		DArray<Real> eta,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Constraint> constraints,
		DArray<Coord> J,
		Real dt
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= constraints.size())
			return;


		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		Real eta_i = Real(0);
		Real invDt = 1.0 / dt;

		eta_i -= J[4 * tId].dot(velocity[idx1] * invDt);
		eta_i -= J[4 * tId + 1].dot(angular_velocity[idx1] * invDt);

		if (idx2 != INVALID)
		{
			eta_i -= J[4 * tId + 2].dot(velocity[idx2] * invDt);
			eta_i -= J[4 * tId + 3].dot(angular_velocity[idx2] * invDt);
		}

		eta[tId] = eta_i;

		if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
		{
			Real beta = Real(0.2);
			Real alpha = 0;

			Real b_error = beta * invDt * constraints[tId].interpenetration;

			printf("%lf\n", b_error);

			Real b_res = 0;

			Coord n = constraints[tId].normal1;
			Coord r1 = constraints[tId].pos1;

			Coord gamma = -velocity[idx1] - angular_velocity[idx1].cross(r1);

			if (idx2 != INVALID)
			{
				Coord r2 = constraints[tId].pos2;
				gamma = gamma + velocity[idx2] + angular_velocity[idx2].cross(r2);
			}

			b_res += alpha * gamma.dot(n);

			eta[tId] -= (b_error + b_res) * invDt;
		}
	}


	template<typename Coord, typename Constraint, typename Real>
	__global__ void NGScalculateError(
		DArray<Real> positionError,
		DArray<Coord> pos,
		DArray<Constraint> constraints,
		Real dt
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= constraints.size())
			return;

		Real beta = 1.0;
		Real invDt = 1 / dt;
		int idx1 = constraints[tId].bodyId1;
		int idx2 = constraints[tId].bodyId2;

		if (constraints[tId].type == ConstraintType::CN_NONPENETRATION)
		{
			Real error = 0;
			error -= (pos[idx1] + constraints[tId].pos1).dot(constraints[tId].normal1);
			if (idx2 != INVALID)
			{
				error += (pos[idx2] + constraints[tId].pos2).dot(constraints[tId].normal1);
			}
			positionError[tId] = -beta * error * invDt * invDt;

			//printf("%d : %lf\n", tId, error);
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_1)
		{
			Coord error = pos[idx2] + constraints[tId].normal2 - pos[idx1] - constraints[tId].normal1;
			positionError[tId] = -beta * error[0] * invDt * invDt;
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_2)
		{
			Coord error = pos[idx2] + constraints[tId].normal2 - pos[idx1] - constraints[tId].normal1;
			positionError[tId] = -beta * error[1] * invDt * invDt;
		}

		if (constraints[tId].type == ConstraintType::CN_ANCHOR_EQUAL_3)
		{
			Coord error = pos[idx2] + constraints[tId].normal2 - pos[idx1] - constraints[tId].normal1;
			positionError[tId] = -beta * error[2] * invDt * invDt;
		}


	}

	template<typename Contact, typename Constraint, typename Coord, typename Matrix>
	__global__ void NGSsetUpContactAndFrictionConstraints(
		DArray<Constraint> constraints,
		DArray<Contact> contacts,
		DArray<Coord> pos,
		DArray<Coord> localPoint,
		DArray<Coord> localNormal,
		DArray<Matrix> rotMat,
		bool hasFriction
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contacts.size())
			return;


		int idx1 = contacts[tId].bodyId1;
		int idx2 = contacts[tId].bodyId2;
		int contact_size = contacts.size();
		constraints[tId].bodyId1 = contacts[tId].bodyId1;
		constraints[tId].bodyId2 = contacts[tId].bodyId2;

		if (idx2 != INVALID)
		{
			constraints[tId].pos1 = contacts[tId].pos2 - contacts[tId].interpenetration * contacts[tId].normal1 - pos[idx1];
			constraints[tId].pos2 = contacts[tId].pos2 - pos[idx2];
		}
		else
		{
			constraints[tId].pos1 = contacts[tId].pos1 - pos[idx1];
		}
		localPoint[2 * tId] = constraints[tId].pos1;
		localPoint[2 * tId + 1] = constraints[tId].pos2;


		
		constraints[tId].normal1 = -contacts[tId].normal1;
		constraints[tId].normal2 = -contacts[tId].normal2;
		constraints[tId].interpenetration = -contacts[tId].interpenetration;
		constraints[tId].type = ConstraintType::CN_NONPENETRATION;
		localNormal[tId] = rotMat[idx1].inverse() * constraints[tId].normal1;

		
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
			constraints[tId * 2 + contact_size].pos1 = constraints[tId].pos1;
			constraints[tId * 2 + contact_size].pos2 = constraints[tId].pos2;
			constraints[tId * 2 + contact_size].normal1 = u1;
			constraints[tId * 2 + contact_size].normal2 = -u1;
			constraints[tId * 2 + contact_size].type = ConstraintType::CN_FRICTION;

			constraints[tId * 2 + 1 + contact_size].bodyId1 = contacts[tId].bodyId1;
			constraints[tId * 2 + 1 + contact_size].bodyId2 = contacts[tId].bodyId2;
			constraints[tId * 2 + 1 + contact_size].pos1 = constraints[tId].pos1;
			constraints[tId * 2 + 1 + contact_size].pos2 = constraints[tId].pos2;
			constraints[tId * 2 + 1 + contact_size].normal1 = u2;
			constraints[tId * 2 + 1 + contact_size].normal2 = -u2;
			constraints[tId * 2 + 1 + contact_size].type = ConstraintType::CN_FRICTION;
		}
	}

	template<typename Contact, typename Constraint, typename Coord, typename Matrix>
	__global__ void NGSsetUpContactConstraints(
		DArray<Constraint> constraints,
		DArray<Contact> contacts,
		DArray<Coord> localPoint,
		DArray<Coord> localNormal,
		DArray<Matrix> rotMat
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contacts.size())
			return;


		int idx1 = contacts[tId].bodyId1;
		int idx2 = contacts[tId].bodyId2;
		constraints[tId].bodyId1 = contacts[tId].bodyId1;
		constraints[tId].bodyId2 = contacts[tId].bodyId2;
		constraints[tId].pos1 = localPoint[2 * tId];
		constraints[tId].pos2 = localPoint[2 * tId + 1];
		constraints[tId].normal1 = rotMat[idx1] * localNormal[tId];
		constraints[tId].normal2 = -rotMat[idx1] *localNormal[tId];
		constraints[tId].interpenetration = -contacts[tId].interpenetration;
		constraints[tId].type = ConstraintType::CN_NONPENETRATION;
		
	}

	

	template<typename Joint, typename Constraint, typename Coord, typename Matrix>
	__global__ void NGSsetUpBallAndSocketJointConstraints(
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
	__global__ void NGStakeOneJacobiIteration(
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
		Real dt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
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

		if (d[tId] > EPSILON)
		{
			int stepInverse = 5;
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

				lambda_new = (abs(lambda_new) > mu * mass_avl * g) ? (lambda_new < 0 ? -mu * mass_avl * g : mu * mass_avl * g) : lambda_new;
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
	void NgsConstraintSolver<TDataType>::initializeJacobian(Real dt)
	{
		int constraint_size = 0;
		int contact_size = this->inContacts()->size();
		mLocalPoint.resize(2 * contact_size);
		mLocalPoint.reset();
		mLocalNormal.resize(contact_size);
		mLocalNormal.reset();

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
			cuExecute(contact_size,
				NGSsetUpContactAndFrictionConstraints,
				mAllConstraints,
				contacts,
				this->inCenter()->getData(),
				mLocalPoint,
				mLocalNormal,
				this->inRotationMatrix()->getData(),
				this->varFrictionEnabled()->getData());
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
				NGSsetUpBallAndSocketJointConstraints,
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
			NGScalculateJacobianAndB,
			mJ,
			mB,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			mAllConstraints,
			this->inRotationMatrix()->getData());

		cuExecute(constraint_size,
			NGScalculateDiagonals,
			mD,
			mJ,
			mB);

		cuExecute(constraint_size,
			NGScalculateEta,
			mEta,
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			mAllConstraints,
			mJ,
			dt);
	}

	template<typename TDataType>
	void NgsConstraintSolver<TDataType>::initializeJacobianForPosition(Real dt)
	{
		int constraint_size = 0;
		//int contact_size = this->inContacts()->size();


		int ballAndSocketJoint_size = this->inBallAndSocketJoints()->size();

		constraint_size += 3 * ballAndSocketJoint_size;
	

		std::cout << constraint_size << std::endl;
		if (constraint_size == 0)
		{
			return;
		}

		mAllConstraints.resize(constraint_size);

		/*if (contact_size != 0)
		{
			auto& contacts = this->inContacts()->getData();
			cuExecute(contact_size,
				NGSsetUpContactConstraints,
				mAllConstraints,
				contacts,
				mLocalPoint,
				mLocalNormal,
				this->inRotationMatrix()->getData());
		}*/

		if (ballAndSocketJoint_size != 0)
		{
			auto& joints = this->inBallAndSocketJoints()->getData();
			int begin_index = 0;
			cuExecute(ballAndSocketJoint_size,
				NGSsetUpBallAndSocketJointConstraints,
				mAllConstraints,
				joints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				begin_index);
		}

		mJ.resize(4 * constraint_size);
		mB.resize(4 * constraint_size);
		mD.resize(constraint_size);
		mPositionError.resize(constraint_size);
		mLambda.resize(constraint_size);

		mJ.reset();
		mB.reset();
		mD.reset();
		mPositionError.reset();
		mLambda.reset();

		cuExecute(constraint_size,
			NGScalculateJacobianAndB,
			mJ,
			mB,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			mAllConstraints,
			this->inRotationMatrix()->getData());

		cuExecute(constraint_size,
			NGScalculateDiagonals,
			mD,
			mJ,
			mB);

		cuExecute(constraint_size,
			NGScalculateError,
			mPositionError,
			this->inCenter()->getData(),
			mAllConstraints,
			dt);
	}


	template<typename Coord>
	__global__ void NGSsetUpGravity(
		DArray<Coord> impulse_ext,
		Real g
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= impulse_ext.size() / 2)
			return;

		impulse_ext[2 * tId] = Coord(0, -g, 0);
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
				NGSsetUpGravity,
				mImpulseExt,
				this->varGravityValue()->getData());
		}

		cuExecute(bodyNum,
			IntegrationVelocity,
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			mImpulseExt,
			dt);

		if (!this->inContacts()->isEmpty() || !this->inBallAndSocketJoints()->isEmpty())
		{
			initializeJacobian(dt);


			int constraint_size = mAllConstraints.size();

			for (int i = 0; i < this->varVelocityIterationNumber()->getData(); i++)
			{
				cuExecute(constraint_size,
					NGStakeOneJacobiIteration,
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
					dt);
			}
		}

		cuExecute(bodyNum,
			IntegrationVelocity,
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			mImpulseC,
			dt);

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

		/*if (!this->inBallAndSocketJoints()->isEmpty())
		{
			for (int i = 0; i < this->varPositionIterationNumber()->getData(); i++)
			{
				initializeJacobianForPosition(dt);

				mImpulseC.reset();
				int constraint_size = mAllConstraints.size();

				cuExecute(constraint_size,
					NGStakeOneJacobiIteration,
					mLambda,
					mImpulseC,
					mD,
					mJ,
					mB,
					mPositionError,
					this->inMass()->getData(),
					mAllConstraints,
					this->varFrictionCoefficient()->getData(),
					this->varGravityValue()->getData(),
					dt);
				cuExecute(bodyNum,
					CorrectPosition,
					mImpulseC,
					this->inCenter()->getData(),
					this->inQuaternion()->getData(),
					this->inRotationMatrix()->getData(),
					this->inInertia()->getData(),
					this->inVelocity()->getData(),
					this->inAngularVelocity()->getData(),
					this->inInitialInertia()->getData(),
					dt);
			}
			
		}*/

	}
	DEFINE_CLASS(NgsConstraintSolver);
}