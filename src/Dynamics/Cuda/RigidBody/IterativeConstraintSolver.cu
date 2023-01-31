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

	template <typename Coord, typename Constraint>
	__global__ void TakeOneJacobiIteration(
		DArray<Real> lambda,
		DArray<Coord> accel,
		DArray<Real> d,
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Real> eta,
		DArray<Real> mass,
		DArray<Constraint> nbq,
		DArray<Real> stepInv)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;

		int idx1 = nbq[pId].bodyId1;
		int idx2 = nbq[pId].bodyId2;

		Real eta_i = eta[pId];
		{
			eta_i -= J[4 * pId].dot(accel[idx1 * 2]);
			eta_i -= J[4 * pId + 1].dot(accel[idx1 * 2 + 1]);
			if (idx2 != INVLIDA_ID)
			{
				eta_i -= J[4 * pId + 2].dot(accel[idx2 * 2]);
				eta_i -= J[4 * pId + 3].dot(accel[idx2 * 2 + 1]);
			}
		}

		if (d[pId] > EPSILON)
		{
			Real delta_lambda = eta_i / d[pId];
			Real stepInverse = stepInv[idx1];
			if (idx2 != INVLIDA_ID)
				stepInverse += stepInv[idx2];
			delta_lambda *= (1.0f / stepInverse);

			//printf("delta_lambda = %.3lf\n", delta_lambda);

			Real lambda_new = lambda[pId] + delta_lambda;

			if (nbq[pId].type == ConstraintType::CN_NONPENETRATION) //	PROJECTION!!!!
			{
				if (lambda_new < 0) lambda_new = 0;

				delta_lambda = lambda_new - lambda[pId];
			}

			if (nbq[pId].type == ConstraintType::CN_FRICTION) //	PROJECTION!!!!
			{
				delta_lambda = lambda_new - lambda[pId];
			}

			lambda[pId] += delta_lambda;

			//printf("inside iteration: %d %d %.5lf   %.5lf\n", idx1, idx2, nbq[pId].s4, delta_lambda);

			atomicAdd(&accel[idx1 * 2][0], B[4 * pId][0] * delta_lambda);
			atomicAdd(&accel[idx1 * 2][1], B[4 * pId][1] * delta_lambda);
			atomicAdd(&accel[idx1 * 2][2], B[4 * pId][2] * delta_lambda);

			atomicAdd(&accel[idx1 * 2 + 1][0], B[4 * pId + 1][0] * delta_lambda);
			atomicAdd(&accel[idx1 * 2 + 1][1], B[4 * pId + 1][1] * delta_lambda);
			atomicAdd(&accel[idx1 * 2 + 1][2], B[4 * pId + 1][2] * delta_lambda);

			if (idx2 != INVLIDA_ID)
			{
				atomicAdd(&accel[idx2 * 2][0], B[4 * pId + 2][0] * delta_lambda);
				atomicAdd(&accel[idx2 * 2][1], B[4 * pId + 2][1] * delta_lambda);
				atomicAdd(&accel[idx2 * 2][2], B[4 * pId + 2][2] * delta_lambda);

				atomicAdd(&accel[idx2 * 2 + 1][0], B[4 * pId + 3][0] * delta_lambda);
				atomicAdd(&accel[idx2 * 2 + 1][1], B[4 * pId + 3][1] * delta_lambda);
				atomicAdd(&accel[idx2 * 2 + 1][2], B[4 * pId + 3][2] * delta_lambda);
			}
		}
	}

	template <typename Coord>
	__global__ void RB_UpdateVelocity(
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> accel,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= accel.size() / 2) return;

		velocity[pId] += accel[2 * pId] * dt;
		velocity[pId] += Coord(0, -9.8f, 0) * dt;

		angular_velocity[pId] += accel[2 * pId + 1] * dt;
	}

	template <typename Coord, typename Matrix, typename Quat>
	__global__ void RB_UpdateGesture(
		DArray<Coord> pos,
		DArray<Quat> rotQuat,
		DArray<Matrix> rotMat,
		DArray<Matrix> inertia,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Matrix> inertia_init,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		pos[pId] += velocity[pId] * dt;

		rotQuat[pId] = rotQuat[pId].normalize();
		rotMat[pId] = rotQuat[pId].toMatrix3x3();

		rotQuat[pId] += dt * 0.5f *
			Quat(angular_velocity[pId][0], angular_velocity[pId][1], angular_velocity[pId][2], 0.0)
			*(rotQuat[pId]);

		inertia[pId] = rotMat[pId] * inertia_init[pId] * rotMat[pId].inverse();
		//inertia[pId] = rotMat[pId] * rotMat[pId].inverse();
	}

	template<typename TDataType>
	void IterativeConstraintSolver<TDataType>::constrain()
	{
		uint num = this->inCenter()->size();
		if (mAccel.size() != num * 2)
		{
			mAccel.resize(num * 2);
		}

		mAccel.reset();

		Real dt = this->inTimeStep()->getData();
		//construct j

		if (!this->inContacts()->isEmpty())
		{
			initializeJacobian(dt);

			int size_constraints = mAllConstraints.size();
			for (int i = 0; i < this->varIterationNumber()->getData(); i++)
			{
				cuExecute(size_constraints,
					TakeOneJacobiIteration,
					mLambda,
					mAccel,
					mD,
					mJ,
					mB,
					mEta,
					this->inMass()->getData(),
					mAllConstraints,
					mContactNumber);
			}
		}

		cuExecute(num,
			RB_UpdateVelocity,
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			mAccel,
			dt);

// 		cuExecute(num,
// 			RB_UpdateGesture,
// 			this->inCenter()->getData(),
// 			this->inQuaternion()->getData(),
// 			this->inRotationMatrix()->getData(),
// 			this->inInertia()->getData(),
// 			this->inVelocity()->getData(),
// 			this->inAngularVelocity()->getData(),
// 			mInitialInertia,
// 			dt);

		cuExecute(num,
			RB_UpdateGesture,
			this->inCenter()->getData(),
			this->inQuaternion()->getData(),
			this->inRotationMatrix()->getData(),
			this->inInertia()->getData(),
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			this->inInitialInertia()->getData(),
			dt);
	}

	template <typename ContactPair>
	__global__ void CalculateNbrCons(
		DArray<ContactPair> nbc,
		DArray<Real> nbrCnt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nbc.size()) return;

		int idx1 = nbc[pId].bodyId1;
		int idx2 = nbc[pId].bodyId2;

		if (idx1 != INVLIDA_ID)
			atomicAdd(&nbrCnt[idx1], 1.0f);
		if (idx2 != INVLIDA_ID)
			atomicAdd(&nbrCnt[idx2], 1.0f);
	}

	template <typename Coord, typename Matrix, typename Constraint>
	__global__ void CalculateJacobians(
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Coord> pos,
		DArray<Matrix> inertia,
		DArray<Real> mass,
		DArray<Constraint> constraints)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;

		int idx1 = constraints[pId].bodyId1;
		int idx2 = constraints[pId].bodyId2;

		Coord n = constraints[pId].normal1;
		Coord r1 = constraints[pId].pos1 - pos[idx1];

		J[4 * pId] = n;
		J[4 * pId + 1] = (r1.cross(n));
		B[4 * pId] = n / mass[idx1];
		B[4 * pId + 1] = inertia[idx1].inverse() * (r1.cross(n));

		if (idx2 != INVLIDA_ID)
		{
			Coord r2 = constraints[pId].pos2 - pos[idx2];

			J[4 * pId + 2] = -n;
			J[4 * pId + 3] = -(r2.cross(n));
			B[4 * pId + 2] = -n / mass[idx2];
			B[4 * pId + 3] = inertia[idx2].inverse() * (-r2.cross(n));
		}
		else
		{
			J[4 * pId + 2] = Coord(0);
			J[4 * pId + 3] = Coord(0);
			B[4 * pId + 2] = Coord(0);
			B[4 * pId + 3] = Coord(0);
		}
	}

	template <typename Coord>
	__global__ void CalculateDiagonals(
		DArray<Real> D,
		DArray<Coord> J,
		DArray<Coord> B)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= J.size() / 4) return;

		Real d = Real(0);
		d += J[4 * tId].dot(B[4 * tId]);
		d += J[4 * tId + 1].dot(B[4 * tId + 1]);
		d += J[4 * tId + 2].dot(B[4 * tId + 2]);
		d += J[4 * tId + 3].dot(B[4 * tId + 3]);

		D[tId] = d;
	}

	// ignore zeta !!!!!!
	template <typename Coord, typename Constraint>
	__global__ void CalculateEta(
		DArray<Real> eta,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> J,
		DArray<Real> mass,
		DArray<Constraint> constraints,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;

		Real invDt = Real(1) / dt;

		//TODO: add user control to beta
		Real beta = Real(1) / Real(15);

		int idx1 = constraints[pId].bodyId1;
		int idx2 = constraints[pId].bodyId2;

		Real ita_i = Real(0);
		ita_i -= J[4 * pId].dot(velocity[idx1]);
		ita_i -= J[4 * pId + 1].dot(angular_velocity[idx1]);

		if (idx2 != INVLIDA_ID)
		{
			ita_i -= J[4 * pId + 2].dot(velocity[idx2]);
			ita_i -= J[4 * pId + 3].dot(angular_velocity[idx2]);
		}

		eta[pId] = ita_i * invDt;

		if (constraints[pId].type == ConstraintType::CN_NONPENETRATION)
		{
			eta[pId] += beta * constraints[pId].interpenetration * invDt * invDt;
		}
	}

	template <typename Contact, typename Constraint>
	__global__ void SetupConstraints(
		DArray<Constraint> constraints,
		DArray<Contact> contacts,
		int contact_size,
		bool hasFriction)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= contact_size) return;

		constraints[pId].bodyId1 = contacts[pId].bodyId1;
		constraints[pId].bodyId2 = contacts[pId].bodyId2;
		constraints[pId].pos1 = contacts[pId].pos1;
		constraints[pId].pos2 = contacts[pId].pos2;
		constraints[pId].normal1 = contacts[pId].normal1;
		constraints[pId].normal2 = contacts[pId].normal2;
		constraints[pId].interpenetration = contacts[pId].interpenetration;
		constraints[pId].type = ConstraintType::CN_NONPENETRATION;

		if (hasFriction)
		{
			Coord3D n = contacts[pId].normal1;
			n /= n.norm();

			Coord3D n1, n2;
			if (abs(n[1]) > EPSILON || abs(n[2]) > EPSILON)
			{
				n1 = Coord3D(0, n[2], -n[1]);
				n1 /= n1.norm();
				n2 = n1.cross(n);
				n2 /= n2.norm();
			}
			else if (abs(n[0]) > EPSILON)
			{
				n1 = Coord3D(n[2], 0, -n[0]);
				n1 /= n1.norm();
				n2 = n1.cross(n);
				n2 /= n2.norm();
			}

			constraints[pId * 2 + contact_size].bodyId1 = contacts[pId].bodyId1;
			constraints[pId * 2 + contact_size].bodyId2 = contacts[pId].bodyId2;
			constraints[pId * 2 + contact_size].pos1 = contacts[pId].pos1;
			constraints[pId * 2 + contact_size].pos2 = contacts[pId].pos2;
			constraints[pId * 2 + contact_size].normal1 = n1;
			constraints[pId * 2 + contact_size].normal2 = -n1;
			constraints[pId * 2 + contact_size].type = ConstraintType::CN_FRICTION;

			constraints[pId * 2 + 1 + contact_size].bodyId1 = contacts[pId].bodyId1;
			constraints[pId * 2 + 1 + contact_size].bodyId2 = contacts[pId].bodyId2;
			constraints[pId * 2 + 1 + contact_size].pos1 = contacts[pId].pos1;
			constraints[pId * 2 + 1 + contact_size].pos2 = contacts[pId].pos2;
			constraints[pId * 2 + 1 + contact_size].normal1 = n2;
			constraints[pId * 2 + 1 + contact_size].normal2 = -n2;
			constraints[pId * 2 + 1 + contact_size].type = ConstraintType::CN_FRICTION;
		}
	}

	template<typename TDataType>
	void IterativeConstraintSolver<TDataType>::initializeJacobian(Real dt)
	{
		if (this->inContacts()->isEmpty())
			return;

		auto& contacts = this->inContacts()->getData();
		int sizeOfContacts = contacts.size();

 		int sizeOfConstraints = sizeOfContacts;
		if (this->varFrictionEnabled()->getData())
		{
			sizeOfConstraints += 2 * sizeOfContacts;
		}

		if (sizeOfConstraints == 0) return;

		mAllConstraints.resize(sizeOfConstraints);

		cuExecute(sizeOfContacts,
			SetupConstraints,
			mAllConstraints,
			contacts,
			sizeOfContacts,
			this->varFrictionEnabled()->getData());

		mJ.resize(4 * sizeOfConstraints);
		mB.resize(4 * sizeOfConstraints);
		mD.resize(sizeOfConstraints);
		mEta.resize(sizeOfConstraints);
		mLambda.resize(sizeOfConstraints);

		auto sizeOfRigids = this->inCenter()->size();
		mContactNumber.resize(sizeOfRigids);

		mJ.reset();
		mB.reset();
		mD.reset();
		mEta.reset();
		mLambda.reset();
		mContactNumber.reset();

		cuExecute(sizeOfConstraints,
			CalculateNbrCons,
			mAllConstraints,
			mContactNumber
		);
		cuExecute(sizeOfConstraints,
			CalculateJacobians,
			mJ,
			mB,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			mAllConstraints);

		cuExecute(sizeOfConstraints,
			CalculateDiagonals,
			mD,
			mJ,
			mB);

		cuExecute(sizeOfConstraints,
			CalculateEta,
			mEta,
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			mJ,
			this->inMass()->getData(),
			mAllConstraints,
			dt);
	}



	DEFINE_CLASS(IterativeConstraintSolver);
}