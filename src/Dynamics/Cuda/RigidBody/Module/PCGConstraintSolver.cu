#include "PCGConstraintSolver.h"
#include "SharedFuncsForRigidBody.h"

namespace dyno
{
	IMPLEMENT_TCLASS(PCGConstraintSolver, TDataType)

	template<typename TDataType>
	PCGConstraintSolver<TDataType>::PCGConstraintSolver()
		:ConstraintModule()
	{
		this->inContacts()->tagOptional(true);
	}

	template<typename TDataType>
	PCGConstraintSolver<TDataType>::~PCGConstraintSolver()
	{

	}

	template<typename TDataType>
	void PCGConstraintSolver<TDataType>::initializeJacobian(Real dt)
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
			if (mContactsInLocalFrame.size() != this->inContacts()->size()) {
				mContactsInLocalFrame.resize(this->inContacts()->size());
			}

			setUpContactsInLocalFrame(
				mContactsInLocalFrame,
				this->inContacts()->getData(),
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData()
			);

			auto& contacts = this->inContacts()->getData();
			setUpContactAndFrictionConstraints(
				mVelocityConstraints,
				mContactsInLocalFrame,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				this->varFrictionEnabled()->getData()
			);
		}

		if (ballAndSocketJoint_size != 0)
		{
			auto& joints = topo->ballAndSocketJoints();
			int begin_index = contact_size;

			if (this->varFrictionEnabled()->getData())
			{
				begin_index += 2 * contact_size;
			}

			setUpBallAndSocketJointConstraints(
				mVelocityConstraints,
				joints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				begin_index
			);
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
			setUpSliderJointConstraints(
				mVelocityConstraints,
				joints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				this->inQuaternion()->getData(),
				begin_index
			);
		}

		if (hingeJoint_size != 0)
		{
			auto& joints = topo->hingeJoints();
			int begin_index = contact_size + 3 * ballAndSocketJoint_size + 8 * sliderJoint_size;
			if (this->varFrictionEnabled()->getData())
			{
				begin_index += 2 * contact_size;
			}
			setUpHingeJointConstraints(
				mVelocityConstraints,
				joints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				this->inQuaternion()->getData(),
				begin_index
			);
		}

		if (fixedJoint_size != 0)
		{
			auto& joints = topo->fixedJoints();
			int begin_index = contact_size + 3 * ballAndSocketJoint_size + 8 * sliderJoint_size + 8 * hingeJoint_size;
			if (this->varFrictionEnabled()->getData())
			{
				begin_index += 2 * contact_size;
			}
			setUpFixedJointConstraints(
				mVelocityConstraints,
				joints,
				this->inRotationMatrix()->getData(),
				this->inQuaternion()->getData(),
				begin_index
			);;
		}

		if (pointJoint_size != 0)
		{
			auto& joints = topo->pointJoints();
			int begin_index = contact_size + 3 * ballAndSocketJoint_size + 8 * sliderJoint_size + 8 * hingeJoint_size + 6 * fixedJoint_size;
			if (this->varFrictionEnabled()->getData())
			{
				begin_index += 2 * contact_size;
			}
			setUpPointJointConstraints(
				mVelocityConstraints,
				joints,
				this->inCenter()->getData(),
				begin_index
			);
		}

		auto sizeOfRigids = this->inCenter()->size();
		mContactNumber.resize(sizeOfRigids);
		mContactNumber.reset();

		mJ.resize(4 * constraint_size);
		mB.resize(4 * constraint_size);
		mEta.resize(constraint_size);
		mLambda.resize(constraint_size);
		mResidual.resize(constraint_size);
		mP.resize(constraint_size);
		tmpArray.resize(constraint_size);
		mAp.resize(constraint_size);
		mZ.resize(constraint_size);
		mCFM.resize(constraint_size);
		mERP.resize(constraint_size);

		mK_1.resize(constraint_size);
		mK_2.resize(constraint_size);
		mK_3.resize(constraint_size);
		
		mJ.reset();
		mB.reset();
		mEta.reset();
		mLambda.reset();
		mResidual.reset();
		tmpArray.reset();
		mAp.reset();
		mP.reset();
		mZ.reset();
		mCFM.reset();
		mERP.reset();


		mK_1.reset();
		mK_2.reset();
		mK_3.reset();

		calculateJacobianMatrix(
			mJ,
			mB,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			this->inRotationMatrix()->getData(),
			mVelocityConstraints
		);

		buildCFMAndERP(
			mJ,
			mB,
			mVelocityConstraints,
			mCFM,
			mERP,
			this->varFrequency()->getValue(),
			this->varDampingRatio()->getValue(),
			dt
		);


		mErrors.resize(constraint_size);
		mErrors.reset();

		calculateEtaVectorWithERP(
			mEta,
			mJ,
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			this->inCenter()->getData(),
			this->inQuaternion()->getData(),
			mVelocityConstraints,
			mERP,
			this->varSlop()->getValue(),
			dt
		);


		calculateKWithCFM(
			mVelocityConstraints,
			mJ,
			mB,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			mK_1,
			mK_2,
			mK_3,
			mCFM
		);


		if (contact_size != 0)
		{
			calculateContactPoints(
				this->inContacts()->getData(),
				mContactNumber);
		}
	}


	template<typename TDataType>
	void PCGConstraintSolver<TDataType>::constrain()
	{
		uint bodyNum = this->inCenter()->size();

		auto topo = this->inDiscreteElements()->constDataPtr();

		mImpulseC.resize(bodyNum * 2);
		mImpulseExt.resize(bodyNum * 2);
		mImpulseC.reset();
		mImpulseExt.reset();

		Real dt = this->inTimeStep()->getData();

		if (this->varGravityEnabled()->getValue())
		{
			setUpGravity(
				mImpulseExt,
				this->varGravityValue()->getValue(),
				dt
			);
		}

		updateVelocity(
			this->inAttribute()->getData(),
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			mImpulseExt,
			this->varLinearDamping()->getValue(),
			this->varAngularDamping()->getValue(),
			dt
		);
	

		if (!this->inContacts()->isEmpty() || topo->totalJointSize() > 0)
		{
			float r_norm_old = 0.0;
			float r_norm_new = 0.0;
			float alpha = 0.0;


			initializeJacobian(dt);

			int constraint_size = mVelocityConstraints.size();
			int contact_size = this->inContacts()->size();

			if (topo->totalJointSize()!= mJointSize)
			{
				mJointSize = topo->totalJointSize();
				mLambdaOldJoint.resize(0);
			}

			if (mLambdaOldJoint.size() != 0 && topo->totalJointSize() > 0)
			{
				if (this->varFrictionEnabled()->getValue())
				{
					mLambda.assign(mLambdaOldJoint, constraint_size - 3 * contact_size, 3 * contact_size, 0);
				}
				else
				{
					mLambda.assign(mLambdaOldJoint, constraint_size - contact_size, contact_size, 0);
				}
			}
			else
			{
				if (topo->totalJointSize() > 0)
				{
					if (this->varFrictionEnabled()->getValue())
					{
						mLambdaOldJoint.resize(constraint_size - 3 * contact_size);
						mLambdaOldJoint.assign(mLambda, constraint_size - 3 * contact_size, 0, 3 * contact_size);
					}
					else
					{
						mLambdaOldJoint.resize(constraint_size - contact_size);
						mLambdaOldJoint.assign(mLambda, constraint_size - contact_size, 0, 3 * contact_size);
					}
				}
			}

			//r = b - Ax
			calculateLinearSystemLHS(
				mJ,
				mB,
				mImpulseC,
				mLambda,
				tmpArray,
				mCFM,
				mVelocityConstraints
			);

			vectorSub(
				mResidual,
				tmpArray,
				mEta,
				mVelocityConstraints
			);


			// z = M^{-1} r
			preconditionedResidual(
				mResidual,
				mZ,
				mK_1,
				mK_2,
				mK_3,
				mVelocityConstraints
			);
			
			mP.assign(mZ);
			r_norm_old = vectorNorm(mResidual, mP);
			float r_norm_init = r_norm_old;


			if (r_norm_old > EPSILON)
			{
				for (int i = 0; i < this->varIterationNumberForVelocitySolverCG()->getValue(); i++)
				{
					// compute Ap
					mImpulseC.reset();
					calculateLinearSystemLHS(
						mJ,
						mB,
						mImpulseC,
						mP,
						mAp,
						mCFM,
						mVelocityConstraints
					);

					alpha = r_norm_old / vectorNorm(mP, mAp);

					// x += alpha * p
					vectorMultiplyScale(
						tmpArray,
						mP,
						alpha,
						mVelocityConstraints
					);
					vectorAdd(
						mLambda,
						mLambda,
						tmpArray,
						mVelocityConstraints
					);
					// clamp x
					vectorClampSupport(
						mLambda,
						mVelocityConstraints
					);

					vectorClampFriction(
						mLambda,
						mVelocityConstraints,
						this->inContacts()->size(),
						this->varFrictionCoefficient()->getValue()
					);

					
					// recompute r
					mImpulseC.reset();
					calculateLinearSystemLHS(
						mJ,
						mB,
						mImpulseC,
						mLambda,
						tmpArray,
						mCFM,
						mVelocityConstraints
					);


					vectorSub(
						mResidual,
						tmpArray,
						mEta,
						mVelocityConstraints
					);

					mZold.assign(mZ);

					preconditionedResidual(
						mResidual,
						mZ,
						mK_1,
						mK_2,
						mK_3,
						mVelocityConstraints
					);

					r_norm_new = vectorNorm(mResidual, mZ);
					float r_norm_true = vectorNorm(mResidual, mResidual);
					//printf("%lf\n", r_norm_true);
					if (r_norm_true <= this->varTolerance()->getValue())
						break;

					float r_norm_spec = vectorNorm(mResidual, mZold);

					// compute p = r + (r_norm_new / r_norm_old) * p
					Real r_tmp = r_norm_new - r_norm_spec < 0 ? 0 : r_norm_new - r_norm_spec;
					vectorMultiplyScale(
						tmpArray,
						mP,
						(r_tmp / r_norm_old),
						mVelocityConstraints
					);
					vectorAdd(
						mP,
						tmpArray,
						mZ,
						mVelocityConstraints);
					r_norm_old = r_norm_new;
				}

				mImpulseC.reset();
				calculateImpulseByLambda(
					mLambda,
					mVelocityConstraints,
					mImpulseC,
					mB
				);

				if (this->varFrictionEnabled()->getValue())
				{
					mLambdaOldJoint.assign(mLambda, constraint_size - 3 * contact_size, 0, 3 * contact_size);
				}
				else
				{
					mLambdaOldJoint.assign(mLambda, constraint_size - contact_size, 0, contact_size);
				}
			}
			
			for (int i = 0; i < this->varIterationNumberForVelocitySolverJacobi()->getValue(); i++)
			{
				JacobiIterationForCFM(
					mLambda,
					mImpulseC,
					mJ,
					mB,
					mEta,
					mVelocityConstraints,
					mContactNumber,
					mK_1,
					mK_2,
					mK_3,
					this->inMass()->getData(),
					mCFM,
					this->varFrictionCoefficient()->getData(),
					this->varGravityValue()->getData(),
					dt
				);
			}

 
			updateVelocity(
				this->inAttribute()->getData(),
				this->inVelocity()->getData(),
				this->inAngularVelocity()->getData(),
				mImpulseC,
				this->varLinearDamping()->getValue(),
				this->varAngularDamping()->getValue(),
				dt
			);

			updateGesture(
				this->inAttribute()->getData(),
				this->inCenter()->getData(),
				this->inQuaternion()->getData(),
				this->inRotationMatrix()->getData(),
				this->inInertia()->getData(),
				this->inVelocity()->getData(),
				this->inAngularVelocity()->getData(),
				this->inInitialInertia()->getData(),
				dt
			);

		}
		else
		{
			updateGesture(
				this->inAttribute()->getData(),
				this->inCenter()->getData(),
				this->inQuaternion()->getData(),
				this->inRotationMatrix()->getData(),
				this->inInertia()->getData(),
				this->inVelocity()->getData(),
				this->inAngularVelocity()->getData(),
				this->inInitialInertia()->getData(),
				dt
			);
		}


	}

	DEFINE_CLASS(PCGConstraintSolver);
}