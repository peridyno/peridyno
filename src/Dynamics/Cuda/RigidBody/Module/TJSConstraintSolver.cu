#include "TJSConstraintSolver.h"
#include "SharedFuncsForRigidBody.h"

namespace dyno
{
	IMPLEMENT_TCLASS(TJSConstraintSolver, TDataType)

		template<typename TDataType>
	TJSConstraintSolver<TDataType>::TJSConstraintSolver()
		:ConstraintModule()
	{
		this->inContacts()->tagOptional(true);
	}

	template<typename TDataType>
	TJSConstraintSolver<TDataType>::~TJSConstraintSolver()
	{
	}

	template<typename TDataType>
	void TJSConstraintSolver<TDataType>::initializeJacobian(Real dt)
	{
		int constraint_size = 0;
		int contact_size = this->inContacts()->size();

		auto topo = this->inDiscreteElements()->constDataPtr();

		int ballAndSocketJoint_size = topo->ballAndSocketJoints().size();
		int sliderJoint_size = topo->sliderJoints().size();
		int hingeJoint_size = topo->hingeJoints().size();
		int fixedJoint_size = topo->fixedJoints().size();
		int pointJoint_size = topo->pointJoints().size();

		if (this->varFrictionEnabled()->getValue()) {
			constraint_size += 3 * contact_size;
		}
		else {
			constraint_size += contact_size;
		}

		constraint_size += 3 * ballAndSocketJoint_size;
		constraint_size += 8 * sliderJoint_size;
		constraint_size += 8 * hingeJoint_size;
		constraint_size += 6 * fixedJoint_size;
		constraint_size += 3 * pointJoint_size;

		if (mVelocityConstraints.size() != constraint_size)
			mVelocityConstraints.resize(constraint_size);

		int current_index = 0;

		if (contact_size != 0)
		{
			auto& contacts = this->inContacts()->getData();
			setUpContactAndFrictionConstraints(
				mVelocityConstraints,
				mContactsInLocalFrame,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				this->varFrictionEnabled()->getData()
			);

			current_index += contact_size;
			if (this->varFrictionEnabled()->getData())
			{
				current_index += 2 * contact_size;
			}
		}

		if (ballAndSocketJoint_size != 0)
		{
			auto& joints = topo->ballAndSocketJoints();

			setUpBallAndSocketJointConstraints(
				mVelocityConstraints,
				joints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				current_index
			);

			current_index += 3 * ballAndSocketJoint_size;
		}

		if (sliderJoint_size != 0)
		{
			auto& joints = topo->sliderJoints();

			setUpSliderJointConstraints(
				mVelocityConstraints,
				joints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				this->inQuaternion()->getData(),
				current_index
			);
			
			current_index += 8 * sliderJoint_size;
		}

		if (hingeJoint_size != 0)
		{
			auto& joints = topo->hingeJoints();

			setUpHingeJointConstraints(
				mVelocityConstraints,
				joints,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				this->inQuaternion()->getData(),
				current_index
			);
			current_index += 8 * hingeJoint_size;
		}

		if (fixedJoint_size != 0)
		{
			auto& joints = topo->fixedJoints();
			setUpFixedJointConstraints(
				mVelocityConstraints,
				joints,
				this->inRotationMatrix()->getData(),
				this->inQuaternion()->getData(),
				current_index
			);
			current_index += 6 * fixedJoint_size;
		}

		if (pointJoint_size != 0)
		{
			auto& joints = topo->pointJoints();

			setUpPointJointConstraints(
				mVelocityConstraints,
				joints,
				this->inCenter()->getData(),
				current_index
			);
		}

		auto sizeOfRigids = this->inCenter()->size();

		if (mJ.size() != 4 * constraint_size) {
			mJ.resize(4 * constraint_size);
			mB.resize(4 * constraint_size);
			mK_1.resize(constraint_size);
			mK_2.resize(constraint_size);
			mK_3.resize(constraint_size);
			mEta.resize(constraint_size);
			mLambda.resize(constraint_size);
		}

		mJ.reset();
		mB.reset();
		mK_1.reset();
		mK_2.reset();
		mK_3.reset();
		mEta.reset();
		mLambda.reset();

		calculateJacobianMatrix(
			mJ,
			mB,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			this->inRotationMatrix()->getData(),
			mVelocityConstraints
		);

		calculateK(
			mVelocityConstraints,
			mJ,
			mB,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			mK_1,
			mK_2,
			mK_3
		);

		calculateEtaVectorForPJSoft(
			mEta,
			mJ,
			this->inVelocity()->getData(),
			this->inAngularVelocity()->getData(),
			this->inCenter()->getData(),
			this->inQuaternion()->getData(),
			mVelocityConstraints,
			this->varSlop()->getValue(),
			this->varDampingRatio()->getValue(),
			this->varHertz()->getValue(),
			this->varSubStepping()->getValue(),
			dt
		);

		
	}

	template<typename TDataType>
	void TJSConstraintSolver<TDataType>::constrain()
	{
		uint bodyNum = this->inCenter()->size();

		auto topo = this->inDiscreteElements()->constDataPtr();
		
		if (mImpulseC.size() != bodyNum * 2) {
			mImpulseC.resize(bodyNum * 2);
			mImpulseExt.resize(bodyNum * 2);
		}
		mImpulseC.reset();
		mImpulseExt.reset();

		Real dt = this->inTimeStep()->getData();

		if (!this->inContacts()->isEmpty() || topo->totalJointSize() > 0) {
			// reduce the contacts
			if (!this->inContacts()->isEmpty()) {
				float normalThreshold = 0.998f;
				float penetrationThreshold = 0.001f;

				reduceContacts_Optimized(this->inContacts()->getData(), normalThreshold, penetrationThreshold);
			}
			if (mContactsInLocalFrame.size() != this->inContacts()->size()) {
				mContactsInLocalFrame.resize(this->inContacts()->size());
			}

			setUpContactsInLocalFrame(
				mContactsInLocalFrame,
				this->inContacts()->getData(),
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData()
			);

			if (mContactNumber.size() != bodyNum) {
				mContactNumber.resize(bodyNum);
			}

			mContactNumber.reset();

			calculateContactPoints(
				this->inContacts()->getData(),
				mContactNumber);

			Real dh = dt / this->varSubStepping()->getValue();

			for (int i = 0; i < this->varSubStepping()->getValue(); i++) {
				if (this->varGravityEnabled()->getValue()) {
					setUpGravity(
						mImpulseExt,
						this->varGravityValue()->getValue(),
						dh
					);
				}

				updateVelocity(
					this->inAttribute()->getData(),
					this->inVelocity()->getData(),
					this->inAngularVelocity()->getData(),
					mImpulseExt,
					this->varLinearDamping()->getValue(),
					this->varAngularDamping()->getValue(),
					dh
				);

				mImpulseC.reset();
				initializeJacobian(dh);

				for (int j = 0; j < this->varIterationNumberForVelocitySolver()->getValue(); j++) {
					JacobiIterationForSoft(
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
						this->inFrictionCoefficients()->getData(),
						this->varGravityValue()->getValue(),
						dh,
						this->varDampingRatio()->getValue(),
						this->varHertz()->getValue()
					);
				}

				updateVelocity(
					this->inAttribute()->getData(),
					this->inVelocity()->getData(),
					this->inAngularVelocity()->getData(),
					mImpulseC,
					this->varLinearDamping()->getValue(),
					this->varAngularDamping()->getValue(),
					dh
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
					dh
				);
			}
		}
		else
		{
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

	DEFINE_CLASS(TJSConstraintSolver);
}