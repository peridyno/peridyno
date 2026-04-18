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
	void TJSConstraintSolver<TDataType>::initializeJacobian(Real dt, bool resetLambda, DArray<ContactPair> contactsInLocalFrame, bool hasFriction)
	{
		int constraint_size = 0;
		int contact_size = contactsInLocalFrame.size();
		int contact_constraint_size = hasFriction ? 3 * contact_size : contact_size;

		auto topo = this->inDiscreteElements()->constDataPtr();

		int ballAndSocketJoint_size = topo->ballAndSocketJoints().size();
		int sliderJoint_size = topo->sliderJoints().size();
		int hingeJoint_size = topo->hingeJoints().size();
		int fixedJoint_size = topo->fixedJoints().size();
		int pointJoint_size = topo->pointJoints().size();

		constraint_size += contact_constraint_size;
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
			setUpContactAndFrictionConstraintsBlock(
				mVelocityConstraints,
				contactsInLocalFrame,
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData(),
				hasFriction
			);

			current_index += contact_constraint_size;
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
		
		if(resetLambda) mLambda.reset();

		calculateJacobianMatrix(
			mJ,
			mB,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			this->inRotationMatrix()->getData(),
			mVelocityConstraints
		);

		calculateKBlock(
			mVelocityConstraints,
			mJ,
			mB,
			this->inCenter()->getData(),
			this->inInertia()->getData(),
			this->inMass()->getData(),
			mK_1,
			mK_2,
			mK_3,
			hasFriction
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
		const bool hasFriction = this->varFrictionEnabled()->getData();
		const int contactConstraintStride = hasFriction ? 3 : 1;

		auto topo = this->inDiscreteElements()->constDataPtr();

		errors.push_back(calculateAverageVelocityMagnitude(this->inVelocity()->getData()));

		float d_mean, d_max;

		calculatePenetration(this->inContacts()->getData(), d_mean, d_max);

		d_means.push_back(d_mean);

		d_maxs.push_back(d_max);

		if (mImpulseC.size() != bodyNum * 2) {
			mImpulseC.resize(bodyNum * 2);
			mImpulseExt.resize(bodyNum * 2);
		}
		mImpulseC.reset();
		mImpulseExt.reset();

		Real dt = this->inTimeStep()->getData();

		if (!this->inContacts()->isEmpty() || topo->totalJointSize() > 0) {
			if (mContactsInLocalFrame.size() != this->inContacts()->size()) {
				mContactsInLocalFrame.resize(this->inContacts()->size());
			}

			setUpContactsInLocalFrame(
				mContactsInLocalFrame,
				this->inContacts()->getData(),
				this->inCenter()->getData(),
				this->inRotationMatrix()->getData()
			);

			DArray<ContactPair>* solverContactsPtr = &mContactsInLocalFrame;
			if (this->varContactReductionEnabled()->getValue() && mContactsInLocalFrame.size() > 0) {
				reduceContacts(
					mReducedContacts,
					mContactsInLocalFrame,
					this->varMaxReducedContactsPerPair()->getValue(),
					this->varContactReductionDistance()->getValue(),
					this->varContactReductionNormalCosThreshold()->getValue()
				);
				if (mReducedContacts.size() > 0) {
					solverContactsPtr = &mReducedContacts;
				}
			}
			else {
				mReducedContacts.resize(0);
			}

			auto& solverContacts = *solverContactsPtr;

			if (mContactNumber.size() != bodyNum) {
				mContactNumber.resize(bodyNum);
			}

			mContactNumber.reset();

			calculateContactPoints(
				solverContacts,
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
				
				if (i == 0) {
					initializeJacobian(dh, true, solverContacts, hasFriction);
					if (this->varwarmStartEnabled()->getValue() && (cacheContacts.size() != 0 || mLambdaOld.size() != 0)) {
						RunWarmStart(
							solverContacts,
							mLambdaOld,
							mLambda,
							cacheContacts,
							this->vardistThreshold()->getValue(),
							this->varGamma()->getValue(),
							contactConstraintStride,
							mPrevContactLambdaCount
						);
					}
				}
				else {
					if (this->varwarmStartEnabled()->getValue()) {
						initializeJacobian(dh, false, solverContacts, hasFriction);
						warmStartLambda(
							mB,
							mLambda,
							mVelocityConstraints,
							mImpulseC,
							this->varGamma()->getValue()
						);
					}
					else {
						initializeJacobian(dh, true, solverContacts, hasFriction);
					}
					
				}
				for (int j = 0; j < this->varIterationNumberForVelocitySolver()->getValue(); j++) {
					JacobiIterationForSoftBlock(
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
						this->varHertz()->getValue(),
						hasFriction
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
		
		frameNum++;

		if (frameNum == 1000) {
			std::ofstream outfile;
			outfile.open("C:/Users/admin/Desktop/TJS_S2_rms_v.txt", std::ios::out | std::ios::trunc);

			if (outfile.is_open()) {
				for (const auto& item : errors) {
					outfile << item << "\n";
				}
				outfile.close();
			}
			else {
				std::cerr << "errors" << std::endl;
			}

			std::ofstream outfile2;
			outfile2.open("C:/Users/admin/Desktop/TJS_S2_d_means.txt", std::ios::out | std::ios::trunc);

			if (outfile2.is_open()) {
				for (const auto& item : d_means) {
					outfile2 << item << "\n";
				}
				outfile2.close();
			}
			else {
				std::cerr << "errors" << std::endl;
			}

			std::ofstream outfile3;
			outfile3.open("C:/Users/admin/Desktop/TJS_S2_d_maxs.txt", std::ios::out | std::ios::trunc);

			if (outfile3.is_open()) {
				for (const auto& item : d_maxs) {
					outfile3 << item << "\n";
				}
				outfile3.close();
			}
			else {
				std::cerr << "errors" << std::endl;
			}
		}
		
		DArray<ContactPair>* cachedContactsPtr = &mContactsInLocalFrame;
		if (this->varContactReductionEnabled()->getValue() && mReducedContacts.size() > 0) {
			cachedContactsPtr = &mReducedContacts;
		}
		auto& cachedContacts = *cachedContactsPtr;

		cacheContacts.resize(cachedContacts.size());
		StoreCacheKernel(
			cachedContacts,
			mLambda,
			cacheContacts,
			contactConstraintStride
		);

		mPrevContactLambdaCount = cachedContacts.size() * contactConstraintStride;
		mLambdaOld.assign(mLambda);
	}
	DEFINE_CLASS(TJSConstraintSolver);
}
