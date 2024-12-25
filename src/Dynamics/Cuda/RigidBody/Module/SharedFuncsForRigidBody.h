#pragma once
#include "Array/ArrayList.h"

#include "STL/Pair.h"

#include "Matrix/Transform3x3.h"

#include "Collision/CollisionData.h"

#include "Topology/DiscreteElements.h"

#include "Algorithm/Reduction.h"

namespace dyno 
{
	void ApplyTransform(
		DArrayList<Transform3f>& instanceTransform,
		const DArray<Vec3f>& translate,
		const DArray<Mat3f>& rotation,
		const DArray<Mat3f>& rotationInit,
		const DArray<Pair<uint, uint>>& binding,
		const DArray<int>& bindingtag);

	void updateVelocity(
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Vec3f> impulse,
		float linearDamping,
		float angularDamping,
		float dt
	);

	void updateGesture(
		DArray<Vec3f> pos,
		DArray<Quat1f> rotQuat,
		DArray<Mat3f> rotMat,
		DArray<Mat3f> inertia,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Mat3f> inertia_init,
		float dt
	);

	void updatePositionAndRotation(
		DArray<Vec3f> pos,
		DArray<Quat1f> rotQuat,
		DArray<Mat3f> rotMat,
		DArray<Mat3f> inertia,
		DArray<Mat3f> inertia_init,
		DArray<Vec3f> impulse_constrain
	);

	void calculateContactPoints(
		DArray<TContactPair<float>> contacts,
		DArray<int> contactCnt
	);


	void calculateJacobianMatrix(
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<Vec3f> pos,
		DArray<Mat3f> inertia,
		DArray<float> mass,
		DArray<Mat3f> rotMat,
		DArray<TConstraintPair<float>> constraints
	);

	void calculateJacobianMatrixForNJS(
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<Vec3f> pos,
		DArray<Mat3f> inertia,
		DArray<float> mass,
		DArray<Mat3f> rotMat,
		DArray<TConstraintPair<float>> constraints
	);


	void calculateEtaVectorForPJS(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<TConstraintPair<float>> constraints
	);

	void calculateEtaVectorForPJSBaumgarte(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Vec3f> pos,
		DArray<Quat1f> rotation_q,
		DArray <TConstraintPair<float>> constraints,
		DArray<float> errors,
		float slop,
		float beta,
		float dt
	);

	void calculateEtaVectorWithERP(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray<Vec3f> pos,
		DArray<Quat1f> rotation_q,
		DArray <TConstraintPair<float>> constraints,
		DArray<float> ERP,
		float slop,
		float dt
	);

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
	);
	
	void calculateEtaVectorForNJS(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> pos,
		DArray<Quat1f> rotation_q,
		DArray <TConstraintPair<float>> constraints,
		float slop,
		float beta
	);
	
	void setUpContactsInLocalFrame(
		DArray<TContactPair<float>> contactsInLocalFrame,
		DArray<TContactPair<float>> contactsInGlobalFrame,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat
	);
	
	void setUpContactAndFrictionConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<TContactPair<float>> contactsInLocalFrame,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat,
		bool hasFriction
	);
	
	void setUpContactConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<TContactPair<float>> contactsInLocalFrame,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat
	);

	void setUpBallAndSocketJointConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<BallAndSocketJoint<float>> joints,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat,
		int begin_index
	);
	
	void setUpSliderJointConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<SliderJoint<float>> joints,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat,
		int begin_index
	);

	void setUpHingeJointConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<HingeJoint<float>> joints,
		DArray<Vec3f> pos,
		DArray<Mat3f> rotMat,
		DArray<Quat1f> rotation_q,
		int begin_index
	);

	void setUpFixedJointConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<FixedJoint<float>> joints,
		DArray<Mat3f> rotMat,
		int begin_index
	);

	void setUpPointJointConstraints(
		DArray<TConstraintPair<float>> constraints,
		DArray<PointJoint<float>> joints,
		DArray<Vec3f> pos,
		int begin_index
	);

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
	);

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
	);


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
	);

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
	);

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
	);

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
	);

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
	);

	void setUpGravity(
		DArray<Vec3f> impulse_ext,
		float g,
		float dt
	);


	Real checkOutError(
		DArray<Vec3f> J,
		DArray<Vec3f> mImpulse,
		DArray<TConstraintPair<float>> constraints,
		DArray<float> eta
	);

	void calculateDiagnals(
		DArray<float> d,
		DArray<Vec3f> J,
		DArray<Vec3f> B
	);

	void preConditionJ(
		DArray<Vec3f> J,
		DArray<float> d,
		DArray<float> eta
	);

	bool saveVectorToFile(
		const std::vector<float>& vec,
		const std::string& filename
	);


	void calculateEtaVectorForRelaxation(
		DArray<float> eta,
		DArray<Vec3f> J,
		DArray<Vec3f> velocity,
		DArray<Vec3f> angular_velocity,
		DArray <TConstraintPair<float>> constraints
	);

	double checkOutErrors(
		DArray<float> errors
	);
	


	void calculateMatrixA(
		DArray<Vec3f> &J,
		DArray<Vec3f> &B,
		DArray<float> &A,
		DArray<TConstraintPair<float>> &constraints,
		float k
	);

	bool saveMatrixToFile(
		DArray<float> &Matrix,
		int n,
		const std::string& filename
	);

	bool saveVectorToFile(
		DArray<float>& vec,
		const std::string& filename
	);

	void vectorSub(
		DArray<float> &ans,
		DArray<float> &subtranhend,
		DArray<float> &minuend,
		DArray<TConstraintPair<float>> &constraints
	);

	void vectorAdd(
		DArray<float>& ans,
		DArray<float>& v1,
		DArray<float>& v2,
		DArray<TConstraintPair<float>>& constraints
	);

	void vectorMultiplyScale(
		DArray<float> &ans,
		DArray<float> &initialVec,
		float scale,
		DArray<TConstraintPair<float>>& constraints
	);

	void vectorClampSupport(
		DArray<float> v,
		DArray<TConstraintPair<float>> constraints
	);

	void vectorClampFriction(
		DArray<float> v,
		DArray<TConstraintPair<float>> constraints,
		int contact_size,
		float mu
	);

	void matrixMultiplyVec(
		DArray<Vec3f> &J,
		DArray<Vec3f> &B,
		DArray<float> &lambda,
		DArray<float> &ans,
		DArray<TConstraintPair<float>> &constraints,
		int bodyNum
	);

	float vectorNorm(
		DArray<float> &a,
		DArray<float> &b
	);

	void vectorMultiplyVector(
		DArray<float>& v1,
		DArray<float>& v2,
		DArray<float>& ans,
		DArray<TConstraintPair<float>>& constraints
	);

	void calculateImpulseByLambda(
		DArray<float> lambda,
		DArray<TConstraintPair<float>> constraints,
		DArray<Vec3f> impulse,
		DArray<Vec3f> B
	);

	void preconditionedResidual(
		DArray<float> &residual,
		DArray<float> &ans,
		DArray<float> &k_1,
		DArray<Mat2f> &k_2,
		DArray<Mat3f> &k_3,
		DArray<TConstraintPair<float>> &constraints
	);

	void buildCFMAndERP(
		DArray<Vec3f> J,
		DArray<Vec3f> B,
		DArray<TConstraintPair<float>> constraints,
		DArray<float> CFM,
		DArray<float> ERP,
		float hertz,
		float zeta,
		float dt
	);

	void calculateLinearSystemLHS(
		DArray<Vec3f>& J,
		DArray<Vec3f>& B,
		DArray<Vec3f>& impulse,
		DArray<float>& lambda,
		DArray<float>& ans,
		DArray<float>& CFM,
		DArray<TConstraintPair<float>>& constraints
	);
}
