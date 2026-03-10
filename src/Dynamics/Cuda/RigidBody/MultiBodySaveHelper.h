#include "RigidBodySystem.h"

namespace dyno
{
	template<typename TDataType>
	class MultiBodySaveHelper
	{
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	typedef typename TDataType::Matrix Matrix;
	typedef typename ::dyno::Pair<uint, uint> BindingPair;

	typedef typename dyno::TSphere3D<Real> Sphere3D;
	typedef typename dyno::TOrientedBox3D<Real> Box3D;
	typedef typename dyno::Quat<Real> TQuat;

	typedef typename dyno::TContactPair<Real> ContactPair;

	typedef typename ::dyno::BallAndSocketJoint<Real> BallAndSocketJoint;
	typedef typename ::dyno::SliderJoint<Real> SliderJoint;
	typedef typename ::dyno::HingeJoint<Real> HingeJoint;
	typedef typename ::dyno::FixedJoint<Real> FixedJoint;
	typedef typename ::dyno::PointJoint<Real> PointJoint;

	public:
		MultiBodySaveHelper() {};
		~MultiBodySaveHelper() override {};

		void initialShapeConfig(
			ShapeConfig& shapeRigid,
			ConfigShapeType type,
			Vec3f center,
			Quat<Real> rot,
			Vec3f halfLength,
			Real radius,
			Real capsuleLength,
			Real density = 100,
			std::vector<Vec3f> tet = std::vector<Vec3f>()
		);

		void initialRigidBodyConfig(
			RigidBodyConfig& configRigid,
			std::string name,
			int rigidbodyID,
			Quat<Real> angle,
			Vector<Real, 3> linearVelocity,
			Vector<Real, 3> angularVelocity,
			Vector<Real, 3> position,
			Vector<Real, 3> offset,
			SquareMatrix<Real, 3> inertia,
			uint bodyId,
			Real friction,
			Real restitution,
			ConfigMotionType motionType,
			ConfigShapeType shapeType,
			ConfigCollisionMask collisionMask,
			std::vector<int> visualShapeIds = std::vector<int>(),
			std::vector<ShapeConfig> shapeConfigs = std::vector<ShapeConfig>()
		);

		ConfigMotionType ToConfigMotionType(BodyType bodyType);

		ConfigShapeType ToConfigShapeType(ElementType element);

		ConfigCollisionMask ToConfigCollisionMask(CollisionMask mask);

		MultiBodyBind getMultiBodyBind(
			const std::vector<RigidBodyInfo>& getRigidBodyStates,
			const std::vector<SphereInfo>& getSpheres,
			const std::vector<BoxInfo>& getBoxes,
			const std::vector<TetInfo>& getTets,
			const std::vector<CapsuleInfo>& getCapsules,
			const std::vector<BallAndSocketJoint>& getJointsBallAndSocket,
			const std::vector<SliderJoint>& getJointsSlider,
			const std::vector<HingeJoint>& getJointsHinge,
			const std::vector<FixedJoint>& getJointsFixed,
			const std::vector<PointJoint>& getJointsPoint,
			const std::vector<Pair<uint, uint>>& getShape2RigidBodyMapping
		);


	};
}