#pragma once
#include "Module/TopologyMapping.h"

#include "Collision/CollisionData.h"
#include "Topology/PointSet.h"
#include "Topology/Joint.h"


namespace dyno
{
	template<typename TDataType>
	class AnchorPointToPointSet : public TopologyMapping
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename BallAndSocketJoint<Real> BallAndSocketJoint;
		typedef typename SliderJoint<Real> SliderJoint;
		//typedef typename HingeJoint<Real> HingeJoint;
		//typedef typename FixedJoint<Real> FixedJoint;

		AnchorPointToPointSet();

	protected:
		bool apply() override;

	public:
		DEF_ARRAY_IN(BallAndSocketJoint, BallAndSocketJoints, DeviceType::GPU, "Ball And Socket Joints");
		DEF_ARRAY_IN(SliderJoint, SliderJoints, DeviceType::GPU, "Slider Joints");
		//DEF_ARRAY_IN(HingeJoint, HingeJoints, DeviceType::GPU, "Hinge Joints");
		//DEF_ARRAY_IN(FixedJoint, FixedJoints, DeviceType::GPU, "Fixed Joints");
		DEF_ARRAY_IN(Coord, Center, DeviceType::GPU, "Center of rigid bodies");
		DEF_ARRAY_IN(Matrix, RotationMatrix, DeviceType::GPU, "Rotation matrix of rigid bodies");

		DEF_INSTANCE_OUT(PointSet<TDataType>, PointSet, "");
	};
}