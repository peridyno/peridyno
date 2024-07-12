
#pragma once
#include <vector>
#include <memory>
#include <string>
#include "Vector.h"
#include "Canvas.h"
#include "OBase.h"
#include "Module.h"

namespace dyno {

	class JointInfo : public OBase
	{

	public:

		JointInfo() {};

		~JointInfo()
		{
			JointInverseBindMatrix.clear();
			JointLocalMatrix.clear();
			JointWorldMatrix.clear();
			JointsId.clear();
		};

		void UpdateJointInfo(
			DArray<Mat4f>& InverseBindMatrix,
			DArray<Mat4f>& LocalMatrix,
			DArray<Mat4f>& WorldMatrix
		)
		{
			JointInverseBindMatrix.assign(InverseBindMatrix);
			JointLocalMatrix.assign(LocalMatrix);
			JointWorldMatrix.assign(WorldMatrix);
		}

		void UpdateJointInfo(
			DArray<Mat4f>& InverseBindMatrix,
			DArray<Mat4f>& LocalMatrix,
			DArray<Mat4f>& WorldMatrix,
			std::vector<int>& all_joints
		)
		{
			JointInverseBindMatrix.assign(InverseBindMatrix);
			JointLocalMatrix.assign(LocalMatrix);
			JointWorldMatrix.assign(WorldMatrix);
			JointsId.assign(all_joints);
		}

		bool isEmpty()
		{
			if(JointInverseBindMatrix.isEmpty() || JointLocalMatrix.isEmpty() || JointWorldMatrix.isEmpty())
				return true;
		}

	public:
		DArray<Mat4f> JointInverseBindMatrix;
		DArray<Mat4f> JointLocalMatrix;
		DArray<Mat4f> JointWorldMatrix;
		DArray<int> JointsId;
	};

}

