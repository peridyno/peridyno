#pragma once
#include "STL/Pair.h"
#include "Module/ComputeModule.h"

namespace dyno
{
	template<typename TDataType> class Frame;
	template<typename TDataType> class PointSet;

	template<typename TDataType>
	class InstanceTransform : public ComputeModule
	{
		DECLARE_TCLASS(InstanceTransform, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename Pair<uint, uint> BindingPair;

		InstanceTransform();
		~InstanceTransform() override;

	public:
		DEF_ARRAY_IN(Coord, Center, DeviceType::GPU, "Center of rigid bodies");

		DEF_ARRAY_IN(Matrix, InitialRotation, DeviceType::GPU, "");

		DEF_ARRAY_IN(Matrix, RotationMatrix, DeviceType::GPU, "Rotation matrix of rigid bodies");

		DEF_ARRAY_IN(BindingPair, BindingPair, DeviceType::GPU, "");

		DEF_ARRAY_IN(int, BindingTag, DeviceType::GPU, "");

		DEF_ARRAYLIST_IN(Transform3f, InstanceTransform, DeviceType::GPU, "Instance transforms");

		DEF_ARRAYLIST_OUT(Transform3f, InstanceTransform, DeviceType::GPU, "Instance transforms");

	protected:
		void compute() override;
	};
}