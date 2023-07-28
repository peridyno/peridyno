#pragma once
#include "CollisionData.h"

#include "Module/ComputeModule.h"

#include "Topology/DiscreteElements.h"

namespace dyno {
	template<typename TDataType>
	class CollistionDetectionBoundingBox : public ComputeModule
	{
		DECLARE_TCLASS(CollistionDetectionBoundingBox, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename ::dyno::TAlignedBox3D<Real> AABB;

		CollistionDetectionBoundingBox();
		~CollistionDetectionBoundingBox() override;

	public:
		DEF_VAR(Coord, UpperBound, Coord(100), "An upper bound for the bounding box");

		DEF_VAR(Coord, LowerBound, Coord(-100, 0, -100), "A lower bound for the bounding box");

	public:
		DEF_INSTANCE_IN(DiscreteElements<TDataType>, DiscreteElements, "");

		DEF_ARRAY_OUT(TContactPair<Real>, Contacts, DeviceType::GPU, "");

	protected:
		void compute() override;

	private:
		DArray<int> mBoundaryContactCounter;

		Reduction<int> mReduce;
		Scan<int> mScan;
	};

	IMPLEMENT_TCLASS(CollistionDetectionBoundingBox, TDataType)
}