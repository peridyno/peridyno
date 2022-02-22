#pragma once
#include "CollisionData.h"

#include "Module/CollisionModel.h"

#include "Topology/DiscreteElements.h"

namespace dyno {
	template<typename TDataType>
	class CollistionDetectionBoundingBox : public CollisionModel
	{
		DECLARE_TCLASS(CollisionModel, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TAlignedBox3D<Real> AABB;

		CollistionDetectionBoundingBox();
		~CollistionDetectionBoundingBox() override;
		
		void doCollision() override;

	public:
		DEF_INSTANCE_IN(DiscreteElements<TDataType>, DiscreteElements, "");

		DEF_ARRAY_OUT(TContactPair<Real>, Contacts, DeviceType::GPU, "");
	private:
		DArray<int> mBoundaryContactCounter;

		Coord mUpperCorner = Coord(100, 5, 100);//(0.4925,0.4925,0.4925);
		Coord mLowerCorner = Coord(-100, 0, -100);//(0.0075,0.0075,0.0075);

		Reduction<int> m_reduce;
		Scan m_scan;
	};
}