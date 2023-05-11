#pragma once
#include "CollisionData.h"

#include "Module/ComputeModule.h"

#include "Topology/DiscreteElements.h"

namespace dyno {
	/**
	 * @brief A class implementation to calculate bounding box
	 * 
	 * @tparam TDataType 
	 */
	template<typename TDataType>
	class CalculateBoundingBox : public ComputeModule
	{
		DECLARE_TCLASS(CalculateBoundingBox, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename ::dyno::TAlignedBox3D<Real> AABB;

		CalculateBoundingBox();
		~CalculateBoundingBox() override;
		
		void compute() override;

	public:
		DEF_INSTANCE_IN(DiscreteElements<TDataType>, DiscreteElements, "");

		DEF_ARRAY_OUT(AABB, AABB, DeviceType::GPU, "");
	};
}