#pragma once
#include "Module/TopologyModule.h"
#include "Topology/TopologyConstants.h"
#include "VkDeviceArray.h"

namespace dyno
{
	template<typename TDataType>
	class PointSet : public TopologyModule
	{
	public:
		using Coord = Vec3f;

		PointSet();
		~PointSet() override;

		DArray<Vec3f>& getPoints() { return mCoords; }
		int getPointSize() const { return mCoords.size(); };

		void copyFrom(PointSet& pointSet) {}

		void setPoints(std::vector<Vec3f>& points);
		void setPoints(const DArray<Vec3f>& points);

		virtual bool isEmpty();
		void clear();

				
		//Transform
		void scale(const Real s) {}
		void scale(const Coord s) {}
		void translate(const Coord t) {}

		virtual void rotate(const Coord angle) {}
		virtual void rotate(const Quat<Real> q) {}

	public:
		DArray<Vec3f> mCoords;
	};
	using PointSet3f = PointSet<DataType3f>;
}

