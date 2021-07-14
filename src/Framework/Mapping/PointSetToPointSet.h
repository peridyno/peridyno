#pragma once
#include "Module/TopologyMapping.h"
#include "Topology/PointSet.h"

namespace dyno
{
	template<typename TDataType> class PointSet;

	template<typename TDataType>
	class PointSetToPointSet : public TopologyMapping
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PointSetToPointSet();
		PointSetToPointSet(std::shared_ptr<PointSet<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to);
		~PointSetToPointSet() override;

		void setSearchingRadius(Real r) { m_radius = r; }

		void setFrom(std::shared_ptr<PointSet<TDataType>> from) { m_from = from; }
		void setTo(std::shared_ptr<PointSet<TDataType>> to) { m_to = to; }

		bool apply() override;

		void match(std::shared_ptr<PointSet<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to);

	protected:
		bool initializeImpl() override;

	private:
		//Searching radius
		Real m_radius = 0.0125;

		DArrayList<int> mNeighborIds;

		std::shared_ptr<PointSet<TDataType>> m_from = nullptr;
		std::shared_ptr<PointSet<TDataType>> m_to = nullptr;

		std::shared_ptr<PointSet<TDataType>> m_initFrom = nullptr;
		std::shared_ptr<PointSet<TDataType>> m_initTo = nullptr;
	};
}