#pragma once
#include "Framework/TopologyMapping.h"
#include "Array/Array.h"
#include "Topology/TriangleSet.h"
#include "Topology/TetrahedronSet.h"

namespace dyno
{
	template<typename TDataType> class TriangleSet;

template<typename TDataType>
class TriangleSetToTriangleSet : public TopologyMapping
{
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;

	TriangleSetToTriangleSet();
	TriangleSetToTriangleSet(std::shared_ptr<TetrahedronSet<TDataType>> from, std::shared_ptr<TriangleSet<TDataType>> to);
	~TriangleSetToTriangleSet() override;

	void setSearchingRadius(Real r) { m_radius = r; }

	void setFrom(std::shared_ptr<TetrahedronSet<TDataType>> from) { m_from = from; }
	void setTo(std::shared_ptr<TriangleSet<TDataType>> to) { m_to = to; }

	bool apply() override;

protected:
	bool initializeImpl() override;

	void match(std::shared_ptr<TetrahedronSet<TDataType>> from, std::shared_ptr<TriangleSet<TDataType>> to);

private:
	//Searching radius
	Real m_radius = 0.0125;

	DArray<int> m_nearestTriangle;
	DArray<Coord> m_barycentric;

	std::shared_ptr<TetrahedronSet<TDataType>> m_initFrom = nullptr;
	std::shared_ptr<TriangleSet<TDataType>> m_initTo = nullptr;
	
	std::shared_ptr<TetrahedronSet<TDataType>> m_from = nullptr;
	std::shared_ptr<TriangleSet<TDataType>> m_to = nullptr;
};
}