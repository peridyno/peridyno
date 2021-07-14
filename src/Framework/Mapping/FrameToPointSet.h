#pragma once
#include "Array/Array.h"
#include "Module/TopologyMapping.h"

namespace dyno
{
	template<typename TDataType> class Frame;
	template<typename TDataType> class PointSet;

template<typename TDataType>
class FrameToPointSet : public TopologyMapping
{
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	typedef typename TDataType::Rigid Rigid;
	typedef typename TDataType::Matrix Matrix;

	FrameToPointSet();
	FrameToPointSet(std::shared_ptr<Frame<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to);
	~FrameToPointSet() override;

	void initialize(const Rigid& rigid, DArray<Coord>& points);

	void applyTransform(const Rigid& rigid, DArray<Coord>& points);

	bool apply() override;

protected:
	bool initializeImpl() override;

private:
	void match(std::shared_ptr<Frame<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to);

	std::shared_ptr<Frame<TDataType>> m_from = nullptr;
	std::shared_ptr<PointSet<TDataType>> m_to = nullptr;

	Rigid m_refRigid;
	DArray<Coord> m_refPoints;

	std::shared_ptr<Frame<TDataType>> m_initFrom;
	std::shared_ptr<PointSet<TDataType>> m_initTo;
};
}