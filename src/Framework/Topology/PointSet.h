#pragma once
#include "TopologyConstants.h"
#include "Module/TopologyModule.h"

namespace dyno
{
	template<typename TDataType>
	class PointSet : public TopologyModule
	{
		DECLARE_TCLASS(PointSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PointSet();
		~PointSet() override;

		void copyFrom(PointSet<TDataType>& pointSet);

		void setPoints(std::vector<Coord>& pos);
		void setPoints(DArray<Coord>& pos);
		void setSize(int size);
		void rotate(Coord angle);

		DArray<Coord>& getPoints() { return m_coords; }

		int getPointSize() { return m_coords.size(); };

		DArrayList<int>* getPointNeighbors();
		virtual void updatePointNeighbors();

		void scale(Real s);
		void scale(Coord s);
		void translate(Coord t);

		void rotate(Quat<Real> q);

		void loadObjFile(std::string filename);

	protected:
		DArray<Coord> m_coords;
		DArrayList<int> m_pointNeighbors;
	};
}

