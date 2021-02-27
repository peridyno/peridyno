#pragma once
#include "Framework/ModuleTopology.h"
#include "Topology/NeighborList.h"
#include "Vector.h"
#include "Array/ArrayList.h"


namespace dyno
{
	template<typename TDataType>
	class PointSet : public TopologyModule
	{
		DECLARE_CLASS_1(PointSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PointSet();
		~PointSet() override;

		void copyFrom(PointSet<TDataType>& pointSet);

		void setPoints(std::vector<Coord>& pos);
		void setPoints(GArray<Coord>& pos);
		void setSize(int size);

		GArray<Coord>& getPoints() { return m_coords; }

		int getPointSize() { return m_coords.size(); };

		ArrayList<int>* getPointNeighbors();
		virtual void updatePointNeighbors();

		void scale(Real s);
		void scale(Coord s);
		void translate(Coord t);

		void loadObjFile(std::string filename);

	protected:
		bool initializeImpl() override;

		GArray<Coord> m_coords;
		ArrayList<int> m_pointNeighbors;
	};


#ifdef PRECISION_FLOAT
	template class PointSet<DataType3f>;
#else
	template class PointSet<DataType3d>;
#endif
}

