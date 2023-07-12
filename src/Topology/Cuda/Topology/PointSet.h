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
		

		int getPointSize() { return mCoords.size(); };

		/**
		 * @brief Return the lower and upper bounds for all points
		 */
		void requestBoundingBox(Coord& lo, Coord& hi);
		
		//Transform
		void scale(Real s);
		void scale(Coord s);
		void translate(Coord t);

		void rotate(Coord angle);
		void rotate(Quat<Real> q);

		void loadObjFile(std::string filename);

		virtual bool isEmpty();

		void clear();

		/**
		 * @brief Return the array of points
		 */
		DArray<Coord>& getPoints() { return mCoords; }

		/**
		 * @brief Return the neighbor lists of points
		 */
		DArrayList<int>& getPointNeighbors();

	protected:
		void updateTopology() override;

		virtual void updatePointNeighbors();

		DArray<Coord> mCoords;

		/**
		 * @brief Neighbor list storing neighborings ids. 
		 * 
		 * Note the lists is empty in default, each derived class should implement its own way to construct the neighbor lists
		 */
		DArrayList<int> mNeighborLists;
	};
}

