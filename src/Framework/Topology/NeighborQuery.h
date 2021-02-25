#pragma once
#include "Framework/ModuleCompute.h"
#include "Framework/FieldVar.h"
#include "Framework/FieldArray.h"
#include "Topology/FieldNeighbor.h"
#include "Topology/GridHash.h"
#include "Utility.h"

namespace dyno {
	template<typename ElementType> class NeighborList;

	template<typename TDataType>
	class NeighborQuery : public ComputeModule
	{
		DECLARE_CLASS_1(NeighborQuery, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		NeighborQuery();
		NeighborQuery(DeviceArray<Coord>& position);
		NeighborQuery(Real s, Coord lo, Coord hi);
		~NeighborQuery() override;
		
		void compute() override;

//		void setRadius(Real r) { m_radius.setValue(r); }
		void setBoundingBox(Coord lowerBound, Coord upperBound);

		void queryParticleNeighbors(NeighborList<int>& nbr, DeviceArray<Coord>& pos, Real radius);

		void setNeighborSizeLimit(int num) { m_maxNum = num; }

	protected:
		bool initializeImpl() override;

	private:
		void queryNeighborSize(DeviceArray<int>& num, DeviceArray<Coord>& pos, Real h);
		void queryNeighborDynamic(NeighborList<int>& nbrList, DeviceArray<Coord>& pos, Real h);

		void queryNeighborFixed(NeighborList<int>& nbrList, DeviceArray<Coord>& pos, Real h);


	public:
		/**
		* @brief Search radius
		* A positive value representing the radius of neighborhood for each point
		*/
		DEF_EMPTY_IN_VAR(Radius, Real, "Search radius");

		/**
		 * @brief Particle position
		 */
		DEF_EMPTY_IN_ARRAY(Position, Coord, DeviceType::GPU, "Particle position");

		/**
		 * @brief Ids of neighboring particles
		 */
		DEF_EMPTY_OUT_NEIGHBOR_LIST(Neighborhood, int, "Neighboring particles' ids");


	private:
		int m_maxNum;

		Coord m_lowBound;
		Coord m_highBound;

		GridHash<TDataType> m_hash;

		int* m_ids;
		Real* m_distance;

		Reduction<int> m_reduce;
		Scan m_scan;

		bool triangle_first = true;
	};

#ifdef PRECISION_FLOAT
	template class NeighborQuery<DataType3f>;
#else
	template class NeighborQuery<DataType3d>;
#endif
}