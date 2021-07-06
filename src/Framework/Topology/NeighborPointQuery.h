#pragma once
#include "Framework/ModuleCompute.h"

namespace dyno 
{
	template<typename TDataType>
	class NeighborPointQuery : public ComputeModule
	{
		DECLARE_CLASS_1(NeighborPointQuery, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		NeighborPointQuery();
		~NeighborPointQuery() override;
		
		void compute() override;

	private:
		void requestDynamicNeighborIds();

		void requestFixedSizeNeighborIds();

	public:
		DEF_VAR(uint, SizeLimit, 0, "Maximum number of neighbors");

		/**
		* @brief Search radius
		* A positive value representing the radius of neighborhood for each point
		*/
		DEF_VAR_IN(Real, Radius, "Search radius");

		/**
		 * @brief A set of points to be required from.
		 */
		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "A set of points to be required from");

		/**
		 * @brief Another set of points the algorithm will require neighbor ids for.
		 *		  If not set, the set of points in Position will be required.
		 */
		DEF_ARRAY_IN(Coord, Other, DeviceType::GPU, 
			"Another set of points the algorithm will require neighbor ids for. If not set, the set of points in Position will be required.");

		/**
		 * @brief Ids of neighboring particles
		 */
		DEF_ARRAYLIST_OUT(int, NeighborIds, DeviceType::GPU, "Return neighbor ids");
	};
}