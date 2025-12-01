#pragma once

#include "Topology/AdaptiveGridSet2D.h"

namespace dyno {

	template<typename TDataType>
	class AdaptiveCapillaryWaveHelper
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename Vector<Real, 2> Coord2D;
		typedef typename Vector<Real, 3> Coord3D;
		typedef typename Vector<Real, 4> Coord4D;

		static void ACWHelper_OneWaveStepVersion1(
			DArray<Coord4D>& grid_next,
			DArray<Coord4D>& grid,
			DArray<AdaptiveGridNode2D>& leaves,
			DArrayList<int>& neighbors,
			uint level_max,
			float GRAVITY,
			float timestep);

		static void ACWHelper_OneWaveStepVersion2(
			DArray<Coord4D>& grid_next,
			DArray<Coord4D>& grid,
			DArray<AdaptiveGridNode2D>& leaves,
			DArrayList<int>& neighbors,
			uint level_max,
			float GRAVITY,
			float timestep);
	};
}
