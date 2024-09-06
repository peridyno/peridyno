#pragma once
#include "Node.h"

#include "VoxelOctree.h"

namespace dyno
{
	template<typename TDataType>
	class VolumeOctree : public Node
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VolumeOctree();
		~VolumeOctree() override;

		virtual void updateVolume() {};

		virtual Coord lowerBound() { return Coord(0); }
		virtual Coord upperBound() { return Coord(0); }

		virtual Real dx() { return Real(0); }

	public:
		DEF_VAR(bool, Inverted, false, "");

		DEF_VAR(int, LevelNumber, 3, "Number of Adaptive Levels");

		DEF_INSTANCE_STATE(VoxelOctree<TDataType>, SDFTopology, "SDF Voxel Octree");

		DArray<Coord> m_object;
		DArray<Coord> m_normal;
	};
}