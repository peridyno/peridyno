/**
 * Copyright 2022 Lixin Ren
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "Topology/AdaptiveGridSet2D.h"
#include "TopologyConstants.h"
#include "Module/TopologyModule.h"
#include "Primitive/Primitive3D.h"
#include "Vector.h"

namespace dyno 
{
	//typedef unsigned short OcIndex;
	//typedef unsigned long long int OcKey;
	//typedef unsigned short Level;

#define MAX_LEVEL 15
//#define DIM 3

	DYN_FUNC static OcKey CalculateMortonCode(OcIndex x, OcIndex y, OcIndex z)
	{
		OcKey key = 0;
		OcKey mask = 1;

		for (int i = 0; i < MAX_LEVEL; i++)
		{
			key |= (x & mask << i) << 2 * i | (y & mask << i) << (2 * i + 1) | (z & mask << i) << (2 * i + 2);
		}
		return key;
	}
	DYN_FUNC static void RecoverFromMortonCode(OcKey key, OcIndex& x, OcIndex& y, OcIndex& z)
	{
		x = 0;
		y = 0;
		z = 0;

		for (int i = 0; i < MAX_LEVEL; i++)
		{
			OcKey x_buf = ((key >> 3 * i) & 1U) << i;
			OcKey y_buf = ((key >> (3 * i + 1)) & 1U) << i;
			OcKey z_buf = ((key >> (3 * i + 2)) & 1U) << i;

			x |= x_buf;
			y |= y_buf;
			z |= z_buf;
		}
	}

	DYN_FUNC inline OcKey splitBy3(const OcIndex a) {
		OcKey x = a;
		x = (x | x << 32) & 0x1f00000000ffff;
		x = (x | x << 16) & 0x1f0000ff0000ff;
		x = (x | x << 8) & 0x100f00f00f00f00f;
		x = (x | x << 4) & 0x10c30c30c30c30c3;
		x = (x | x << 2) & 0x1249249249249249;
		return x;
	}
	// ENCODE 3D 64-bit morton code : Magic bits
	DYN_FUNC inline OcKey morton3D_64_Encode_magicbits(const OcIndex x, const OcIndex y, const OcIndex z) {
		return splitBy3(x) | (splitBy3(y) << 1) | (splitBy3(z) << 2);
	}

	DYN_FUNC inline OcIndex getThirdBits(const OcKey a) {
		OcKey x = a & 0x1249249249249249;
		x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
		x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
		x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
		x = (x ^ (x >> 16)) & 0x1f00000000ffff;
		x = (x ^ (x >> 32)) & 0x1fffff;
		return (OcIndex)x;
	}
	// DECODE 3D 64-bit morton code : Magic bits
	DYN_FUNC inline void morton3D_64_Decode_magicbits(const OcKey morton, OcIndex& x, OcIndex& y, OcIndex& z) {
		x = getThirdBits(morton);
		y = getThirdBits(morton >> 1);
		z = getThirdBits(morton >> 2);
	}

	class AdaptiveGridNode
	{
	public:
		DYN_FUNC AdaptiveGridNode()
		{
			m_level = 0;
			m_morton = 0;
			m_fchild = EMPTY;
			m_position = Vec3f(0.0, 0.0, 0.0);
		};
		DYN_FUNC AdaptiveGridNode(Level l,OcKey key, Vec3f pos)
		{
			m_level = l;
			m_morton = key;
			m_fchild = EMPTY;
			m_position = pos;
		};

		DYN_FUNC inline bool isContainedStrictlyIn(const AdaptiveGridNode& mc2) const
		{
			if (m_level >= mc2.m_level)
			{
				return false;
			}

			auto k1 = m_morton >> 3 * (mc2.m_level - m_level);
			auto k2 = mc2.m_morton;

			return k1 == k2;
		};
		DYN_FUNC bool operator> (const AdaptiveGridNode& mc2) const
		{
			if (isContainedStrictlyIn(mc2))
			{
				return true;
			}

			auto k1 = m_morton;
			auto k2 = mc2.m_morton;

			m_level < mc2.m_level ? (k1 = k1 >> 3 * (mc2.m_level - m_level)) : (k2 = k2 >> 3 * (m_level - mc2.m_level));

			return k1 > k2;
		};

		DYN_FUNC bool isLeaf() { return m_fchild == EMPTY; };

		Level m_level;
		OcKey m_morton;
		int m_fchild;
		Vec3f m_position;
	};



	template<typename TDataType>
	class AdaptiveGridSet : public TopologyModule
	{
		DECLARE_TCLASS(AdaptiveGridSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		AdaptiveGridSet();
		~AdaptiveGridSet() override;

		DYN_FUNC inline void setLevelNum(Level num) { m_level_num = num; }
		DYN_FUNC inline void setLevelMax(Level num) { m_level_max = num; }
		DYN_FUNC inline void setDx(Real dx) { m_dx = dx; }
		DYN_FUNC inline void setOrigin(Coord origin) { m_origin = origin; }
		DYN_FUNC inline void setOctreeType(int type) { m_octree_type = type; }
		DYN_FUNC inline void setNeighborType(int type) { m_neighbor_type = type; }

		DYN_FUNC inline Level adaptiveGridLevelNum() { return m_level_num; }
		DYN_FUNC inline Level adaptiveGridLevelMax() { return m_level_max; }
		DYN_FUNC inline Real adaptiveGridDx() { return m_dx; }
		DYN_FUNC inline Coord adaptiveGridOrigin() { return m_origin; }
		DYN_FUNC inline int adaptiveGridType() { return m_octree_type; }
		DYN_FUNC inline int adaptiveGridLeafNum() { return m_leafs_num; }

		void extractLeafs27	(DArray<Coord>& pos, DArrayList<int>& neighbors);
		void extractLeafs6(DArray<AdaptiveGridNode>& leafs, DArrayList<int>& neighbors);
		void extractLeafs(DArray<AdaptiveGridNode>& leafs);
		/*m_node2Ver: the size is 8 * (m_octree.size()), anti - clockwise direction(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1); (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
		  m_vertex_neighbor: the size is 6*(vertex_num), neighbor order is: -x,+x,-y,+y,-z,+z */
		void extractVertex(DArray<Coord>& m_vertex, DArray<int>& m_vertex_neighbor, DArray<int>& m_node2Ver);

		//get the leaf nodes that overlap with the z=zAxis plane
		void extractLeafs2D(DArray<AdaptiveGridNode>& pos, DArrayList<int>& neighbors,Real zAxis);

		GPU_FUNC bool accessRandom(int& index, OcKey morton, Level level);
		GPU_FUNC bool accessRandomLeafs(int& index, OcKey morton, Level level);
		void accessRandom(DArray<int>& index, DArray<Coord>& pos);

		DArray<AdaptiveGridNode>& adaptiveGridNode() {return m_octree;}
		DArrayList<int>& adaptiveGridNeighbors() { return m_neighbors; }
		DArrayList<int>& adaptiveGridNeighbors27() { return m_neighbors27; }

		void setNodes(DArray<AdaptiveGridNode>& nodes);
		void setNeighbors(DArrayList<int>& nodes);

	protected:
		void updateTopology() override;

	private:
		void ConstructLeafs();
		void ConstructNeighborsSix(); 
		void ConstructNeighborsTwentySeven();

		Coord m_origin;
		Real m_dx;
		Level m_level_num;
		Level m_level_max;
		int m_octree_type;//0--vertex_balanced, 1--edge_balanced, 2--face_balanced, 3--non_balanced
		int m_neighbor_type;//0--SIX_NEIGHBOR, 1--TWENTY_SEVEN_NEIGHBOR
		int m_leafs_num;//the number of leaf nodes

		DArray<AdaptiveGridNode> m_octree;
		DArrayList<int> m_neighbors;//the size is 6*(m_octree.size()),the order is: -x,+x,-y,+y,-z,+z
		DArrayList<int> m_neighbors27;//the size is (m_octree.size())
		DArray<int> m_leafIndex;//the index of leaf in all node
 	};

	template<typename TDataType>
	GPU_FUNC bool AdaptiveGridSet<TDataType>::accessRandom(int& index, OcKey morton, Level level)//the range of level is [levelmin,max_level]
	{
		//compute the l-th level`s morton, the range of l is [levelmin,level]
		auto alpha = [&](Level l) -> int {
			OcKey mo = morton >> (3 * (level - l));
			return (mo & 7);
			};

		int node_index = morton >> (3 * (level - (m_level_max - m_level_num + 1)));
		index = node_index;
		if (m_octree[node_index].isLeaf()) return true;

		node_index = m_octree[node_index].m_fchild;
		for (Level i = (m_level_max - m_level_num + 2); i <= level; i++)
		{
			node_index = node_index + alpha(i);
			index = node_index;
			if (m_octree[node_index].isLeaf())
				return true;
			else
				node_index = m_octree[node_index].m_fchild;
		}

		return false;
	}

	template<typename TDataType>
	GPU_FUNC bool AdaptiveGridSet<TDataType>::accessRandomLeafs(int& index, OcKey morton, Level level)//the range of level is [levelmin,max_level]
	{
		//compute the l-th level`s morton, the range of l is [levelmin,level]
		auto alpha = [&](Level l) -> int {
			OcKey mo = morton >> (3 * (level - l));
			return (mo & 7);
			};

		int node_index = morton >> (3 * (level - (m_level_max - m_level_num + 1)));
		if (m_octree[node_index].isLeaf())
		{
			index = m_leafIndex[node_index];
			return true;
		}

		node_index = m_octree[node_index].m_fchild;
		for (Level i = (m_level_max - m_level_num + 2); i <= level; i++)
		{
			node_index = node_index + alpha(i);
			if (m_octree[node_index].isLeaf())
			{
				index = m_leafIndex[node_index];
				return true;
			}
			else
				node_index = m_octree[node_index].m_fchild;
		}

		return false;
	}
}
