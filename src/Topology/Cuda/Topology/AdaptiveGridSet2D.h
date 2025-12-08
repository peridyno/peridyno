/**
 * Copyright 2024 Lixin Ren
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

// #include "Topology/AdaptiveGridSet.h"
//#include "Topology/TriangleSet.h"
#include "TopologyConstants.h"
#include "Module/TopologyModule.h"
#include "Primitive/Primitive2D.h"
#include "Vector/Vector2D.h"

namespace dyno 
{
	typedef unsigned int OcIndex;
	typedef unsigned long long int OcKey;
	typedef unsigned short Level;

#define MAX_LEVEL_2D 28
//#define DIM 3

	DYN_FUNC static OcKey CalculateMortonCode2D(OcIndex x, OcIndex y)
	{
		OcKey key = 0;
		OcKey mask = 1;

		for (int i = 0; i < MAX_LEVEL_2D; i++)
		{
			key |= (x & mask << i) << i | (y & mask << i) << (i + 1);
		}
		return key;
	}
	DYN_FUNC static void RecoverFromMortonCode2D(OcKey key, OcIndex& x, OcIndex& y)
	{
		x = 0;
		y = 0;

		for (int i = 0; i < MAX_LEVEL_2D; i++)
		{
			OcKey x_buf = ((key >> 2 * i) & 1U) << i;
			OcKey y_buf = ((key >> (2 * i + 1)) & 1U) << i;

			x |= x_buf;
			y |= y_buf;
		}
	}

	class AdaptiveGridNode2D
	{
	public:
		DYN_FUNC AdaptiveGridNode2D()
		{
			m_level = 0;
			m_morton = 0;
			m_fchild = EMPTY;
			m_position = Vec2f(0.0, 0.0);
		};
		DYN_FUNC AdaptiveGridNode2D(Level l,OcKey key, Vec2f pos)
		{
			m_level = l;
			m_morton = key;
			m_fchild = EMPTY;
			m_position = pos;
		};

		DYN_FUNC inline bool isContainedStrictlyIn(const AdaptiveGridNode2D& mc2) const
		{
			if (m_level >= mc2.m_level)
			{
				return false;
			}

			auto k1 = m_morton >> 2 * (mc2.m_level - m_level);
			auto k2 = mc2.m_morton;

			return k1 == k2;
		};
		DYN_FUNC bool operator> (const AdaptiveGridNode2D& mc2) const
		{
			if (isContainedStrictlyIn(mc2))
			{
				return true;
			}

			auto k1 = m_morton;
			auto k2 = mc2.m_morton;

			m_level < mc2.m_level ? (k1 = k1 >> 2 * (mc2.m_level - m_level)) : (k2 = k2 >> 2 * (m_level - mc2.m_level));

			return k1 > k2;
		};

		DYN_FUNC bool isLeaf() { return m_fchild == EMPTY; };

		Level m_level;
		OcKey m_morton;
		int m_fchild;
		Vec2f m_position;
	};



	template<typename TDataType>
	class AdaptiveGridSet2D : public TopologyModule
	{
		DECLARE_TCLASS(AdaptiveGridSet2D, TDataType)
	public:
		typedef typename TDataType::Real Real;
		//typedef typename TDataType::Coord Coord2D;
		typedef typename Vector<Real, 2> Coord2D;

		AdaptiveGridSet2D();
		~AdaptiveGridSet2D() override;

		void clear();

		DYN_FUNC inline void setLevelNum(Level num) { m_level_num = num; }
		DYN_FUNC inline void setLevelMax(Level num) { m_level_max = num; }
		DYN_FUNC inline void setDx(Real dx) { m_dx = dx; }
		DYN_FUNC inline void setOrigin(Coord2D origin) { m_origin = origin; }
		DYN_FUNC inline void setQuadType(int type) { m_quadtree_type = type; }

		DYN_FUNC inline Level adaptiveGridLevelNum2D() { return m_level_num; }
		DYN_FUNC inline Level adaptiveGridLevelMax2D() { return m_level_max; }
		DYN_FUNC inline Real adaptiveGridDx2D() { return m_dx; }
		DYN_FUNC inline Coord2D adaptiveGridOrigin2D() { return m_origin; }
		DYN_FUNC inline int adaptiveGridType2D() { return m_quadtree_type; }
		DYN_FUNC inline uint adaptiveGridLeafNum2D() { return m_leaf_num; }

		void extractLeafs(DArray<AdaptiveGridNode2D>& leaves);
		void extractLeafs(DArray<Coord2D>& pos, DArray<Real>& scale);
		void extractLeafs(DArray<Coord2D>& pos, DArrayList<int>& neighbors);
		void extractLeafs(DArray<AdaptiveGridNode2D>& leaves, DArrayList<int>& neighbors);
		void extractLeafs(DArray<Coord2D>& pos, DArray<AdaptiveGridNode2D>& leaves, DArrayList<int>& neighbors);


		DArray<AdaptiveGridNode2D>& adaptiveGridNode2D() {return m_quadtree;}
		DArrayList<int>& adaptiveGridNeighbors2D() { return m_neighbors; }
		void setAGrids(DArray<AdaptiveGridNode2D>& nodes);
		void setNeighbors(DArrayList<int>& nodes);

		void extractVertexs(DArray<Coord2D>& vertex, DArray<int>& n2v, DArrayList<int>& ver2node);

		void accessRandom(DArray<int>& index, DArray<Coord2D>& pos);

		GPU_FUNC bool accessRandomLeafs2D(int& index, OcKey morton, Level level);
		GPU_FUNC bool accessRandom2D(int& index, OcKey morton, Level level);


	protected:
		void updateTopology() override;

	private:
		void ConstructNeighbors4();

		Coord2D m_origin;
		Real m_dx;
		Level m_level_num;
		Level m_level_max;
		int m_quadtree_type;//0--vertex_balanced, 1--edge_balanced, 3--non_balanced
		uint m_leaf_num;

		DArray<AdaptiveGridNode2D> m_quadtree;
		DArrayList<int> m_neighbors;//the size is 4*(m_quadtree.size())£¬the order is: -x,+x,-y,+y
		DArray<int> m_leafIndex;//the index of leaf in all node
 	};

	//"index" represents the index in the leaf node
	template<typename TDataType>
	GPU_FUNC bool AdaptiveGridSet2D<TDataType>::accessRandomLeafs2D(int& index, OcKey morton, Level level)
	{
		//compute the l-th level`s morton, the range of l is [levelmin,level]
		auto alpha = [&](Level l) -> int {
			OcKey mo = morton >> (2 * (level - l));
			return (mo & 3);
			};

		Level levelmin = m_level_max - m_level_num + 1;
		int node_index = morton >> (2 * (level - levelmin));
		if (m_quadtree[node_index].isLeaf())
		{
			index = m_leafIndex[node_index];
			return true;
		}
		node_index = m_quadtree[node_index].m_fchild;
		for (Level i = (levelmin + 1); i <= level; i++)
		{
			node_index = node_index + alpha(i);
			if (m_quadtree[node_index].isLeaf())
			{
				index = m_leafIndex[node_index];
				return true;
			}
			else
				node_index = m_quadtree[node_index].m_fchild;
		}

		return false;
	}

	template<typename TDataType>
	GPU_FUNC bool AdaptiveGridSet2D<TDataType>::accessRandom2D(int& index,OcKey morton,Level level)//the range of level is [levelmin,max_level]
	{
		//compute the l-th level`s morton, the range of l is [levelmin,level]
		auto alpha = [&](Level l) -> int {
			OcKey mo = morton >> (2 * (level - l));
			return (mo & 3);
			};

		int node_index = morton >> (2 * (level - (m_level_max - m_level_num + 1)));
		if (m_quadtree[node_index].isLeaf())
		{
			index = node_index;
			return true;
		}
		node_index = m_quadtree[node_index].m_fchild;
		for (Level i = (m_level_max - m_level_num + 2); i <= level; i++)
		{
			node_index = node_index + alpha(i);
			if (m_quadtree[node_index].isLeaf())
			{
				index = node_index;
				return true;
			}
			else
				node_index = m_quadtree[node_index].m_fchild;
		}
		return false;
	}
}
