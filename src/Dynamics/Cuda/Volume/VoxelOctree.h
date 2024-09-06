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

#include "Topology/TriangleSet.h"
#include "Module/TopologyModule.h"
#include "Primitive/Primitive3D.h"
#include "Vector.h"

namespace dyno 
{
	typedef unsigned short OcIndex;
	typedef unsigned long long int OcKey;
	typedef unsigned short Level;

#define MAX_LEVEL 15
#define DIM 3

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

	template<typename TCoord>
	class VoxelOctreeNode
	{
	public:
		DYN_FUNC VoxelOctreeNode();
		DYN_FUNC VoxelOctreeNode(Level l, OcKey key);
		DYN_FUNC VoxelOctreeNode(Level l, OcIndex x, OcIndex y, OcIndex z);
		DYN_FUNC VoxelOctreeNode(Level l, OcIndex x, OcIndex y, OcIndex z, TCoord point_pos);

		DYN_FUNC bool operator> (const VoxelOctreeNode<TCoord>&) const;
		DYN_FUNC inline bool isContainedStrictlyIn(const VoxelOctreeNode<TCoord>&) const;

		DYN_FUNC inline OcKey key() const { return m_key; }
		DYN_FUNC inline Level level() const { return  m_level; }
		DYN_FUNC inline TCoord position() const { return m_position; }
		DYN_FUNC inline bool midside() const { return midside_node; }
		DYN_FUNC inline int child() const { return m_first_child_loc; }
		DYN_FUNC inline int value() const { return m_value_loc; }

		DYN_FUNC inline void	setKey(OcKey key) { m_key=key; }
		DYN_FUNC inline void	setLevel(Level lev) { m_level = lev; }
		DYN_FUNC inline void	setMidsideNode() { midside_node = true; }
		DYN_FUNC inline void	setChildIndex(int id) { m_first_child_loc = id; }
		DYN_FUNC inline void	setValueLocation(int id) { m_value_loc = id; }
		DYN_FUNC inline void	setPosition(TCoord pos) { m_position = pos; }

		DYN_FUNC void plusChildIndex(int id);

		int m_neighbor[6]; //x-1.x+1,y-1,y+1,z-1,z+1

	protected:
		OcKey m_key;
		Level m_level;

		int m_first_child_loc = EMPTY;

		bool midside_node = false;

		int m_value_loc = EMPTY;
		TCoord m_position;
	};

	template<typename TCoord>
	DYN_FUNC VoxelOctreeNode<TCoord>::VoxelOctreeNode()
		: m_key(0)
		, m_level(0)
	{
		m_position = TCoord(0, 0, 0);

		for (int i = 0; i < 6; i++)
		{
			m_neighbor[i] = EMPTY;
		}
	}

	template<typename TCoord>
	DYN_FUNC VoxelOctreeNode<TCoord>::VoxelOctreeNode(Level l, OcKey key)
		: m_key(key)
		, m_level(l)
	{
		m_position = TCoord(0, 0, 0);

		for (int i = 0; i < 6; i++)
		{
			m_neighbor[i] = EMPTY;
		}
	}

	template<typename TCoord>
	DYN_FUNC VoxelOctreeNode<TCoord>::VoxelOctreeNode(Level l, OcIndex x, OcIndex y, OcIndex z)
		: m_key(0)
		, m_level(l)
	{
		m_key = CalculateMortonCode(x, y, z);

		m_position = TCoord(0, 0, 0);

		for (int i = 0; i < 6; i++)
		{
			m_neighbor[i] = EMPTY;
		}
	}

	template<typename TCoord>
	DYN_FUNC VoxelOctreeNode<TCoord>::VoxelOctreeNode(Level l, OcIndex x, OcIndex y, OcIndex z, TCoord point_pos)
		: m_key(0)
		, m_level(l)
	{
		m_key = CalculateMortonCode(x, y, z);

		m_position = point_pos;

		for (int i = 0; i < 6; i++)
		{
			m_neighbor[i] = EMPTY;
		}
	}

	template<typename TCoord>
	DYN_FUNC bool VoxelOctreeNode<TCoord>::isContainedStrictlyIn(const VoxelOctreeNode<TCoord>& mc2) const
	{
		if (m_level >= mc2.m_level)
		{
			return false;
		}

		auto k1 = m_key >> 3 * (mc2.m_level - m_level);
		auto k2 = mc2.key();

		return k1 == k2;
	}

	template<typename TCoord>
	DYN_FUNC bool VoxelOctreeNode<TCoord>::operator>(const VoxelOctreeNode<TCoord>& mc2) const
	{
		if (isContainedStrictlyIn(mc2))
		{
			return true;
		}

		auto k1 = m_key;
		auto k2 = mc2.m_key;

		m_level < mc2.m_level ? (k1 = k1 >> 3 * (mc2.m_level - m_level)) : (k2 = k2 >> 3 * (m_level - mc2.m_level));

		return k1 > k2;
	}

	template<typename TCoord>
	DYN_FUNC void VoxelOctreeNode<TCoord>::plusChildIndex(int id)
	{
		if (m_first_child_loc == EMPTY)
			m_first_child_loc = id;
		else
			m_first_child_loc += id;
	}

	template<typename TDataType>
	class VoxelOctree : public TopologyModule
	{
		DECLARE_TCLASS(VoxelOctree, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VoxelOctree();
		~VoxelOctree() override;


		DYN_FUNC inline uint size() const { return m_octree.size(); }

		DYN_FUNC inline void setLevelNum(int num) { level_num = num; }
		DYN_FUNC inline void setGrid(int nx, int ny, int nz) { m_nx = nx; m_ny = ny; m_nz = nz; }
		DYN_FUNC inline void setDx(Real dx) { m_dx = dx; }
		DYN_FUNC inline void setOrigin(Coord origin) { m_origin = origin; }
		DYN_FUNC inline void setLevel0(int level0) { m_level0 = level0; }

		DYN_FUNC inline int getLevelNum() { return level_num; }
		DYN_FUNC inline Real getDx() { return m_dx; }
		DYN_FUNC inline Coord getOrigin() { return m_origin; }
		DYN_FUNC inline Coord getTopOrigin() { return (m_origin + m_dx * Coord(m_nx, m_ny, m_nz)); }
		DYN_FUNC inline int getLevel0() { return m_level0; }
		DYN_FUNC inline void getGrid(int& nx, int& ny, int& nz) { nx = m_nx; ny = m_ny; nz = m_nz; }

		GPU_FUNC inline VoxelOctreeNode<Coord>& operator [] (unsigned int id) {
			return m_octree[id];
		}

		void setVoxelOctree(DArray<VoxelOctreeNode<Coord>>& oct);

		GPU_FUNC void getNode(int point_i, int point_j, int point_k, VoxelOctreeNode<Coord>& node_index,int& id)
		{
			Real top_index = pow(Real(2), int(level_num - 1));
			int top_nx = m_nx / top_index;
			int top_ny = m_ny / top_index;
			int top_nz = m_nz / top_index;
			int top_level_start = m_octree.size() - top_nx * top_ny*top_nz;

			int i_top = std::floor(point_i / top_index);
			int j_top = std::floor(point_j / top_index);
			int k_top = std::floor(point_k / top_index);

			int point_index = i_top + j_top * top_nx + k_top * top_nx*top_ny;

			id = top_level_start + point_index;
			VoxelOctreeNode<Coord> node = m_octree[top_level_start + point_index];

			while (node.midside())
			{
				int l_this = node.level();
				int i_this = std::floor(point_i / pow(Real(2), int(l_this - 1)));
				int j_this = std::floor(point_j / pow(Real(2), int(l_this - 1)));
				int k_this = std::floor(point_k / pow(Real(2), int(l_this - 1)));
				OcKey key_this = CalculateMortonCode(i_this, j_this, k_this);
				int gIndex = key_this & 7U;

				int child_index = node.child();
				id = child_index + (7 - gIndex);
				node = m_octree[child_index + (7 - gIndex)];
				//std::printf("the_first_node: gIndex: %d; the level is: %d; the key is: %lld; the distance is %f; the first child loc is: %d \n", gIndex, node.level(), node.key(), node.value(), node.child());
			}
			node_index = node;
		}

		void getLeafs(DArray<Coord>& pos, DArray<int>& pos_pos);

		//返回所有的叶节点的顶点坐标
		void getCellVertices(DArray<Coord>& pos);
		//返回最精细一层的叶节点坐标
		void getCellVertices0(DArray<Coord>& pos);
		//返回均匀网格（dx）的坐标
		void getCellVertices1(DArray<Coord>& pos);
		//返回最精细一层内部的叶节点坐标
		void getCellVertices2(DArray<Coord>& pos);

		void setSdfValues(DArray<Real>& vals);

		DArray<Real>& getSdfValues() {
			return sdfValues;
		};

		void getLeafsValue(DArray<Coord>& pos, DArray<Real>& val);

		//void getOctreeBW(DArray<int>& nodes_bw);

		void updateNeighbors();

		DArray<VoxelOctreeNode<Coord>>& getVoxelOctree() {return m_octree;}
		DArrayList<int>& getNeighbors() { return m_neighbors; }


		void getSignDistance(
			DArray<Coord> point_pos,
			DArray<Real>& point_sdf,
			DArray<Coord>& point_normal,
			bool inverted = false);

		void getSignDistanceKernel(
			DArray<Coord> point_pos,
			DArray<Real>& point_sdf);

		void getSignDistanceMLS(
			DArray<Coord> point_pos,
			DArray<Real>& point_sdf,
			DArray<Coord>& point_normal,
			bool inverted = false);

	private:
		int m_nx;
		int m_ny;
		int m_nz;

		int level_num = 3;
		int m_level0;

		Real m_dx;
		Coord m_origin;

		DArray<VoxelOctreeNode<Coord>> m_octree;
		DArray<Real> sdfValues;
		DArrayList<int> m_neighbors;
	};

}
