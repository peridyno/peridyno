#pragma once

#include "Volume/VolumeOctree.h"

namespace dyno {

	class PositionNode
	{
	public:

		DYN_FUNC PositionNode()
		{
			surface_index = EMPTY;
			position_index = 0;
		}
		DYN_FUNC PositionNode(int surf, OcKey pos)
		{
			surface_index = surf;
			position_index = pos;
		}
		DYN_FUNC bool operator> (const PositionNode& ug) const
		{
			return position_index > ug.position_index;
		}

		int surface_index;
		OcKey position_index;
	};
	struct PositionCmp
	{
		DYN_FUNC bool operator()(const PositionNode& A, const PositionNode& B)
		{
			return A > B;
		}
	};

	class IndexNode
	{
	public:
		DYN_FUNC IndexNode()
		{
			node_index = EMPTY;
			xyz_index = EMPTY;
		}

		DYN_FUNC IndexNode(int xyz, int node)
		{
			node_index = node;
			xyz_index = xyz;
		}

		DYN_FUNC bool operator< (const IndexNode& ug) const
		{
			return xyz_index < ug.xyz_index;
		}

		int node_index;
		int xyz_index;
	};
	struct IndexCmp
	{
		DYN_FUNC bool operator()(const IndexNode& A, const IndexNode& B)
		{
			return A < B;
		}
	};

	template<typename TDataType>
	class VolumeHelper
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		static void levelBottom(DArray<VoxelOctreeNode<Coord>>& grid0,
			DArray<Real>& grid0_value,
			DArray<Coord>& grid0_object,
			DArray<Coord>& grid0_normal,
			std::shared_ptr<TriangleSet<TDataType>> triSet,
			Coord m_origin,
			int m_nx,
			int m_ny,
			int m_nz,
			int padding,
			int& m_level0,
			Real m_dx);

		static void levelMiddle(DArray<VoxelOctreeNode<Coord>>& grid1,
			DArray<Real>& grid1_value,
			DArray<Coord>& grid1_object,
			DArray<Coord>& grid1_normal,
			DArray<VoxelOctreeNode<Coord>>& grid0,
			DArray<Coord>& grid0_object,
			DArray<Coord>& grid0_normal,
			Coord m_origin,
			Level multi_level,
			int m_nx,
			int m_ny,
			int m_nz,
			Real m_dx);

		static void levelTop(DArray<VoxelOctreeNode<Coord>>& grid2,
			DArray<Real>& grid2_value,
			DArray<Coord>& grid2_object,
			DArray<Coord>& grid2_normal,
			DArray<VoxelOctreeNode<Coord>>& grid1,
			DArray<Coord>& grid1_object,
			DArray<Coord>& grid1_normal,
			Coord m_origin,
			Level multi_level,
			int m_nx,
			int m_ny,
			int m_nz,
			Real m_dx);

		static void levelCollection(DArray<VoxelOctreeNode<Coord>>& grids,
			DArray<Real>& grids_value,
			DArray<Coord>& grids_object,
			DArray<Coord>& grids_normal,
			DArray<VoxelOctreeNode<Coord>>& grid1,
			DArray<Real>& grid1_value,
			DArray<Coord>& grid1_object,
			DArray<Coord>& grid1_normal,
			int uplevel_num);

		static void collectionGridsTwo(
			DArray<VoxelOctreeNode<Coord>>& total_nodes,
			DArray<Real>& total_value,
			DArray<Coord>& total_object,
			DArray<Coord>& total_normal,
			DArray<VoxelOctreeNode<Coord>>& level0_nodes,
			DArray<Real>& level0_value,
			DArray<Coord>& level0_object,
			DArray<Coord>& level0_normal,
			DArray<VoxelOctreeNode<Coord>>& levelT_nodes,
			DArray<Real>& levelT_value,
			DArray<Coord>& levelT_object,
			DArray<Coord>& levelT_normal,
			int level0_num,
			int grid_total_num);

		static void collectionGridsThree(
			DArray<VoxelOctreeNode<Coord>>& total_nodes,
			DArray<Real>& total_value,
			DArray<Coord>& total_object,
			DArray<Coord>& total_normal,
			DArray<VoxelOctreeNode<Coord>>& level0_nodes,
			DArray<Real>& level0_value,
			DArray<Coord>& level0_object,
			DArray<Coord>& level0_normal,
			DArray<VoxelOctreeNode<Coord>>& level1_nodes,
			DArray<Real>& level1_value,
			DArray<Coord>& level1_object,
			DArray<Coord>& level1_normal,
			DArray<VoxelOctreeNode<Coord>>& levelT_nodes,
			DArray<Real>& levelT_value,
			DArray<Coord>& levelT_object,
			DArray<Coord>& levelT_normal,
			int level0_num,
			int level1_num,
			int grid_total_num);

		static void collectionGridsFour(
			DArray<VoxelOctreeNode<Coord>>& total_nodes,
			DArray<Real>& total_value,
			DArray<Coord>& total_object,
			DArray<Coord>& total_normal,
			DArray<VoxelOctreeNode<Coord>>& level0_nodes,
			DArray<Real>& level0_value,
			DArray<Coord>& level0_object,
			DArray<Coord>& level0_normal,
			DArray<VoxelOctreeNode<Coord>>& level1_nodes,
			DArray<Real>& level1_value,
			DArray<Coord>& level1_object,
			DArray<Coord>& level1_normal,
			DArray<VoxelOctreeNode<Coord>>& level2_nodes,
			DArray<Real>& level2_value,
			DArray<Coord>& level2_object,
			DArray<Coord>& level2_normal,
			DArray<VoxelOctreeNode<Coord>>& levelT_nodes,
			DArray<Real>& levelT_value,
			DArray<Coord>& levelT_object,
			DArray<Coord>& levelT_normal,
			int level0_num,
			int level1_num,
			int level2_num,
			int grid_total_num);

		static void collectionGridsFive(
			DArray<VoxelOctreeNode<Coord>>& total_nodes,
			DArray<Real>& total_value,
			DArray<Coord>& total_object,
			DArray<Coord>& total_normal,
			DArray<VoxelOctreeNode<Coord>>& level0_nodes,
			DArray<Real>& level0_value,
			DArray<Coord>& level0_object,
			DArray<Coord>& level0_normal,
			DArray<VoxelOctreeNode<Coord>>& level1_nodes,
			DArray<Real>& level1_value,
			DArray<Coord>& level1_object,
			DArray<Coord>& level1_normal,
			DArray<VoxelOctreeNode<Coord>>& level2_nodes,
			DArray<Real>& level2_value,
			DArray<Coord>& level2_object,
			DArray<Coord>& level2_normal,
			DArray<VoxelOctreeNode<Coord>>& level3_nodes,
			DArray<Real>& level3_value,
			DArray<Coord>& level3_object,
			DArray<Coord>& level3_normal,
			DArray<VoxelOctreeNode<Coord>>& levelT_nodes,
			DArray<Real>& levelT_value,
			DArray<Coord>& levelT_object,
			DArray<Coord>& levelT_normal,
			int level0_num,
			int level1_num,
			int level2_num,
			int level3_num,
			int grid_total_num);

		static void VolumeHelper<TDataType>::finestLevelBoolean(
			DArray<VoxelOctreeNode<Coord>>& grid0,
			DArray<Real>& grid0_value,
			DArray<Coord>& grid0_object,
			DArray<Coord>& grid0_normal,
			DArray<VoxelOctreeNode<Coord>>& sdfOctreeNode_a,
			DArray<Real>& sdfValue_a,
			DArray<Coord>& object_a,
			DArray<Coord>& normal_a,
			DArray<VoxelOctreeNode<Coord>>& sdfOctreeNode_b,
			DArray<Real>& sdfValue_b,
			DArray<Coord>& object_b,
			DArray<Coord>& normal_b,
			std::shared_ptr<VoxelOctree<TDataType>> sdfOctree_a,
			std::shared_ptr<VoxelOctree<TDataType>> sdfOctree_b,
			int offset_ax,
			int offset_ay,
			int offset_az,
			int offset_bx,
			int offset_by,
			int offset_bz,
			int level0_a,
			int level0_b,
			int& m_level0,
			Coord m_origin,
			Real m_dx,
			int m_nx,
			int m_ny,
			int m_nz,
			int boolean);

		static void VolumeHelper<TDataType>::finestLevelReconstruction(
			DArray<PositionNode>& recon_node,
			DArray<Real>& recon_sdf,
			DArray<Coord>& recon_object,
			DArray<Coord>& recon_normal,
			DArray<VoxelOctreeNode<Coord>>& grid,
			DArray<Coord>& grid_object,
			DArray<Coord>& grid_normal,
			Coord m_origin,
			Real m_dx,
			int m_nx,
			int m_ny,
			int m_nz,
			Real m_dx_old,
			int level_0,
			int& level_0_recon);

		static void VolumeHelper<TDataType>::finestLevelReconstBoolean(
			DArray<VoxelOctreeNode<Coord>>& grid0,
			DArray<Real>& grid0_value,
			DArray<Coord>& grid0_object,
			DArray<Coord>& grid0_normal,
			DArray<PositionNode>& recon_node,
			DArray<Real>& recon_sdf,
			DArray<Coord>& recon_object,
			DArray<Coord>& recon_normal,
			DArray<VoxelOctreeNode<Coord>>& sdfOctreeNode_2,
			DArray<Real>& sdfValue_2,
			DArray<Coord>& object_2,
			DArray<Coord>& normal_2,
			std::shared_ptr<VoxelOctree<TDataType>> sdfOctree_a,
			std::shared_ptr<VoxelOctree<TDataType>> sdfOctree_b,
			int level0_1,
			int level0_2,
			int& m_level0,
			int offset_nx,
			int offset_ny,
			int offset_nz,
			Coord m_origin,
			Real m_dx,
			int m_nx,
			int m_ny,
			int m_nz,
			int boolean);

		static void VolumeHelper<TDataType>::updateBooleanSigned(
			DArray<Real>& leaf_value,
			DArray<int>& leaf_index,
			DArray<Real>& leaf_value_a,
			DArray<Real>& leaf_value_b,
			int boolean);
	};
}
