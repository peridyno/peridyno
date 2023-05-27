#include "VolumeOctreeBoolean.h"
#include "VoxelOctree.h"
#include "Algorithm/Reduction.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <thrust/sort.h>
#include <ctime>

namespace dyno
{
	IMPLEMENT_TCLASS(VolumeOctreeBoolean, TDataType)

	template<typename TDataType>
	VolumeOctreeBoolean<TDataType>::VolumeOctreeBoolean()
		: VolumeOctree<TDataType>()
	{
		auto connect = std::make_shared<FCallBackFunc>(
			[=]() {
				auto vol0 = this->getOctreeA();
				auto vol1 = this->getOctreeB();
				if (vol0 != nullptr && vol1 != nullptr)
				{
					vol0->setVisible(false);
					vol1->setVisible(false);
				}
			}
		);

		this->importOctreeA()->attach(connect);
		this->importOctreeB()->attach(connect);
	}

	template<typename TDataType>
	VolumeOctreeBoolean<TDataType>::~VolumeOctreeBoolean()
	{
	}

	template<typename TDataType>
	void VolumeOctreeBoolean<TDataType>::initParameter()
	{
		auto sdfOctree_a = this->getOctreeA()->stateSDFTopology()->getDataPtr();
		auto sdfOctree_b = this->getOctreeB()->stateSDFTopology()->getDataPtr();

		if (this->varMinDx()->getData() == true)
			m_dx = std::min(sdfOctree_a->getDx(), sdfOctree_b->getDx());
		else
			m_dx = std::max(sdfOctree_a->getDx(), sdfOctree_b->getDx());

		if ((sdfOctree_a->getDx()) == (sdfOctree_b->getDx()))
		{
			m_reconstructed_model = 0;
		}
		else
		{
			if (m_dx == (sdfOctree_a->getDx()))
				m_reconstructed_model = 2;
			else if (m_dx == (sdfOctree_b->getDx()))
				m_reconstructed_model = 1;
		}

		int level = this->varLevelNumber()->getData();
		int coef = std::floor(pow(float(2), int(level - 1)) + float(0.1));
		Coord origin_a = sdfOctree_a->getOrigin();
		Coord origin_b = sdfOctree_b->getOrigin();
		m_origin[0] = std::min(origin_a[0], origin_b[0]);
		m_origin[1] = std::min(origin_a[1], origin_b[1]);
		m_origin[2] = std::min(origin_a[2], origin_b[2]);
		Coord torigin;
		auto torigin_a = sdfOctree_a->getTopOrigin();
		auto torigin_b = sdfOctree_b->getTopOrigin();
		torigin[0] = std::max(torigin_a[0], torigin_b[0]);
		torigin[1] = std::max(torigin_a[1], torigin_b[1]);
		torigin[2] = std::max(torigin_a[2], torigin_b[2]);

		int nx_offset = std::floor(m_origin[0] / m_dx + 1e-5);
		int ny_offset = std::floor(m_origin[1] / m_dx + 1e-5);
		int nz_offset = std::floor(m_origin[2] / m_dx + 1e-5);
		if ((nx_offset % coef) < 0)
			nx_offset = nx_offset - (coef - abs(nx_offset % coef));
		else
			nx_offset = nx_offset - (nx_offset % coef);

		if ((ny_offset % coef) < 0)
			ny_offset = ny_offset - (coef - abs(ny_offset % coef));
		else
			ny_offset = ny_offset - (ny_offset % coef);

		if ((nz_offset % coef) < 0)
			nz_offset = nz_offset - (coef - abs(nz_offset % coef));
		else
			nz_offset = nz_offset - (nz_offset % coef);
		m_origin[0] = nx_offset * m_dx;
		m_origin[1] = ny_offset * m_dx;
		m_origin[2] = nz_offset * m_dx;


		int nx_offset1 = std::floor(torigin[0] / m_dx + 1e-5);
		int ny_offset1 = std::floor(torigin[1] / m_dx + 1e-5);
		int nz_offset1 = std::floor(torigin[2] / m_dx + 1e-5);
		if ((nx_offset1 % coef) < 0)
			nx_offset1 = nx_offset1 - (nx_offset1 % coef);
		else
			nx_offset1 = nx_offset1 + (coef - (nx_offset1 % coef));

		if ((ny_offset1 % coef) < 0)
			ny_offset1 = ny_offset1 - (ny_offset1 % coef);
		else
			ny_offset1 = ny_offset1 + (coef - (ny_offset1 % coef));

		if ((nz_offset1 % coef) < 0)
			nz_offset1 = nz_offset1 - (nz_offset1 % coef);
		else
			nz_offset1 = nz_offset1 + (coef - (nz_offset1 % coef));
		torigin[0] = nx_offset1 * m_dx;
		torigin[1] = ny_offset1 * m_dx;
		torigin[2] = nz_offset1 * m_dx;

		m_nx = std::ceil((torigin[0] - m_origin[0]) / m_dx - float(0.1));
		m_ny = std::ceil((torigin[1] - m_origin[1]) / m_dx - float(0.1));
		m_nz = std::ceil((torigin[2] - m_origin[2]) / m_dx - float(0.1));

		if (m_reconstructed_model == 0)
		{
			m_offset_ax = std::round((origin_a[0] - m_origin[0]) / m_dx - 1e-5);
			m_offset_ay = std::round((origin_a[1] - m_origin[1]) / m_dx - 1e-5);
			m_offset_az = std::round((origin_a[2] - m_origin[2]) / m_dx - 1e-5);

			m_offset_bx = std::round((origin_b[0] - m_origin[0]) / m_dx - 1e-5);
			m_offset_by = std::round((origin_b[1] - m_origin[1]) / m_dx - 1e-5);
			m_offset_bz = std::round((origin_b[2] - m_origin[2]) / m_dx - 1e-5);
		}
		else if (m_reconstructed_model == 1)
		{
			m_offset_bx = std::ceil((origin_b[0] - m_origin[0]) / m_dx - 1e-5);
			m_offset_by = std::ceil((origin_b[1] - m_origin[1]) / m_dx - 1e-5);
			m_offset_bz = std::ceil((origin_b[2] - m_origin[2]) / m_dx - 1e-5);
		}
		else if (m_reconstructed_model == 2)
		{
			m_offset_ax = std::ceil((origin_a[0] - m_origin[0]) / m_dx - 1e-5);
			m_offset_ay = std::ceil((origin_a[1] - m_origin[1]) / m_dx - 1e-5);
			m_offset_az = std::ceil((origin_a[2] - m_origin[2]) / m_dx - 1e-5);
		}

		printf("octree union: %d \n", m_reconstructed_model);
		//std::printf("The origin, dx, nx, ny, nz are: %f  %f  %f, %f, %d  %d  %d, %d %d %d, %d %d %d \n",
		//	m_origin[0], m_origin[1], m_origin[2], m_dx, m_nx, m_ny, m_nz,
		//	m_offset_ax, m_offset_ay, m_offset_az, m_offset_bx, m_offset_by, m_offset_bz);
	}
	
	template<typename TDataType>
	void VolumeOctreeBoolean<TDataType>::updateStates()
	{
		
//		getLeafsValue();
	}




	template<typename TDataType>
	void VolumeOctreeBoolean<TDataType>::updateSignOperation()
	{
		DArray<Coord> leaf_nodes;
		DArray<int> leaf_index;
		DArray<Real> leaf_value_a;
		DArray<Real> leaf_value_b;

		auto m_octree_a = this->getOctreeA();
		auto m_octree_b = this->getOctreeB();

		auto& octree_inter_value = this->stateSDFTopology()->getDataPtr()->getSdfValues();
		auto octree_inter = this->stateSDFTopology()->getDataPtr();

		DArray<Coord> leaf_normal_a,leaf_normal_b;
		octree_inter->getLeafs(leaf_nodes, leaf_index);
		m_octree_a->stateSDFTopology()->getDataPtr()->getSignDistance(leaf_nodes, leaf_value_a,leaf_normal_a);
		m_octree_b->stateSDFTopology()->getDataPtr()->getSignDistance(leaf_nodes, leaf_value_b,leaf_normal_b);
		leaf_normal_a.clear();
		leaf_normal_b.clear();

		auto booleanType = this->varBooleanType()->getDataPtr()->currentKey();
		VolumeHelper<TDataType>::updateBooleanSigned(
			octree_inter_value,
			leaf_index,
			leaf_value_a,
			leaf_value_b,
			booleanType);

		leaf_nodes.clear();
		leaf_index.clear();
		leaf_value_a.clear();
		leaf_value_b.clear();
	}

	template<typename TDataType>
	void VolumeOctreeBoolean<TDataType>::resetStates()
	{
		auto booleanType = this->varBooleanType()->getDataPtr()->currentKey();
		int level = this->varLevelNumber()->getData();
		initParameter();

		// initialize data
		auto m_octree_a = this->getOctreeA();
		auto m_octree_b = this->getOctreeB();

		auto sdfOctree_a = m_octree_a->stateSDFTopology()->getDataPtr();
		auto& sdfOctreeNode_a = sdfOctree_a->getVoxelOctree();
		auto& sdfValue_a = sdfOctree_a->getSdfValues();
		auto& object_a = m_octree_a->m_object;
		auto& normal_a = m_octree_a->m_normal;

		auto sdfOctree_b = m_octree_b->stateSDFTopology()->getDataPtr();
		auto& sdfOctreeNode_b = sdfOctree_b->getVoxelOctree();
		auto& sdfValue_b = sdfOctree_b->getSdfValues();
		auto& object_b = m_octree_b->m_object;
		auto& normal_b = m_octree_b->m_normal;

		int level0_a = sdfOctree_a->getLevel0();
		int level0_b = sdfOctree_b->getLevel0();

		DArray<VoxelOctreeNode<Coord>> grid0;
		DArray<Real> grid0_value;
		DArray<Coord> grid0_object;
		DArray<Coord> grid0_normal;
		if (m_reconstructed_model == 0)
		{
			VolumeHelper<TDataType>::finestLevelBoolean(
				grid0,
				grid0_value,
				grid0_object,
				grid0_normal,
				sdfOctreeNode_a,
				sdfValue_a,
				object_a,
				normal_a,
				sdfOctreeNode_b,
				sdfValue_b,
				object_b,
				normal_b,
				sdfOctree_a,
				sdfOctree_b,
				m_offset_ax,
				m_offset_ay,
				m_offset_az,
				m_offset_bx,
				m_offset_by,
				m_offset_bz,
				level0_a,
				level0_b,
				m_level0,
				m_origin,
				m_dx,
				m_nx,
				m_ny,
				m_nz,
				booleanType);
		}
		else if (m_reconstructed_model == 1)
		{
			DArray<PositionNode> recon_level0;
			DArray<Real> recon_sdf;
			DArray<Coord> recon_object, recon_normal;
			int level0_a_recon;
			VolumeHelper<TDataType>::finestLevelReconstruction(
				recon_level0,
				recon_sdf,
				recon_object,
				recon_normal,
				sdfOctreeNode_a,
				object_a,
				normal_a,
				m_origin,
				m_dx,
				m_nx,
				m_ny,
				m_nz,
				sdfOctree_a->getDx(),
				level0_a,
				level0_a_recon);

			VolumeHelper<TDataType>::finestLevelReconstBoolean(
				grid0,
				grid0_value,
				grid0_object,
				grid0_normal,
				recon_level0,
				recon_sdf,
				recon_object,
				recon_normal,
				sdfOctreeNode_b,
				sdfValue_b,
				object_b,
				normal_b,
				sdfOctree_a,
				sdfOctree_b,
				level0_a_recon,
				level0_b,
				m_level0,
				m_offset_bx,
				m_offset_by,
				m_offset_bz,
				m_origin,
				m_dx,
				m_nx,
				m_ny,
				m_nz,
				booleanType);

			recon_level0.clear();
			recon_sdf.clear();
			recon_object.clear();
			recon_normal.clear();
		}
		else if (m_reconstructed_model == 2)
		{
			DArray<PositionNode> recon_level0;
			DArray<Real> recon_sdf;
			DArray<Coord> recon_object, recon_normal;
			int level0_b_recon;
			VolumeHelper<TDataType>::finestLevelReconstruction(
				recon_level0,
				recon_sdf,
				recon_object,
				recon_normal,
				sdfOctreeNode_b,
				object_b,
				normal_b,
				m_origin,
				m_dx,
				m_nx,
				m_ny,
				m_nz,
				sdfOctree_b->getDx(),
				level0_b,
				level0_b_recon);

			if (booleanType == 2) booleanType = 3;
			VolumeHelper<TDataType>::finestLevelReconstBoolean(
				grid0,
				grid0_value,
				grid0_object,
				grid0_normal,
				recon_level0,
				recon_sdf,
				recon_object,
				recon_normal,
				sdfOctreeNode_a,
				sdfValue_a,
				object_a,
				normal_b,
				sdfOctree_b,
				sdfOctree_a,
				level0_b_recon,
				level0_a,
				m_level0,
				m_offset_ax,
				m_offset_ay,
				m_offset_az,
				m_origin,
				m_dx,
				m_nx,
				m_ny,
				m_nz,
				booleanType);

			recon_level0.clear();
			recon_sdf.clear();
			recon_object.clear();
			recon_normal.clear();
		}

		if (grid0.isEmpty())
		{
			return;
		}

		DArray<VoxelOctreeNode<Coord>> gridT;
		DArray<Real> gridT_value;
		DArray<Coord> gridT_object, gridT_normal;
		DArray<VoxelOctreeNode<Coord>> grid_total;
		DArray<Real> grid_total_value;
		int grid_total_num = 0;

		if (level == 2)
		{
			VolumeHelper<TDataType>::levelTop(
				gridT,
				gridT_value,
				gridT_object,
				gridT_normal,
				grid0,
				grid0_object,
				grid0_normal,
				m_origin,
				Level(1),
				m_nx,
				m_ny,
				m_nz,
				m_dx);

			grid_total_num = grid0.size() + gridT.size();
			grid_total.resize(grid_total_num);
			grid_total_value.resize(grid_total_num);
			m_object.resize(grid_total_num);
			m_normal.resize(grid_total_num);

			VolumeHelper<TDataType>::collectionGridsTwo(
				grid_total,
				grid_total_value,
				m_object,
				m_normal,
				grid0,
				grid0_value,
				grid0_object,
				grid0_normal,
				gridT,
				gridT_value,
				gridT_object,
				gridT_normal,
				grid0.size(),
				grid_total_num);
		}

		if (level == 3)
		{
			DArray<VoxelOctreeNode<Coord>> grid1;
			DArray<Real> grid1_value;
			DArray<Coord> grid1_object;
			DArray<Coord> grid1_normal;
			VolumeHelper<TDataType>::levelMiddle(
				grid1,
				grid1_value,
				grid1_object,
				grid1_normal,
				grid0,
				grid0_object,
				grid0_normal,
				m_origin,
				Level(1),
				m_nx,
				m_ny,
				m_nz,
				m_dx);

			VolumeHelper<TDataType>::levelTop(
				gridT,
				gridT_value,
				gridT_object,
				gridT_normal,
				grid1,
				grid1_object,
				grid1_normal,
				m_origin,
				Level(2),
				m_nx,
				m_ny,
				m_nz,
				m_dx);

			grid_total_num = grid0.size() + grid1.size() + gridT.size();
			grid_total.resize(grid_total_num);
			grid_total_value.resize(grid_total_num);
			m_object.resize(grid_total_num);
			m_normal.resize(grid_total_num);

			VolumeHelper<TDataType>::collectionGridsThree(
				grid_total,
				grid_total_value,
				m_object,
				m_normal,
				grid0,
				grid0_value,
				grid0_object,
				grid0_normal,
				grid1,
				grid1_value,
				grid1_object,
				grid1_normal,
				gridT,
				gridT_value,
				gridT_object,
				gridT_normal,
				grid0.size(),
				(grid0.size() + grid1.size()),
				grid_total_num);

			grid1.clear();
			grid1_value.clear();
			grid1_object.clear();
			grid1_normal.clear();
		}

		if (level == 4)
		{
			DArray<VoxelOctreeNode<Coord>> grid1;
			DArray<Real> grid1_value;
			DArray<Coord> grid1_object;
			DArray<Coord> grid1_normal;
			VolumeHelper<TDataType>::levelMiddle(
				grid1,
				grid1_value,
				grid1_object,
				grid1_normal,
				grid0,
				grid0_object,
				grid0_normal,
				m_origin,
				Level(1),
				m_nx,
				m_ny,
				m_nz,
				m_dx);

			DArray<VoxelOctreeNode<Coord>> grid2;
			DArray<Real> grid2_value;
			DArray<Coord> grid2_object;
			DArray<Coord> grid2_normal;
			VolumeHelper<TDataType>::levelMiddle(grid2,
				grid2_value,
				grid2_object,
				grid2_normal,
				grid1,
				grid1_object,
				grid1_normal,
				m_origin,
				Level(2),
				m_nx,
				m_ny,
				m_nz,
				m_dx);

			VolumeHelper<TDataType>::levelTop(
				gridT,
				gridT_value,
				gridT_object,
				gridT_normal,
				grid2,
				grid2_object,
				grid2_normal,
				m_origin,
				Level(3),
				m_nx,
				m_ny,
				m_nz,
				m_dx);

			grid_total_num = grid0.size() + grid1.size() + grid2.size() + gridT.size();
			grid_total.resize(grid_total_num);
			grid_total_value.resize(grid_total_num);
			m_object.resize(grid_total_num);
			m_normal.resize(grid_total_num);


			VolumeHelper<TDataType>::collectionGridsFour(
				grid_total,
				grid_total_value,
				m_object,
				m_normal,
				grid0,
				grid0_value,
				grid0_object,
				grid0_normal,
				grid1,
				grid1_value,
				grid1_object,
				grid1_normal,
				grid2,
				grid2_value,
				grid2_object,
				grid2_normal,
				gridT,
				gridT_value,
				gridT_object,
				gridT_normal,
				grid0.size(),
				(grid0.size() + grid1.size()),
				(grid0.size() + grid1.size() + grid2.size()),
				grid_total_num);

			grid1.clear();
			grid1_value.clear();
			grid1_object.clear();
			grid1_normal.clear();
			grid2.clear();
			grid2_value.clear();
			grid2_object.clear();
			grid2_normal.clear();
		}

		if (level == 5)
		{
			DArray<VoxelOctreeNode<Coord>> grid1;
			DArray<Real> grid1_value;
			DArray<Coord> grid1_object;
			DArray<Coord> grid1_normal;
			VolumeHelper<TDataType>::levelMiddle(
				grid1,
				grid1_value,
				grid1_object,
				grid1_normal,
				grid0,
				grid0_object,
				grid0_normal,
				m_origin,
				Level(1),
				m_nx,
				m_ny,
				m_nz,
				m_dx);

			DArray<VoxelOctreeNode<Coord>> grid2;
			DArray<Real> grid2_value;
			DArray<Coord> grid2_object;
			DArray<Coord> grid2_normal;
			VolumeHelper<TDataType>::levelMiddle(grid2,
				grid2_value,
				grid2_object,
				grid2_normal,
				grid1,
				grid1_object,
				grid1_normal,
				m_origin,
				Level(2),
				m_nx,
				m_ny,
				m_nz,
				m_dx);

			DArray<VoxelOctreeNode<Coord>> grid3;
			DArray<Real> grid3_value;
			DArray<Coord> grid3_object;
			DArray<Coord> grid3_normal;
			VolumeHelper<TDataType>::levelMiddle(grid3,
				grid3_value,
				grid3_object,
				grid3_normal,
				grid2,
				grid2_object,
				grid2_normal,
				m_origin,
				Level(3),
				m_nx,
				m_ny,
				m_nz,
				m_dx);

			VolumeHelper<TDataType>::levelTop(
				gridT,
				gridT_value,
				gridT_object,
				gridT_normal,
				grid3,
				grid3_object,
				grid3_normal,
				m_origin,
				Level(4),
				m_nx,
				m_ny,
				m_nz,
				m_dx);

			grid_total_num = grid0.size() + grid1.size() + grid2.size() + grid3.size() + gridT.size();
			grid_total.resize(grid_total_num);
			grid_total_value.resize(grid_total_num);
			m_object.resize(grid_total_num);
			m_normal.resize(grid_total_num);

			VolumeHelper<TDataType>::collectionGridsFive(
				grid_total,
				grid_total_value,
				m_object,
				m_normal,
				grid0,
				grid0_value,
				grid0_object,
				grid0_normal,
				grid1,
				grid1_value,
				grid1_object,
				grid1_normal,
				grid2,
				grid2_value,
				grid2_object,
				grid2_normal,
				grid3,
				grid3_value,
				grid3_object,
				grid3_normal,
				gridT,
				gridT_value,
				gridT_object,
				gridT_normal,
				grid0.size(),
				(grid0.size() + grid1.size()),
				(grid0.size() + grid1.size() + grid2.size()),
				(grid0.size() + grid1.size() + grid2.size() + grid3.size()),
				grid_total_num
			);

			grid1.clear();
			grid1_value.clear();
			grid1_object.clear();
			grid1_normal.clear();
			grid2.clear();
			grid2_value.clear();
			grid2_object.clear();
			grid2_normal.clear();
			grid3.clear();
			grid3_value.clear();
			grid3_object.clear();
			grid3_normal.clear();
		}

		auto sdf_oct = this->stateSDFTopology()->allocate();
		sdf_oct->setLevelNum(level);
		sdf_oct->setGrid(m_nx, m_ny, m_nz);
		sdf_oct->setVoxelOctree(grid_total);
		sdf_oct->setDx(m_dx);
		sdf_oct->setOrigin(m_origin);
		sdf_oct->setLevel0(m_level0);
		sdf_oct->updateNeighbors();
		sdf_oct->setSdfValues(grid_total_value);

		grid_total.clear();
		grid_total_value.clear();
		grid0.clear();
		grid0_value.clear();
		grid0_object.clear();
		grid0_normal.clear();
		gridT.clear();
		gridT_value.clear();
		gridT_object.clear();
		gridT_normal.clear();

		updateSignOperation();
	}

	template<typename TDataType>
	bool VolumeOctreeBoolean<TDataType>::validateInputs()
	{
		if (this->getOctreeA() == nullptr || this->getOctreeB() == nullptr) {
			return false;
		}

		return VolumeOctree<TDataType>::validateInputs();
	}

	DEFINE_CLASS(VolumeOctreeBoolean);
}