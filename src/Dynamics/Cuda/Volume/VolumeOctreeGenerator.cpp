#include "VolumeOctreeGenerator.h"
#include "VoxelOctree.h"
#include "Algorithm/Reduction.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <thrust/sort.h>
#include <ctime>

#include "VolumeHelper.h"

namespace dyno
{
	IMPLEMENT_TCLASS(VolumeOctreeGenerator, TDataType)

	template<typename TDataType>
	VolumeOctreeGenerator<TDataType>::VolumeOctreeGenerator()
		: VolumeOctree<TDataType>()
	{
		this->varSpacing()->setRange(0.001, 1.0);

		auto connect = std::make_shared<FCallBackFunc>(
			[=]() {
				auto input = this->inTriangleSet();
				Node* src = dynamic_cast<Node*>(input->getSource()->parent());
				if (src != nullptr)
				{
					src->setVisible(false);
				}
			}
		);

		this->inTriangleSet()->attach(connect);
	}

	template<typename TDataType>
	VolumeOctreeGenerator<TDataType>::~VolumeOctreeGenerator()
	{
	}

	template<typename TDataType>
	void VolumeOctreeGenerator<TDataType>::load(std::string filename, Coord rotate_value, Real scale_value, Coord translate_value)
	{
		std::shared_ptr<TriangleSet<TDataType>> triSet = std::make_shared<TriangleSet<TDataType>>();
		triSet->loadObjFile(filename);

		triSet->scale(scale_value);
	
		triSet->translate(translate_value);

		Quat<Real> q(rotate_value[0], rotate_value[1], rotate_value[2], 1.0f );
		triSet->rotate(q);

		this->inTriangleSet()->setDataPtr(triSet);

		this->inTriangleSet()->getDataPtr()->update();

	}

	template<typename TDataType>
	void VolumeOctreeGenerator<TDataType>::load(std::string filename)
	{
		std::shared_ptr<TriangleSet<TDataType>> triSet = std::make_shared<TriangleSet<TDataType>>();
		triSet->loadObjFile(filename);

		this->inTriangleSet()->setDataPtr(triSet);
	}

	template<typename TDataType>
	void VolumeOctreeGenerator<TDataType>::initParameter()
	{
		// initialize data
		auto triSet = this->inTriangleSet()->getDataPtr();
		auto& points_pos = triSet->getPoints();

		Reduction<Coord> reduce;
		Coord min_box = reduce.minimum(points_pos.begin(), points_pos.size());
		Coord max_box = reduce.maximum(points_pos.begin(), points_pos.size());

		uint padding = this->varPadding()->getData();
		Real dx = this->varSpacing()->getData();

		Coord unit(1, 1, 1);
		min_box -= padding * dx *unit;
		max_box += padding * dx *unit;

		int level = this->varLevelNumber()->getData();
		int coef = std::floor(pow(float(2), int(level - 1)) + float(0.1));

		int nx_offset = std::floor(min_box[0] / dx + 1e-5);
		int ny_offset = std::floor(min_box[1] / dx + 1e-5);
		int nz_offset = std::floor(min_box[2] / dx + 1e-5);
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
		min_box[0] = nx_offset * dx;
		min_box[1] = ny_offset * dx;
		min_box[2] = nz_offset * dx;

		int nx_offset1 = std::floor(max_box[0] / dx + 1e-5);
		int ny_offset1 = std::floor(max_box[1] / dx + 1e-5);
		int nz_offset1 = std::floor(max_box[2] / dx + 1e-5);
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
		max_box[0] = nx_offset1 * dx;
		max_box[1] = ny_offset1 * dx;
		max_box[2] = nz_offset1 * dx;

		m_nx = std::ceil((max_box[0] - min_box[0]) / dx - float(0.1));
		m_ny = std::ceil((max_box[1] - min_box[1]) / dx - float(0.1));
		m_nz = std::ceil((max_box[2] - min_box[2]) / dx - float(0.1));

		m_origin = min_box;

		//std::printf("The origin, dx, nx, ny, nz are: %f  %f  %f, %f, %d %d %d \n", m_origin[0], m_origin[1], m_origin[2], m_dx, m_nx, m_ny, m_nz);
	}

	template<typename TDataType>
	void VolumeOctreeGenerator<TDataType>::resetStates()
	{
		int level = this->varLevelNumber()->getData();

		Real dx = this->varSpacing()->getData();

		initParameter();

		//std::clock_t Time0 = clock();
		DArray<VoxelOctreeNode<Coord>> grid0;
		DArray<Real> grid0_value;
		DArray<Coord> grid0_object;
		DArray<Coord> grid0_normal;
		VolumeHelper<TDataType>::levelBottom(grid0,
			grid0_value,
			grid0_object,
			grid0_normal,
			this->inTriangleSet()->getDataPtr(),
			m_origin,
			m_nx,
			m_ny,
			m_nz,
			this->varAABBPadding()->getData(),
			m_level0,
			dx);
		//std::clock_t Time1 = clock();

		DArray<VoxelOctreeNode<Coord>> grid1;
		DArray<Real> grid1_value;
		DArray<Coord> grid1_object;
		DArray<Coord> grid1_normal;
		VolumeHelper<TDataType>::levelMiddle(grid1,
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
			dx);
		//std::clock_t Time2 = clock();

		DArray<VoxelOctreeNode<Coord>> gridT;
		DArray<Real> gridT_value;
		DArray<Coord> gridT_object, gridT_normal;
		DArray<VoxelOctreeNode<Coord>> grid_total;
		DArray<Real> grid_total_value;
		int grid_total_num = 0;
		if (level == 3)
		{
			VolumeHelper<TDataType>::levelTop(gridT,
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
				dx);

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

// 			cuExecute(grid_total_num,
// 				SO_CollectionGridsThree,
// 				grid_total,
// 				grid_total_value,
// 				m_object,
// 				m_normal,
// 				grid0,
// 				grid0_value,
// 				grid0_object,
// 				grid0_normal,
// 				grid1,
// 				grid1_value,
// 				grid1_object,
// 				grid1_normal,
// 				gridT,
// 				gridT_value,
// 				gridT_object,
// 				gridT_normal,
// 				grid0.size(),
// 				(grid0.size() + grid1.size()));
		}

		if (level == 4)
		{
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
				dx);

			VolumeHelper<TDataType>::levelTop(gridT,
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
				dx);

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
				grid_total_num
			);

// 			cuExecute(grid_total_num,
// 				SO_CollectionGridsFour,
// 				grid_total,
// 				grid_total_value,
// 				m_object,
// 				m_normal,
// 				grid0,
// 				grid0_value,
// 				grid0_object,
// 				grid0_normal,
// 				grid1,
// 				grid1_value,
// 				grid1_object,
// 				grid1_normal,
// 				grid2,
// 				grid2_value,
// 				grid2_object,
// 				grid2_normal,
// 				gridT,
// 				gridT_value,
// 				gridT_object,
// 				gridT_normal,
// 				grid0.size(),
// 				(grid0.size() + grid1.size()),
// 				(grid0.size() + grid1.size() + grid2.size()));

			grid2.clear();
			grid2_value.clear();
			grid2_object.clear();
			grid2_normal.clear();
		}

		if (level == 5)
		{
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
				dx);

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
				dx);

			VolumeHelper<TDataType>::levelTop(gridT,
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
				dx);

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
				grid_total_num);

// 			cuExecute(grid_total_num,
// 				SO_CollectionGridsFive,
// 				grid_total,
// 				grid_total_value,
// 				m_object,
// 				m_normal,
// 				grid0,
// 				grid0_value,
// 				grid0_object,
// 				grid0_normal,
// 				grid1,
// 				grid1_value,
// 				grid1_object,
// 				grid1_normal,
// 				grid2,
// 				grid2_value,
// 				grid2_object,
// 				grid2_normal,
// 				grid3,
// 				grid3_value,
// 				grid3_object,
// 				grid3_normal,
// 				gridT,
// 				gridT_value,
// 				gridT_object,
// 				gridT_normal,
// 				grid0.size(),
// 				(grid0.size() + grid1.size()),
// 				(grid0.size() + grid1.size() + grid2.size()),
// 				(grid0.size() + grid1.size() + grid2.size() + grid3.size()));

			grid2.clear();
			grid2_value.clear();
			grid2_object.clear();
			grid2_normal.clear();
			grid3.clear();
			grid3_value.clear();
			grid3_object.clear();
			grid3_normal.clear();
		}

		//std::clock_t Time5 = clock();

		auto sdf_oct = this->stateSDFTopology()->allocate();
		sdf_oct->setLevelNum(level);
		sdf_oct->setGrid(m_nx, m_ny, m_nz);
		sdf_oct->setVoxelOctree(grid_total);
		sdf_oct->setDx(dx);
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
		grid1.clear();
		grid1_value.clear();
		grid1_object.clear();
		grid1_normal.clear();
		gridT.clear();
		gridT_value.clear();
		gridT_object.clear();
		gridT_normal.clear();

		//std::clock_t Time6 = clock();
		//std::printf("Generation time is: %d %d %d clocks \n", int(Time5 - Time0), int(Time6 - Time5), int(Time6 - Time0));

		std::printf("Generated ASDF is: %d %d %d, %d %d %d, %f, %f %f %f \n",
			grid_total_num, level, m_level0, m_nx, m_ny, m_nz, dx, m_origin[0], m_origin[1], m_origin[2]);

//		getLeafsValue();
	}

	template<typename TDataType>
	void VolumeOctreeGenerator<TDataType>::updateTopology()
	{
		Coord move_vector = this->varForwardVector()->getData();

		if (move_vector.norm() > REAL_EPSILON)
		{
			auto triSet = this->inTriangleSet()->getDataPtr();

			triSet->translate(move_vector);
		}
	}

	template<typename TDataType>
	void VolumeOctreeGenerator<TDataType>::updateStates()
	{
		//this->reset();
	}

	DEFINE_CLASS(VolumeOctreeGenerator);
}