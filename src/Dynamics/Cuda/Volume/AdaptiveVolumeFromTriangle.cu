#include "AdaptiveVolumeFromTriangle.h"
#include <thrust/sort.h>
#include <ctime>
#include "Timer.h"
#include "Collision/Distance3D.h"


namespace dyno 
{
	IMPLEMENT_TCLASS(AdaptiveVolumeFromTriangle, TDataType)


	template<typename TDataType>
	void AdaptiveVolumeFromTriangle<TDataType>::load(std::string filename)
	{
		printf("Triangles to points load??? \n");

		std::shared_ptr<TriangleSet<TDataType>> triSet = std::make_shared<TriangleSet<TDataType>>();
		triSet->loadObjFile(filename);
		//triSet->scale(20);

		this->inTriangleSet()->setDataPtr(triSet);
	}

	template<typename TDataType>
	void AdaptiveVolumeFromTriangle<TDataType>::initParameter()
	{
		// initialize data
		auto triSet = this->inTriangleSet()->getDataPtr();

		if (triSet->isEmpty())
			return;

		auto& points_pos = triSet->getPoints();

		Reduction<Coord> reduce;
		Coord m_origin = reduce.minimum(points_pos.begin(), points_pos.size());
		Coord max_box = reduce.maximum(points_pos.begin(), points_pos.size());
		Coord center = (max_box + m_origin) / 2;

		//uint padding = this->varPadding()->getData();
		Real m_dx = this->varDx()->getData();
		int rs = std::max(max_box[0] - m_origin[0], std::max(max_box[1] - m_origin[1], max_box[2] - m_origin[2])) / m_dx;
		int rs0 = rs;

		//rs += 2 * padding;
		Level m_levelmax = std::ceil(std::log2(float(rs)));
		m_levelmax = max(m_levelmax, this->varMaxLevel()->getValue());

		Coord unit(1, 1, 1);
		rs = (1 << m_levelmax);
		m_origin = center - (m_dx*rs / 2)*unit;

		std::printf("The origin, dx, levelmax are: %f  %f  %f, %f, %d %d %d \n", m_origin[0], m_origin[1], m_origin[2], m_dx, m_levelmax, rs, rs0);

		this->stateOrigin()->setValue(m_origin);
		this->varMaxLevel()->setValue(m_levelmax);
		//this->outDx()->setValue(m_dx);
		//this->outOrigin()->setValue(min_box);
	}

	template <typename Real, typename Coord>
	GPU_FUNC int AVFT_ComputeGrid(
		int& nx_lo,
		int& ny_lo,
		int& nz_lo,
		int& nx_hi,
		int& ny_hi,
		int& nz_hi,
		Coord surf_p,
		Coord surf_q,
		Coord surf_r,
		Coord origin_,
		Real dx_,
		int re,
		int extend_band)
	{
		Coord fp = (surf_p - origin_) / dx_;
		Coord fq = (surf_q - origin_) / dx_;
		Coord fr = (surf_r - origin_) / dx_;

		extend_band = 1;
		nx_hi = clamp(int(maximum(fp[0], maximum(fq[0], fr[0]))) + extend_band + 1, 0, re - 1);
		ny_hi = clamp(int(maximum(fp[1], maximum(fq[1], fr[1]))) + extend_band + 1, 0, re - 1);
		nz_hi = clamp(int(maximum(fp[2], maximum(fq[2], fr[2]))) + extend_band + 1, 0, re - 1);

		nx_lo = clamp(int(minimum(fp[0], minimum(fq[0], fr[0]))) - extend_band, 0, re - 1);
		ny_lo = clamp(int(minimum(fp[1], minimum(fq[1], fr[1]))) - extend_band, 0, re - 1);
		nz_lo = clamp(int(minimum(fp[2], minimum(fq[2], fr[2]))) - extend_band, 0, re - 1);

		return (nz_hi - nz_lo + 1) * (ny_hi - ny_lo + 1) * (nx_hi - nx_lo + 1);
	}

	template <typename Real, typename Coord, typename Triangle>
	__global__ void AVFT_SurfaceCount(
		DArray<uint> counter,
		DArray<Triangle> surf_triangles,
		DArray<Coord> surf_points,
		Coord origin_,
		Real dx_,
		int resolution,
		int extend_band)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		int p = surf_triangles[tId][0];
		int q = surf_triangles[tId][1];
		int r = surf_triangles[tId][2];

		int nx_lo;
		int ny_lo;
		int nz_lo;

		int nx_hi;
		int ny_hi;
		int nz_hi;

		counter[tId] = AVFT_ComputeGrid(nx_lo, ny_lo, nz_lo, nx_hi, ny_hi, nz_hi, surf_points[p], surf_points[q], surf_points[r], origin_, dx_, resolution, extend_band);
	}

	template <typename Real, typename Coord, typename Triangle>
	__global__ void AVFT_SurfaceInit(
		DArray<OcKey> nodes,
		DArray<uint> counter,
		DArray<Triangle> surf_triangles,
		DArray<Coord> surf_points,
		Coord origin_,
		Real dx_,
		int resolution,
		int extend_band)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		int p = surf_triangles[tId][0];
		int q = surf_triangles[tId][1];
		int r = surf_triangles[tId][2];

		int nx_lo;
		int ny_lo;
		int nz_lo;

		int nx_hi;
		int ny_hi;
		int nz_hi;

		int num = AVFT_ComputeGrid(nx_lo, ny_lo, nz_lo, nx_hi, ny_hi, nz_hi, surf_points[p], surf_points[q], surf_points[r], origin_, dx_, resolution, extend_band);

		if (num > 0)
		{
			int acc_num1 = 0;
			for (int k = nz_lo; k <= nz_hi; k++) for (int j = ny_lo; j <= ny_hi; j++) for (int i = nx_lo; i <= nx_hi; i++)
			{
				OcKey index = CalculateMortonCode(i, j, k);
				nodes[counter[tId] + acc_num1] = index;
				acc_num1++;
			}
		}
	}

	__global__ void AVFT_CountNonRepeatedPosition(
		DArray<uint> counter,
		DArray<OcKey> nodes)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		if (tId == 0 || nodes[tId] != nodes[tId - 1])
			counter[tId] = 1;
	}

	__global__ void AVFT_FetchNonRepeatedPosition(
		DArray<OcKey> nodes,
		DArray<OcKey> all_nodes,
		DArray<uint> counter)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= counter.size()) return;

		if (tId == 0 || all_nodes[tId] != all_nodes[tId - 1])
			nodes[counter[tId]] = all_nodes[tId];
	}

	template<typename TDataType>
	void AdaptiveVolumeFromTriangle<TDataType>::computeSeeds()
	{
		GTimer timer;

		Level m_levelmax = this->varMaxLevel()->getData();
		Coord m_origin = this->stateOrigin()->getData();
		Real  m_dx = this->varDx()->getData();
		int padding = this->varAABBPadding()->getData();

		int resolution = (1 << m_levelmax);
		// initialize data
		auto triSet = this->inTriangleSet()->getDataPtr();
		auto& triangles = triSet->triangleIndices();
		auto& points = triSet->getPoints();
		printf("The model: %d  %d \n", points.size(), triangles.size());

		DArray<uint> data_count(triangles.size());
		data_count.reset();
		//count the number of active grids that overlap the triangles(with repeat)
		cuExecute(triangles.size(),
			AVFT_SurfaceCount,
			data_count,
			triangles,
			points,
			m_origin,
			m_dx,
			resolution,
			padding);
		Reduction<uint> reduce;
		int grid_num = reduce.accumulate(data_count.begin(), data_count.size());
		Scan<uint> scan;
		scan.exclusive(data_count.begin(), data_count.size());
		printf("Seed nodes %d \n", grid_num);

		DArray<OcKey> grid_buf(grid_num);
		//take the active grids
		cuExecute(triangles.size(),
			AVFT_SurfaceInit,
			grid_buf,
			data_count,
			triangles,
			points,
			m_origin,
			m_dx,
			resolution,
			padding);

		//timer.start();
		thrust::sort(thrust::device, grid_buf.begin(), grid_buf.begin() + grid_buf.size(), thrust::greater<OcKey>());
		//timer.stop();
		//printf("Sort1:  %d  %f\n", grid_buf.size(), timer.getElapsedTime());

		//compute the nodes(no reapet)
		data_count.resize(grid_num);
		data_count.reset();
		cuExecute(data_count.size(),
			AVFT_CountNonRepeatedPosition,
			data_count,
			grid_buf);
		int node_num = reduce.accumulate(data_count.begin(), data_count.size());
		scan.exclusive(data_count.begin(), data_count.size());
		printf("InitialLeafs  Leafs nodes: %d \n", node_num);

		auto& mSeeds = this->statepMorton()->getData();
		mSeeds.resize(node_num);
		cuExecute(data_count.size(),
			AVFT_FetchNonRepeatedPosition,
			mSeeds,
			grid_buf,
			data_count);
		grid_buf.clear();

		auto m_AGrid = this->stateAGridSet()->getDataPtr();
		m_AGrid->setOrigin(m_origin);
		m_AGrid->setDx(m_dx);
		m_AGrid->setLevelMax(m_levelmax);
		m_AGrid->setLevelNum(m_levelmax);
	}

	template<typename TDataType>
	void AdaptiveVolumeFromTriangle<TDataType>::resetStates()
	{
		this->stateAGridSet()->allocate();
		this->stateAGridSDF()->allocate();
		this->statepMorton()->allocate();

		initParameter();

		computeSeeds();

		this->animationPipeline()->update();
	}

	template<typename TDataType>
	void AdaptiveVolumeFromTriangle<TDataType>::updateStates()
	{
		Coord move_vector = this->varForwardVector()->getData();

		printf("The topology moved!   \n");
		auto triSet = this->inTriangleSet()->getDataPtr();
		triSet->translate(move_vector);
		computeSeeds();

		AdaptiveVolume<TDataType>::updateStates();
	}

	DEFINE_CLASS(AdaptiveVolumeFromTriangle);
}