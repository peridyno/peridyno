#include "VolumeUniformGenerator.h"
#include "Algorithm/Reduction.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <thrust/sort.h>
#include <ctime>

namespace dyno
{
	//DYN_FUNC static void kernel1(Real& val, Real val_x)
	//{
	//	if (std::abs(val_x) < 1)
	//		val = (1 - std::abs(val_x));
	//	else
	//		val = 0;
	//}
	//DYN_FUNC static void kernel2(Real& val, Real val_x)
	//{
	//	if (std::abs(val_x) < 0.5)
	//		val = (0.75 - (std::abs(val_x)*std::abs(val_x)));
	//	else if (std::abs(val_x) < 1.5)
	//		val = 0.5*(1.5 - (std::abs(val_x)))*(1.5 - (std::abs(val_x)));
	//	else
	//		val = 0;
	//	//std::printf("the val_x and val is: %f %f \n", val_x, val);
	//}
	//DYN_FUNC static void kernel3(Real& val, Real val_x)
	//{
	//	if (std::abs(val_x) < 1)
	//		val = (0.5*(std::abs(val_x)*std::abs(val_x)*std::abs(val_x)) - (std::abs(val_x)*std::abs(val_x)) + 2 / 3);
	//	else if (std::abs(val_x) < 2)
	//		val = (1 / 6)*(2 - (std::abs(val_x)))*(2 - (std::abs(val_x)))*(2 - (std::abs(val_x)));
	//	else
	//		val = 0;
	//}

	IMPLEMENT_TCLASS(VolumeUniformGenerator, TDataType)

	template<typename TDataType>
	VolumeUniformGenerator<TDataType>::VolumeUniformGenerator()
		: VolumeOctree<TDataType>()
	{
	}

	template<typename TDataType>
	VolumeUniformGenerator<TDataType>::~VolumeUniformGenerator()
	{
	}

	template<typename TDataType>
	void VolumeUniformGenerator<TDataType>::load(std::string filename)
	{
		std::shared_ptr<TriangleSet<TDataType>> triSet = std::make_shared<TriangleSet<TDataType>>();
		triSet->loadObjFile(filename);

		this->inTriangleSet()->setDataPtr(triSet);

		this->inTriangleSet()->getDataPtr()->update();
	}

	template<typename TDataType>
	void VolumeUniformGenerator<TDataType>::initParameter()
	{
		Reduction<Coord> reduce;

		// initialize data
		auto triSet = this->inTriangleSet()->getDataPtr();
		auto& points_pos = triSet->getPoints();

		Coord min_box = reduce.minimum(points_pos.begin(), points_pos.size());
		Coord max_box = reduce.maximum(points_pos.begin(), points_pos.size());
		std::printf("the begin origin is: %f  %f  %f %f %f %f\n", min_box[0], min_box[1], min_box[2], max_box[0], max_box[1], max_box[2]);

		uint padding = this->varPadding()->getData();
		m_dx = this->varSpacing()->getData();

		Coord unit(1, 1, 1);
		min_box -= padding * m_dx*unit;
		max_box += padding * m_dx*unit;

		m_origin = min_box;

		m_nx = std::ceil((max_box[0] - m_origin[0]) / m_dx);
		m_ny = std::ceil((max_box[1] - m_origin[1]) / m_dx);
		m_nz = std::ceil((max_box[2] - m_origin[2]) / m_dx);

		this->outUniformOrigin()->setValue(m_origin);
		this->outUnx()->setValue((uint)m_nx);
		this->outUny()->setValue((uint)m_ny);
		this->outUnz()->setValue((uint)m_nz);

		std::printf("the origin is: %f  %f  %f %f  %f  %f  %f  %f  %f\n", m_origin[0], m_origin[1], m_origin[2],(max_box[0]-m_origin[0]),(max_box[1]-m_origin[1]),(max_box[2]-m_origin[2]), (max_box[0] - m_origin[0]) / m_dx, (max_box[1] - m_origin[1]) / m_dx, (max_box[2] - m_origin[2]) / m_dx);
		std::printf("the padding and dx is: %d  %f \n", padding, m_dx);
		std::printf("the dx, dy, dz is: %d  %d  %d \n", m_nx, m_ny, m_nz);
	}
	
	template <typename Real, typename Coord>
	__global__ void SO_GetSignDistance(
		DArray<Real> point_sdf,
		DArray<Coord> point_pos,
		DArray<Real> uniform_value,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= point_pos.size()) return;

		Coord poi = point_pos[tId] - origin_;
		int nx_pos = clamp(int(std::floor(poi[0] / dx_)), 0, nx_ - 1);
		int ny_pos = clamp(int(std::floor(poi[1] / dx_)), 0, ny_ - 1);
		int nz_pos = clamp(int(std::floor(poi[2] / dx_)), 0, nz_ - 1);
		//std::printf("tId: %d  get sign distance ijk %d, %d, %d  \n", tId, nx_pos, ny_pos, nz_pos);

		int index = nz_pos * nx_*ny_ + ny_pos * nx_ + nx_pos;
		point_sdf[tId] = uniform_value[index];
	}

	template<typename TDataType>
	void VolumeUniformGenerator<TDataType>::getSignDistance(DArray<Coord> point_pos, DArray<Real>& point_sdf)
	{
		point_sdf.resize(point_pos.size());
		//std::printf("getSignDistance : the number of leafs is: %d \n", point_sdf.size());

		cuExecute(point_pos.size(),
			SO_GetSignDistance,
			point_sdf,
			point_pos,
			this->stateSDFTopology()->getDataPtr()->getSdfValues(),
			m_origin,
			m_dx,
			m_nx,
			m_ny,
			m_nz);
	}

	template <typename Real, typename Coord>
	GPU_FUNC int SO_ComputeGrid(
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
		int nx_,
		int ny_,
		int nz_,
		int extend_band)
	{
		Coord fp = (surf_p - origin_) / dx_;
		Coord fq = (surf_q - origin_) / dx_;
		Coord fr = (surf_r - origin_) / dx_;

		nx_hi = clamp(int(maximum(fp[0], maximum(fq[0], fr[0]))) + extend_band, 0, nx_ - 1);
		ny_hi = clamp(int(maximum(fp[1], maximum(fq[1], fr[1]))) + extend_band, 0, ny_ - 1);
		nz_hi = clamp(int(maximum(fp[2], maximum(fq[2], fr[2]))) + extend_band, 0, nz_ - 1);

		nx_lo = clamp(int(minimum(fp[0], minimum(fq[0], fr[0]))) - extend_band, 0, nx_ - 1);
		ny_lo = clamp(int(minimum(fp[1], minimum(fq[1], fr[1]))) - extend_band, 0, ny_ - 1);
		nz_lo = clamp(int(minimum(fp[2], minimum(fq[2], fr[2]))) - extend_band, 0, nz_ - 1);

		if ((nx_hi % 2) != 1) nx_hi++;
		if ((ny_hi % 2) != 1) ny_hi++;
		if ((nz_hi % 2) != 1) nz_hi++;
		if ((nx_lo % 2) != 0) nx_lo--;
		if ((ny_lo % 2) != 0) ny_lo--;
		if ((nz_lo % 2) != 0) nz_lo--;

		return (nz_hi - nz_lo + 1) * (ny_hi - ny_lo + 1) * (nx_hi - nx_lo + 1);
	}

	template <typename Real, typename Coord, typename Triangle>
	__global__ void SO_SurfaceCount(
		DArray<int> counter,
		DArray<Triangle> surf_triangles,
		DArray<Coord> surf_points,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_,
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

		counter[tId] = SO_ComputeGrid(nx_lo, ny_lo, nz_lo, nx_hi, ny_hi, nz_hi, surf_points[p], surf_points[q], surf_points[r], origin_, dx_, nx_, ny_, nz_, extend_band);
		//std::printf("this is1: %d \t the active nodes` num of this surface is %d \n", tId, counter[tId]);
	}


	template <typename Real, typename Coord, typename Triangle>
	__global__ void SO_SurfaceInit(
		DArray<UniformNode> nodes,
		DArray<int> counter,
		DArray<Triangle> surf_triangles,
		DArray<Coord> surf_points,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_,
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

		int num = SO_ComputeGrid(nx_lo, ny_lo, nz_lo, nx_hi, ny_hi, nz_hi, surf_points[p], surf_points[q], surf_points[r], origin_, dx_, nx_, ny_, nz_, extend_band);
		//std::printf("this is2: %d \t the active nodes` num of this surface is %d; counter[tId]: %d; cube: %d %d %d; %d %d %d; \n", tId, num, counter[tId],nx_lo,ny_lo,nz_lo,nx_hi,ny_hi,nz_hi);

		if (num > 0)
		{
			int acc_num = 0;

			for (int k = nz_lo; k <= nz_hi; k++) {
				for (int j = ny_lo; j <= ny_hi; j++) {
					for (int i = nx_lo; i <= nx_hi; i++)
					{
						int index = k * nx_*ny_ + j * nx_ + i;
						nodes[counter[tId] + acc_num].set_value(tId,index);

						//std::printf("tId1: %d %d %d %d\n", tId, index, nodes[counter[tId] + acc_num].position_index, nodes[counter[tId] + acc_num].surface_index);
						
						acc_num++;
					}
				}
			}
		}
		//std::printf("tId: %d; counter[tId]: %d; the surface is: %d, %d, %d; the key is: %lld; the distance is %f \n", tId, counter[tId], p, q, r, nodes[counter[tId]].key(), nodes_value[counter[tId]]);
	}

	template <typename Real, typename Coord, typename Triangle>
	__global__ void SO_GridInit(
		DArray<Real> nodes_value,
		DArray<int> nodes_surface,
		DArray<int> nodes_count,
		DArray<UniformNode> nodes,
		DArray<Triangle> surf_triangles,
		DArray<Coord> surf_points,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= nodes.size()) return;

		//std::printf("tId2: %d %d %d\n", tId, nodes[tId].position_index, nodes[tId].surface_index);

		if ((tId == 0 || nodes[tId].position_index != nodes[tId - 1].position_index))
		{
			int index = nodes[tId].position_index;
			int gnz = (int)(index / (nx_ * ny_));
			int gny = (int)((index % (nx_ * ny_)) / nx_);
			int gnx = (int)((index % (nx_ * ny_)) % nx_);
			Coord point_pos(origin_[0] + (gnx + 0.5)*dx_, origin_[1] + (gny + 0.5)*dx_, origin_[2] + (gnz + 0.5)*dx_);
			TPoint3D<Real> pos(point_pos);

			int g_surf = nodes[tId].surface_index;
			int p = surf_triangles[g_surf][0];
			int q = surf_triangles[g_surf][1];
			int r = surf_triangles[g_surf][2];
			TTriangle3D<Real> p_tri(surf_points[p], surf_points[q], surf_points[r]);

			Real node_value = pos.distance(p_tri);
			int node_surface = g_surf;
			//std::printf("tId3: %d %d %d %d %d %d %f\n", tId, index, gnx, gny, gnz, node_surface, node_value);

			int rep_num = 1;
			while (((tId + rep_num) < nodes.size()) && (nodes[tId].position_index == nodes[tId + rep_num].position_index))
			{
				int g_surf_i = nodes[tId + rep_num].surface_index;
				int p_i = surf_triangles[g_surf_i][0];
				int q_i = surf_triangles[g_surf_i][1];
				int r_i = surf_triangles[g_surf_i][2];
				TTriangle3D<Real> p_tri_i(surf_points[p_i], surf_points[q_i], surf_points[r_i]);

				Real node_value_i = pos.distance(p_tri_i);
				//std::printf("tId4: %d %d %d %d %d %d %f %f\n", tId, index, gnx, gny, gnz, g_surf_i, node_value_i, node_value);

				if (std::abs(node_value_i) < abs(node_value))
				{
					node_value = node_value_i;
					node_surface = g_surf_i;
				}
				rep_num++;
			}

			nodes_value[index] = node_value;
			nodes_surface[index] = node_surface;
			nodes_count[index] = 1;
			//std::printf("tId6: %d %d %d %d %d %d %f\n", tId, index, gnx, gny, gnz, node_surface, node_value);
		}
	}

	template <typename Real, typename Coord, typename Triangle>
	DYN_FUNC void SO_ComputeGridWithNeighbor(
		bool& update_id,
		Real& grid_value,
		int& grid_surface,
		int neighbor_surface,
		DArray<Triangle>& surf_triangles,
		DArray<Coord>& surf_points,
		Coord grid_pos)
	{
		int p = surf_triangles[neighbor_surface][0];
		int q = surf_triangles[neighbor_surface][1];
		int r = surf_triangles[neighbor_surface][2];
		TTriangle3D<Real> p_tri(surf_points[p], surf_points[q], surf_points[r]);

		TPoint3D<Real> g_pos(grid_pos);
		Real dist = g_pos.distance(p_tri);
		if (std::abs(dist) < std::abs(grid_value))
		{
			grid_value = dist;
			grid_surface = neighbor_surface;
			update_id = true;
		}
	}


	template <typename Real, typename Coord, typename Triangle>
	__global__ void SO_FIMComputeGrids(
		DArray<Real> nodes_value,
		DArray<int> nodes_surf,
		DArray<int> nodes_count,
		DArray<int> nodes_count_temp,
		DArray<Triangle> surf_triangles,
		DArray<Coord> surf_points,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= nodes_count.size()) return;

		int gnz = (int)(tId / (nx_ * ny_));
		int gny = (int)((tId % (nx_ * ny_)) / nx_);
		int gnx = (int)((tId % (nx_ * ny_)) % nx_);

		Coord point_pos(origin_[0] + (gnx + 0.5)*dx_, origin_[1] + (gny + 0.5)*dx_, origin_[2] + (gnz + 0.5)*dx_);

		bool update = false;
		int surf_index;
		Real value_index = std::numeric_limits<Real>::max();
		if (nodes_count_temp[tId] == 1)
		{
			value_index = nodes_value[tId];
			surf_index = nodes_surf[tId];
		}
		if (gnx > 0)
			if (nodes_count_temp[tId - 1] == 1)
			{
				SO_ComputeGridWithNeighbor(update, value_index, surf_index, nodes_surf[tId - 1], surf_triangles, surf_points, point_pos);
				//if (tId == 6947)
				//	std::printf("x-1FIM the incorrect node is: %d; %d %f\n", tId, surf_index, value_index);
			}
		if (gnx < (nx_ - 1))
			if (nodes_count_temp[tId + 1] == 1)
			{
				SO_ComputeGridWithNeighbor(update, value_index, surf_index, nodes_surf[tId + 1], surf_triangles, surf_points, point_pos);
				//if (tId == 6947)
				//	std::printf("x+1FIM the incorrect node is: %d; %d %f\n", tId, surf_index, value_index);
			}
		if (gny > 0)
			if (nodes_count_temp[tId - nx_] == 1)
			{
				SO_ComputeGridWithNeighbor(update, value_index, surf_index, nodes_surf[tId - nx_], surf_triangles, surf_points, point_pos);
				//if (tId == 6947)
				//	std::printf("y-1FIM the incorrect node is: %d; %d %f\n", tId, surf_index, value_index);
			}
		if (gny < (ny_ - 1))
			if (nodes_count_temp[tId + nx_] == 1)
			{
				SO_ComputeGridWithNeighbor(update, value_index, surf_index, nodes_surf[tId + nx_], surf_triangles, surf_points, point_pos);
				//if (tId == 6947)
				//	std::printf("y+1FIM the incorrect node is: %d; %d %f\n", tId, surf_index, value_index);
			}
		if (gnz > 0)
			if (nodes_count_temp[tId - nx_ * ny_] == 1)
			{
				SO_ComputeGridWithNeighbor(update, value_index, surf_index, nodes_surf[tId - nx_ * ny_], surf_triangles, surf_points, point_pos);
				//if (tId == 6947)
				//	std::printf("z-1FIM the incorrect node is: %d; %d %f\n", tId, surf_index, value_index);
			}
		if (gnz < (nz_ - 1))
			if (nodes_count_temp[tId + nx_ * ny_] == 1)
			{
				SO_ComputeGridWithNeighbor(update, value_index, surf_index, nodes_surf[tId + nx_ * ny_], surf_triangles, surf_points, point_pos);
				//if (tId == 6947)
				//	std::printf("z+1FIM the incorrect node is: %d; %d %f\n", tId, surf_index, value_index);
			}

		if (update)
		{
			//if (value_index < 0 &&node_ind_temp[tId]==0)
			//	std::printf("FIM the incorrect node 1 is: %d; %d %d %d %f %f %f; %d %f %f %f %f %f %f %f %f %f; %f\n", tId, gnx, gny, gnz, gpos[0], gpos[1], gpos[2], surf_index, surf_points[p][0], surf_points[p][1], surf_points[p][2], surf_points[q][0], surf_points[q][1], surf_points[q][2], surf_points[r][0], surf_points[r][1], surf_points[r][2], value_index);

			nodes_value[tId] = value_index;
			nodes_surf[tId] = surf_index;
			nodes_count[tId] = 1;

			//if (tId==6947)
			//	std::printf("2FIM the incorrect node is: %d; %d %d %d %f %f %f; %d %f %f %f %f %f %f %f %f %f; %f \n", tId, gnx, gny, gnz, gpos[0], gpos[1], gpos[2], surf_index, surf_points[p][0], surf_points[p][1], surf_points[p][2], surf_points[q][0], surf_points[q][1], surf_points[q][2], surf_points[r][0], surf_points[r][1], surf_points[r][2], value_index);
			//if (value_index < 0)
			//	std::printf("FIM the incorrect node 2 is: %d; %d %d %d %f %f %f; %d %f %f %f %f %f %f %f %f %f; %f\n", tId, gnx, gny, gnz, gpos[0], gpos[1], gpos[2], surf_index, surf_points[p][0], surf_points[p][1], surf_points[p][2], surf_points[q][0], surf_points[q][1], surf_points[q][2], surf_points[r][0], surf_points[r][1], surf_points[r][2], value_index);
		}
	}

	template<typename TDataType>
	void VolumeUniformGenerator<TDataType>::resetStates()
	{
		std::clock_t startTime = clock();

		initParameter();

		// initialize data
		auto triSet = this->inTriangleSet()->getDataPtr();
		auto& triangles = triSet->getTriangles();
		auto& points = triSet->getPoints();

		int nxyz = m_nx * m_ny*m_nz;
		DArray<Real> grids_value;
		DArray<int> grids_surface;
		grids_value.resize(nxyz);
		grids_surface.resize(nxyz);

		DArray<int> data_count;
		data_count.resize(triangles.size());
		std::printf("the num of surface is: %d \n", data_count.size());
		
		//数一下level_0中active grid的数目
		cuExecute(data_count.size(),
			SO_SurfaceCount,
			data_count,
			triangles,
			points,
			m_origin,
			m_dx,
			m_nx,
			m_ny,
			m_nz,
			(int)0);

		int grid_num = thrust::reduce(thrust::device, data_count.begin(), data_count.begin() + data_count.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, data_count.begin(), data_count.begin() + data_count.size(), data_count.begin());
		std::printf("the active grid(with repeat) of level 0 is: %d \n", grid_num);
		//std::clock_t Time1 = clock();
		//std::printf("the time of counting active grid of level 0 is: %d clock \n", Time1-startTime);

		DArray<UniformNode> grids_buf1;
		grids_buf1.resize(grid_num);
		//将active grid取出
		cuExecute(data_count.size(),
			SO_SurfaceInit,
			grids_buf1,
			data_count,
			triangles,
			points,
			m_origin,
			m_dx,
			m_nx,
			m_ny,
			m_nz,
			(int)0);
		//std::clock_t Time2 = clock();
		//std::printf("the time of initializing active grid of level 0 is: %d clock \n", Time2 - Time1);

		thrust::sort(thrust::device, grids_buf1.begin(), grids_buf1.begin() + grids_buf1.size(), NodeCmp());

		//std::clock_t Time3 = clock();
		//std::printf("the time of sorting active grid of level 0 is: %d clock \n", Time3 - Time2);

		std::printf("total grids: %d \n", nxyz);
		data_count.resize(nxyz);
		data_count.reset();
		//将active grid存到对应的网格中
		cuExecute(grids_buf1.size(),
			SO_GridInit,
			grids_value,
			grids_surface,
			data_count,
			grids_buf1,
			triangles,
			points,
			m_origin,
			m_dx,
			m_nx,
			m_ny,
			m_nz);
		grids_buf1.clear();
		std::clock_t Time4 = clock();
		//std::printf("the time of initializing grids is: %d clock \n", Time4 - Time3);

		Reduction<int> reduce;
		DArray<int> data_count_temp;
		int total_num = reduce.accumulate(data_count.begin(), data_count.size());
		while (total_num < (nxyz))
		{
			//std::printf("start FIM iteration,the total num now is: %d \n", total_num);
			data_count_temp.assign(data_count);
			//topside节点中的值FIM更新
			cuExecute(nxyz,
				SO_FIMComputeGrids,
				grids_value,
				grids_surface,
				data_count,
				data_count_temp,
				triangles,
				points,
				m_origin,
				m_dx,
				m_nx,
				m_ny,
				m_nz);

			total_num = reduce.accumulate(data_count.begin(), data_count.size());
		}

		data_count.clear();
		data_count_temp.clear();
		std::clock_t Time5 = clock();
		std::printf("the time of FIM iteration is: %d clock \n", Time5 - Time4);

		auto& sdf_val = this->stateSDFTopology()->getDataPtr()->getSdfValues();
		sdf_val.assign(grids_value);

		std::clock_t endTime = clock();
		std::printf("the uniform grids construction time is: %d clocks\n", int(endTime - startTime));
	}

	template<typename TDataType>
	void VolumeUniformGenerator<TDataType>::updateStates()
	{
		this->reset();
	}

		DEFINE_CLASS(VolumeUniformGenerator);
}