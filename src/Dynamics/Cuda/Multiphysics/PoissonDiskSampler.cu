#include "GLPointVisualModule.h"
#include "PoissonDiskSampler.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace dyno
{
	template <typename TDataType>
	PoissonDiskSampler<TDataType>::PoissonDiskSampler()
		: SdfSampler<TDataType>()
	{
		this->varSpacing()->setRange(0.004, 1.0);
	};

	template <typename TDataType>
	PoissonDiskSampler<TDataType>::~PoissonDiskSampler()
	{

	};

	template <typename TDataType>
	int PoissonDiskSampler<TDataType>::pointNumberRecommend()
	{
		int num = 0;
		Coord box = area_b - area_a;
		Real r = this->varSpacing()->getData();
		return
			(area_b[0] - area_a[0]) * (area_b[1] - area_a[1]) * (area_b[2] - area_a[2]) / (r * r * r);
	};

	template <typename TDataType>
	void PoissonDiskSampler<TDataType>::ConstructGrid()
	{
		auto r = this->varSpacing()->getData();

		m_grid_dx = r / sqrt(3);

		if (m_grid_dx < EPSILON)
		{
			std::cout << " The smapling distance in Poisson disk sampling module can not be ZERO!!!! " << std::endl;
			exit(0);
		}

		nx = abs(area_b[0] - area_a[0]) / m_grid_dx;
		ny = abs(area_b[1] - area_a[1]) / m_grid_dx;
		nz = abs(area_b[2] - area_a[2]) / m_grid_dx;

		unsigned int gnum = nx * ny * nz;

		m_grid.resize(gnum);
		for (int i = 0; i < gnum; i++)	m_grid[i] = -1;
	}


	template <typename TDataType>
	GridIndex PoissonDiskSampler<TDataType>::searchGrid(Coord point)
	{
		GridIndex index;
		index.i = (int)((point[0] - area_a[0]) / m_grid_dx);
		index.j = (int)((point[1] - area_a[1]) / m_grid_dx);
		index.k = (int)((point[2] - area_a[2]) / m_grid_dx);

		return index;
	};

	template <typename TDataType>
	int PoissonDiskSampler<TDataType>::indexTransform(int i, int j, int k)
	{
		return  i + j * nx + k * nx * ny;
	};


	template <typename TDataType>
	bool PoissonDiskSampler<TDataType>::collisionJudge2D(Coord point)
	{
		bool flag = false;
		auto r = this->varSpacing()->getData();
		GridIndex d_index;
		d_index = searchGrid(point);
		for (int i = -2; i < 3; i++)
			for (int j = -2; j < 3; j++)
			{
				int mi = d_index.i + i;
				int mj = d_index.j + j;
				if ((mi > 0) && (mj > 0) && (mi < nx) && (mj < ny))
				{
					int index = indexTransform(mi, mj, d_index.k);
					if (m_grid[index] != -1)
					{
						Coord d = (m_points[m_grid[index]] - point);
						if (sqrt(d[0] * d[0] + d[1] * d[1]) - r < EPSILON)
						{
							flag = true;
						}
					}
				}
			}
		return flag;
	};

	template <typename TDataType>
	bool PoissonDiskSampler<TDataType>::collisionJudge(Coord point)
	{

		bool flag = false;
		auto r = this->varSpacing()->getData();
		GridIndex d_index;
		d_index = searchGrid(point);
		for (int i = -2; i < 3; i++)
			for (int j = -2; j < 3; j++)
				for (int k = -2; k < 3; k++)
				{
					int mi = d_index.i + i;
					int mj = d_index.j + j;
					int mk = d_index.k + k;
					if ((mi > 0) && (mj > 0) && (mk > 0) && (mi < nx) && (mj < ny) && (mk < nz))
					{
						int index = indexTransform(mi, mj, mk);
						if (m_grid[index] != -1)
						{
							Coord d = (m_points[m_grid[index]] - point);
							if (sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) - r < EPSILON)
							{
								flag = true;
							}
						}
					}
				}
		return flag;
	};


	//template<typename TDataType>
	//std::shared_ptr<DistanceField3D<TDataType>> PoissonDiskSampler<TDataType>::loadSdf()
	//{
	//	auto varfilename = this->varSdfFileName()->getData();
	//	auto filename = varfilename.string();
	//	std::shared_ptr<DistanceField3D<TDataType>> sdf = std::make_shared<DistanceField3D<TDataType>>();
	//	if (filename.size() > 0)
	//	{
	//		if (filename.size() < 5 || filename.substr(filename.size() - 4) != std::string(".sdf")) {
	//			std::cerr << "Error: Expected OBJ file with filename of the form <name>.obj.\n";
	//			exit(-1);
	//		}
	//		std::ifstream infile(filename);
	//		if (!infile) {
	//			std::cerr << "Failed to open. Terminating.\n";
	//			exit(-1);
	//		}
	//		//sdf->loadSDF(filename, false);
	//		area_a = sdf->lowerBound();
	//		area_b = sdf->upperBound();
	//		m_h = sdf->getGridSpacing();
	//		m_left = sdf->lowerBound();
	//		std::cout << "SDF is loaded: " << area_a << ", " << area_b << std::endl;
	//		return sdf;
	//	}
	//	else
	//	{
	//		return sdf;
	//	}
	//};

	template<typename Real, typename Coord, typename TDataType>
	__global__ void PDS_PosDetect
	(
		DArray<Coord> posArr,
		DistanceField3D<TDataType> df,
		DArray<Real> dist
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;
		Coord pos = posArr[pId];
		Real temp_dist;
		Coord normal;
		df.getDistance(pos, temp_dist, normal);
		dist[pId] = temp_dist;

	}



	template <typename TDataType>
	typename TDataType::Real PoissonDiskSampler<TDataType>::lerp(Real a, Real b, Real alpha)
	{
		return (1.0f - alpha) * a + alpha * b;
	};


	template <typename TDataType>
	typename TDataType::Real PoissonDiskSampler<TDataType>::getDistanceFromSDF(const Coord& p,
		Coord& normal)
	{

		Real d = 0.0f;

		Real m_h = m_inputSDF->getGridSpacing();

		Coord m_left = m_inputSDF->lowerBound();

		Coord fp = (p - m_left) * Coord(1.0 / m_h, 1.0 / m_h, 1.0 / m_h);
		const int i = (int)floor(fp[0]);
		const int j = (int)floor(fp[1]);
		const int k = (int)floor(fp[2]);

		bool m_bInverted = false;

		if (i < 0 || i >= host_dist.nx() - 1 || j < 0 || j >= host_dist.ny() - 1 || k < 0 || k >= host_dist.nz() - 1) {
			if (m_bInverted) d = -100000.0f;
			else d = 100000.0f;
			normal = Coord(0);
			return d;
		}

		Coord ip = Coord(i, j, k);

		Coord alphav = fp - ip;
		Real alpha = alphav[0];
		Real beta = alphav[1];
		Real gamma = alphav[2];


		Real d000 = host_dist(i, j, k);
		Real d100 = host_dist(i + 1, j, k);
		Real d010 = host_dist(i, j + 1, k);
		Real d110 = host_dist(i + 1, j + 1, k);
		Real d001 = host_dist(i, j, k + 1);
		Real d101 = host_dist(i + 1, j, k + 1);
		Real d011 = host_dist(i, j + 1, k + 1);
		Real d111 = host_dist(i + 1, j + 1, k + 1);

		Real dx00 = lerp(d000, d100, alpha);
		Real dx10 = lerp(d010, d110, alpha);
		Real dxy0 = lerp(dx00, dx10, beta);

		Real dx01 = lerp(d001, d101, alpha);
		Real dx11 = lerp(d011, d111, alpha);
		Real dxy1 = lerp(dx01, dx11, beta);

		Real d0y0 = lerp(d000, d010, beta);
		Real d0y1 = lerp(d001, d011, beta);
		Real d0yz = lerp(d0y0, d0y1, gamma);

		Real d1y0 = lerp(d100, d110, beta);
		Real d1y1 = lerp(d101, d111, beta);
		Real d1yz = lerp(d1y0, d1y1, gamma);

		Real dx0z = lerp(dx00, dx01, gamma);
		Real dx1z = lerp(dx10, dx11, gamma);

		normal[0] = d0yz - d1yz;
		normal[1] = dx0z - dx1z;
		normal[2] = dxy0 - dxy1;

		Real l = normal.norm();
		if (l < 0.0001f) normal = Coord(0);
		else normal = normal.normalize();

		d = (1.0f - gamma) * dxy0 + gamma * dxy1;
		return d;
	};

	template<typename TDataType>
	typename TDataType::Coord PoissonDiskSampler<TDataType>::getOnePointInsideSDF()
	{
		Coord normal(0.0f);
		for (Real ax = area_a[0]; ax < area_b[0]; ax += this->varSpacing()->getData())
			for (Real ay = area_a[1]; ay < area_b[1]; ay += this->varSpacing()->getData())
				for (Real az = area_a[2]; az < area_b[2]; az += this->varSpacing()->getData())
				{
					Real dr = getDistanceFromSDF(Coord(ax, ay, az), normal);
					if (dr < 0.0f)
					{
						return Coord(ax, ay, az);
					}
				}
	};


	template<typename TDataType>
	void PoissonDiskSampler<TDataType>::resetStates()
	{

		Real r = this->varSpacing()->getData();

		if (this->getVolume() != nullptr)
		{
			m_inputSDF = std::make_shared<dyno::DistanceField3D<TDataType>>();
			m_inputSDF->assign(this->getVolume()->stateLevelSet()->getData().getSDF());
		}
		else
		{
			return;
		}

		Coord minPoint = m_inputSDF->lowerBound();
		Coord maxPoint = m_inputSDF->upperBound();
		area_a = minPoint - Coord(r * 3);
		area_b = maxPoint + Coord(r * 3);

		Real m_h = m_inputSDF->getGridSpacing();

		host_dist.resize(m_inputSDF->nx(), m_inputSDF->ny(), m_inputSDF->nz());
		host_dist.assign(m_inputSDF->distances());

		unsigned int desired_points = pointNumberRecommend();

		std::cout << "Poisson Disk Sampling Start." << std::endl;

		Coord normal(0.0f);
		this->ConstructGrid();
		Coord seed_point = (area_a + (area_b - area_a)) / 2;

		/*The seed point must be inside the .SDF boundary */
		seed_point = getOnePointInsideSDF();

		m_points.push_back(seed_point);
		GridIndex gridIndex = searchGrid(seed_point);
		int index = indexTransform(gridIndex.i, gridIndex.j, gridIndex.k);
		m_grid[index] = 0;

		int head = 0;
		int tail = 1;

		while ((head < desired_points) && (head < tail))
		{
			Coord source = m_points[head];
			head++;
			for (int ppp = 0; ppp < m_attempted_times; ppp++)
			{
				Real theta = (Real)(rand() % 100) / 100.0f;
				theta = theta * 2 * 3.1415926535;

				Real phi = (Real)(rand() % 100) / 100.0f;
				phi = phi * 2 * 3.1415926535;

				Real dr = (1 + (Real)(rand() % 100) / 100.0f) * r;
				Coord offset(0.0f);

				offset = Coord(cos(theta) * sin(phi) * dr, sin(theta) * sin(phi) * dr, cos(phi) * r);

				Coord new_point = source + offset;

				if (((new_point[0] > area_a[0] + r * 3) && (new_point[0] < area_b[0] - r * 3)
					&& (new_point[1] > area_a[1] + r * 3) && (new_point[1] < area_b[1] - r * 3)
					&& (new_point[2] > area_a[2] + r * 3) && (new_point[2] < area_b[2] - r * 3)
					))
				{
					if (!collisionJudge(new_point) && (tail < desired_points) && (getDistanceFromSDF(new_point, normal) < 0.0f)) {
						m_points.push_back(new_point);
						gridIndex = searchGrid(new_point);
						m_grid[indexTransform(gridIndex.i, gridIndex.j, gridIndex.k)] = tail;
						tail++;
					}
				}
			}
		}

		auto ptSet = this->statePointSet()->getDataPtr();
		ptSet->setPoints(m_points);
		std::cout << "Poisson Disk Sampling is finished." << std::endl;

		m_points.clear();
	}

	DEFINE_CLASS(PoissonDiskSampler);
}