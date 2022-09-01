#include "GLPointVisualModule.h"
#include "PoissonDiskSampling.h"


namespace dyno
{
	template <typename TDataType>
	PoissonDiksSampling<TDataType>::PoissonDiksSampling()
		: Sampler()
	{
			this->varSamplingDistance()->setRange(0.001, 1.0);

			area_a = Coord(0.0f);
			area_b = Coord(1.5f);
			desired_points = pointNumberRecommend();

	};


	template <typename TDataType>
	int PoissonDiksSampling<TDataType>::pointNumberRecommend()
	{
		int num = 0;
		Coord box = area_b - area_a;
		Real r = this->varSamplingDistance()->getData();
		for(Real d = area_a[0]; d < area_b[0]; d += r)
			for (Real d = area_a[1]; d < area_b[1]; d += r)
			{
				num++;
			}
		return num;
	};


	template <typename TDataType>
	void PoissonDiksSampling<TDataType>::ConstructGrid() 
	{
		auto r = this->varSamplingDistance()->getData();
		dx = r / sqrt(2);
		if (dx == 0)
		{
			std::cout <<" The smapling distance in Poisson disk sampling module can not be ZERO!!!! " << std::endl;
			exit(0);
		}

		int dimension = this->varDimension()->getData();

		nx = abs(area_b[0] - area_a[0]) / dx;
		ny = abs(area_b[1] - area_a[1]) / dx;

		gnum = nx * ny;
		m_grid.resize(gnum);

		for (int i = 0; i < gnum; i++)	m_grid[i] = -1;
	}


	template <typename TDataType>
	GridIndex PoissonDiksSampling<TDataType>::searchGrid(Coord point)
	{
		GridIndex index;
		index.i = (int)((point[0] - area_a[0]) / dx);
		index.j = (int)((point[1] - area_a[1]) / dx);
		index.k = 0;
		return index;
	};


	template <typename TDataType>
	int PoissonDiksSampling<TDataType>::indexTransform(int i, int j, int k) 
	{
		return i + j * nx;
	};

	template <typename TDataType>
	bool PoissonDiksSampling<TDataType>::collisionJudge(Coord point)
	{
		bool flag = false;
		auto r = this->varSamplingDistance()->getData();
		GridIndex d_index;
		d_index = searchGrid(point);
		for(int i = -2; i < 3; i++)
			for (int j = -2; j < 3; j++)
			{

				int mi = d_index.i + i;
				int mj = d_index.j + j;
				if ((mi > 0) && ( mj > 0) && (mi < nx) && (mj < ny)) 
				{
					int index = indexTransform(mi, mj, d_index.k);
					if (m_grid[index] != -1)
					{
						Coord d = (points[m_grid[index]] - point);
						if (sqrt(d[0] * d[0] + d[1] * d[1]) - r < EPSILON)
						{
							flag = true;
						}
					}
				}
			}
		return flag;
	};


	template<typename TDataType>
	void PoissonDiksSampling<TDataType>::resetStates()
	{

		std::cout << "Poisson Disk Sampling!!!!" << std::endl;

		auto r = this->varSamplingDistance()->getData();

		ConstructGrid();

		seed_point = (area_a + (area_b - area_a)) / 2;
		std::cout << "seed_point " << seed_point << std::endl;

		points.push_back(seed_point);


		gridIndex = searchGrid(seed_point);
		int index = indexTransform(gridIndex.i, gridIndex.j, gridIndex.k);
		m_grid[index] = 0;

		int head = 0;
		int tail = 1;


		while ( (head < desired_points) && (head < tail ) )
		{
			Coord source = points[head];
			head++;

			for (int ppp = 0; ppp < 100; ppp++)
			{
				Real theta = (Real)(rand()%100) / 100.0f ;
				Real dr = (1 + (Real)(rand() % 100) / 100.0f) * r;
				
				theta = theta * 2 * 3.1415926535;

				Coord offset = Coord(cos(theta) * dr, sin(theta) * dr, 0.0f);

				Coord new_point = source + offset;

				if ((new_point[0] > area_a[0] + 0.01) && (new_point[0] < area_b[0] - 0.01)
					&& (new_point[1] > area_a[1] + 0.01) && (new_point[1] < area_b[1] - 0.01))
				{
					if (!collisionJudge(new_point) && (tail < desired_points)) {
						points.push_back(new_point);
						gridIndex = searchGrid(new_point);
						m_grid[indexTransform(gridIndex.i, gridIndex.j, gridIndex.k)] = tail;
						tail++;
					}
				}
			}
		}

		std::cout << "Finish!!!" << std::endl;
		auto ptSet = this->statePointSet()->getDataPtr();

		ptSet->setPoints(points);
	}



	DEFINE_CLASS(PoissonDiksSampling);
}