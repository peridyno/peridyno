#include "GLPointVisualModule.h"
#include "PoissonDiskSampling.h"


namespace dyno
{
	template <typename TDataType>
	PoissonDiksSampling<TDataType>::PoissonDiksSampling()
		: SamplingPoints()
	{
			this->varSamplingDistance()->setRange(0.001, 1.0);

			area_a = Coord(0.0f);
			area_b = Coord(1.0f);
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

		if (dimension == 2)
		{
			nx = abs(area_b[0] - area_a[0]) / dx;
			ny = abs(area_b[1] - area_a[1]) / dx;

			gnum = nx * ny;
			m_grid.resize(gnum);

			for (int i = 0; i < gnum; i++)	m_grid[i] = -1;
		}

	}


	template <typename TDataType>
	void PoissonDiksSampling<TDataType>::searchGrid(Coord point, int& i, int &j, int&k)
	{
		if (this->varDimension()->getData() == 2)
		{
			i = (int)(point[0]/dx);
			j = (int)(point[1]/dx);
			//k = (int)(points[2]/dx);
		}
	};


	template <typename TDataType>
	int PoissonDiksSampling<TDataType>::indexTransform(int i, int j, int k) 
	{
		int num;
		if (this->varDimension()->getData() == 2)
		{
			num = i + j * nx;
		}
		return num;
	}
	;

	template <typename TDataType>
	bool PoissonDiksSampling<TDataType>::collisionJudge(Coord point)
	{
		bool flag = false;
		int xi, xj, xk;
		auto r = this->varSamplingDistance()->getData();
		searchGrid(point, xi, xj, xk);
		for(int i = -2; i < 3; i++)
			for (int j = -2; j < 3; j++)
			{

				int mi = xi + i;
				int mj = xj + j;
				if ((mi > 0) && ( mj > 0) && (mi < nx) && (mj < ny)) 
				{
					int index = indexTransform(mi, mj, xk);
					if (m_grid[index] != -1)
					{
						Coord d = (points[m_grid[index]] - point);
						if (sqrt(d[0] * d[0] + d[1] * d[1]) - r < EPSILON)
						{
							flag = true;
							std::cout << "" << std::endl;
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

		int gi, gj, gk = 0;
		searchGrid(seed_point, gi, gj, gk);

		std::cout << "Grid index: " << gi << ", " << gj << ", " << gk << std::endl;
		int index = gi + gj * nx;
		std::cout << "Index: " << index << ", " << indexTransform(gi, gj, gk);

		m_grid[index] = 0;

		int head = 0;
		int tail = 1;


		//for (int i = 0; i < 1000; i++)
		//{
		//	Coord temp = Coord(( i * 0.001), 0.5, 0.0f);
		//	std::cout<< collisionJudge(temp) << ": " << ( i * 0.001 ) <<std::endl;
		//}


		int iii = 0;
		while ((iii < 100)&&(tail>head))
		{
			iii++;
			Coord source = points[head];
			head++;

			for (int ppp = 0; ppp < 100; ppp++)
			{
				Real theta = (Real)(rand()%100) / 100.0f;
				//std::cout << theta << std::endl;
				Real dr = (1 + (Real)(rand() % 100) / 100.0f) * r;
				
				Coord offset = Coord(cos(theta) * dr, sin(theta) * dr, 0.0f);

				Coord new_point = source + offset;

				if ((new_point[0] > area_a[0]) && (new_point[0] < area_b[0])
					&& (new_point[1] > area_a[1]) && (new_point[1] < area_b[1]))
				{
					if (collisionJudge(new_point)) {
						points.push_back(new_point);
						searchGrid(new_point, gi, gj, gk);
						m_grid[indexTransform(gi, gj, gk)] = points.size() - 1;
						tail++;
						
					}
					else
					{
						std::cout << "++++" <<std::endl;
					}
				}
			//	if (collisionJudge(new_point))


			}
		}

		auto ptSet = this->statePointSet()->getDataPtr();

		ptSet->setPoints(points);




	}



	DEFINE_CLASS(PoissonDiksSampling);
}