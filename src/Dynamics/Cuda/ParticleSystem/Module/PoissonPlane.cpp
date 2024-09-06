#include "GLPointVisualModule.h"
#include "PoissonPlane.h"
#include <ctime> 

namespace dyno
{
	template <typename TDataType>
	PoissonPlane<TDataType>::PoissonPlane()
		: ComputeModule()
	{
	};


	template <typename TDataType>
	int PoissonPlane<TDataType>::pointNumberRecommend()
	{
		Vec2f area_a = this->varLower()->getValue();
		Vec2f area_b = this->varUpper()->getValue();

		float r = this->varSamplingDistance()->getData();

		return abs((area_b[0] - area_a[0]) * (area_b[1] - area_a[1]) / (r * r));	 
	};


	template <typename TDataType>
	void PoissonPlane<TDataType>::ConstructGrid()
	{
		auto r = this->varSamplingDistance()->getData();
		dx = r / sqrt(2);
		if (dx == 0)
		{
			std::cout << " The smapling distance in Poisson disk sampling module can not be ZERO!!!! " << std::endl;
			exit(0);
		}
		
		Vec2f area_a = this->varLower()->getValue();
		Vec2f area_b = this->varUpper()->getValue();

		mOrigin = area_a - Vec2f(4 * dx, 4 * dx);
		mUpperBound = area_b + Vec2f(4 * dx, 4 * dx);

		nx = abs(mUpperBound[0] - mOrigin[0]) / dx;
		ny = abs(mUpperBound[1] - mOrigin[1]) / dx;

		gnum = nx * ny;

		m_grid.clear();
		m_grid.resize(gnum);

		for (int i = 0; i < gnum; i++)	m_grid[i] = -1;
	}


	template <typename TDataType>
	Vec2u PoissonPlane<TDataType>::searchGrid(Vec2f point)
	{
		Vec2u index;
		Vec2f area_a = this->varLower()->getValue();
		Vec2f area_b = this->varUpper()->getValue();

		index[0] = (int)((point[0] - mOrigin[0]) / dx);
		index[1] = (int)((point[1] - mOrigin[1]) / dx);
		return index;
	};


	template <typename TDataType>
	int PoissonPlane<TDataType>::indexTransform(uint i, uint j)
	{
		return i + j * nx;
	};

	template <typename TDataType>
	bool PoissonPlane<TDataType>::collisionJudge(Vec2f point)
	{
		bool flag = false;
		auto r = this->varSamplingDistance()->getData();
		Vec2u d_index;
		d_index = searchGrid(point);
		for (int i = -2; i < 3; i++)
			for (int j = -2; j < 3; j++)
			{

				int mi = d_index[0] + i;
				int mj = d_index[1] + j;
				if ((mi > 0) && (mj > 0) && (mi < nx) && (mj < ny))
				{
					int index = indexTransform(mi, mj);
					if (m_grid[index] != -1)
					{
						Vec2f d = (points[m_grid[index]] - point);
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
	void PoissonPlane<TDataType>::compute()
	{
		auto r = this->varSamplingDistance()->getData();
		desired_points = pointNumberRecommend();

		ConstructGrid();

		Vec2f area_a = this->varLower()->getValue();
		Vec2f area_b = this->varUpper()->getValue();

		Vec2f seed_point = (area_a + (area_b - area_a) / 2);
		seed_point += (Vec2f((float)(rand() % 100) / 100.0f, (float)(rand() % 100) / 100.0f) - Vec2f(0.5, 0.5)) * r;
		//std::cout <<"Offset: " << (Vec2f((float)(rand() % 100) / 100.0f, (float)(rand() % 100) / 100.0f) - Vec2f(0.5, 0.5))  << std::endl;
		
		points.clear();
		points.push_back(seed_point);

		gridIndex = searchGrid(seed_point);
		int index = indexTransform(gridIndex[0], gridIndex[1]);
		m_grid[index] = 0;

		int head = 0;
		int tail = 1;

		while ((head < desired_points) && (head < tail))
		{
			Vec2f source = points[head];
			head++;

			for (int ppp = 0; ppp < 100; ppp++)
			{
				float theta = (float)(rand() % 100) / 100.0f;
				float dr = (1 + (float)(rand() % 100) / 100.0f) * r;

				theta = theta * 2 * 3.1415926535;

				Vec2f offset = Vec2f(cos(theta) * dr, sin(theta) * dr);

				Vec2f new_point = source + offset;

				if ((new_point[0] > area_a[0] ) && (new_point[0] < area_b[0] )
					&& (new_point[1] > area_a[1] ) && (new_point[1] < area_b[1] ))
				{
					if (!collisionJudge(new_point) && (tail < desired_points)) {
						points.push_back(new_point);
						gridIndex = searchGrid(new_point);
						m_grid[indexTransform(gridIndex[0], gridIndex[1])] = tail;
						tail++;
					}
				}
			}
		}
	}



	DEFINE_CLASS(PoissonPlane);
}