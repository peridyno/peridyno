#include "CubeSampler.h"

#include "GLPointVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	CubeSampler<TDataType>::CubeSampler()
		: Sampler<TDataType>()
	{
		this->varSamplingDistance()->setRange(0.001, 1.0);
	}

	template<typename TDataType>
	void CubeSampler<TDataType>::resetStates()
	{
		auto box = this->inCube()->getData();

		Coord u = box.u;
		Coord v = box.v;
		Coord w = box.w;

		Coord ext = box.extent;
		Coord center = box.center;

		Real s = std::max(Real(0.001), this->varSamplingDistance()->getData());

		int nx = ext[0] / s;
		int ny = ext[1] / s;
		int nz = ext[2] / s;

		std::vector<Coord> points;
		for (int i = -nx; i <= nx; i++)
		{
			for (int j = -ny; j <= ny; j++)
			{
				for (int k = -nz; k <= nz; k++)
				{
					Coord p = center + (i * s) * u + (j * s) * v + (k * s) * w;
					points.push_back(p);
				}
			}
		}

		auto ptSet = this->statePointSet()->getDataPtr();

		ptSet->setPoints(points);

		points.clear();
	}

	DEFINE_CLASS(CubeSampler);
}