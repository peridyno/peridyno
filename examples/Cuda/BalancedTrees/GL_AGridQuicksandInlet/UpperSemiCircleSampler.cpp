#include "UpperSemiCircleSampler.h"

#include "GLPointVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	UpperSemiCircleSampler<TDataType>::UpperSemiCircleSampler()
		: Sampler<TDataType>()
	{
		this->varSamplingDistance()->setRange(0.001, 1.0);
	}

	template<typename TDataType>
	void UpperSemiCircleSampler<TDataType>::resetStates()
	{
		Coord center = this->varCenter()->getData();
		Real redius = this->varRadius()->getData();
		Real yplane = this->varYPlane()->getData();
		Real dx = this->varDx()->getData();

		Real s = std::max(Real(0.0001), this->varSamplingDistance()->getData());

		int nx = redius / s;
		int ny = redius / s;
		int nz = (dx / 2) / s;

		std::vector<Coord> points;
		for (int i = -nx; i <= nx; i++)
		{
			for (int j = -ny; j <= ny; j++)
			{
				for (int k = -nz; k <=nz; k++)
				{
					Coord poffset((i)*s, (j)*s, 0.0f);
					if (poffset.norm() > redius) break;

					if ((center[1] + poffset[1]) < yplane) break;

					Coord p = center + poffset + Coord(0.0f, 0.0f, k*s);
					points.push_back(p);
				}
			}
		}

		auto ptSet = this->statePointSet()->getDataPtr();
		ptSet->setPoints(points);

		points.clear();
	}

	DEFINE_CLASS(UpperSemiCircleSampler);
}