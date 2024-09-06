#include "SphereSampler.h"

#include "GLPointVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	SphereSampler<TDataType>::SphereSampler()
		: Sampler<TDataType>()
	{
		this->varSamplingDistance()->setRange(0.001, 1.0);
	}

	template<typename TDataType>
	void SphereSampler<TDataType>::resetStates()
	{
		auto box = this->inSphere()->getData();

		Real r = box.radius;
		Coord center = box.center;
		Real distance = this->varSamplingDistance()->getValue();

		std::vector<Coord> points;
		for (Real x = center[0] - r; x < center[0] + r; x += distance) {
			for (Real y = center[1] - r; y < center[1] + r; y += distance) {
				for (Real z = center[2] - r; z < center[2] + r; z += distance)
				{
					Real h = (Coord(x, y, z) - center).norm();
					if (h < r)
					{
						points.push_back(Coord(x, y, z));
					}
				}
			}
		}

		auto ptSet = this->statePointSet()->getDataPtr();
		ptSet->setPoints(points);
		points.clear();

	}

	DEFINE_CLASS(SphereSampler);
}