#include "CubeSampler.h"

#include "GLPointVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	CubeSampler<TDataType>::CubeSampler()
		: Node()
	{
		this->varSamplingDistance()->setRange(0.01, 1.0);

		this->statePointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());

		auto module = std::make_shared<GLPointVisualModule>();
		module->setColor(Vec3f(0.25, 0.52, 0.8));
		module->setVisible(true);
		module->varPointSize()->setValue(0.01);
		this->statePointSet()->connect(module->inPointSet());
		this->graphicsPipeline()->pushModule(module);
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

		Real s = std::max(Real(0.01), this->varSamplingDistance()->getData());

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