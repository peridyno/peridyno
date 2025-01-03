#include "ShapeSampler.h"

#include "BasicShapes/CubeModel.h"
#include "BasicShapes/SphereModel.h"

namespace dyno
{
	template<typename TDataType>
	ShapeSampler<TDataType>::ShapeSampler()
		: Sampler<TDataType>()
	{
		this->varSamplingDistance()->setRange(0.001, 1.0);
	}

	template<typename TDataType>
	void ShapeSampler<TDataType>::resetStates()
	{
		auto shape = this->getShape();

		if (shape == nullptr) return;

		auto shapeType = shape->getShapeType();
		if (shapeType == BasicShapeType::CUBE)
		{
			auto model = dynamic_cast<CubeModel<TDataType>*>(shape);

			if (model == nullptr) return;

			auto box = model->outCube()->getValue();
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
		else if (shapeType == BasicShapeType::SPHERE)
		{
			auto model = dynamic_cast<SphereModel<TDataType>*>(shape);

			if (model == nullptr) return;

			auto sphere = model->outSphere()->getValue();
			Real r = sphere.radius;
			Coord center = sphere.center;
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
		
	}

	DEFINE_CLASS(ShapeSampler);
}