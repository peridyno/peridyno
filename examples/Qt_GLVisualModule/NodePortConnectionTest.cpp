#include "NodePortConnectionTest.h"

#include <Module/CalculateNorm.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>

namespace dyno
{
	IMPLEMENT_TCLASS(NodePortConnectionTest, TDataType)

	template<typename TDataType>
	NodePortConnectionTest<TDataType>::NodePortConnectionTest()
		: Node()
	{
		this->stateTopology()->setDataPtr(std::make_shared<PointSet<TDataType>>());

		std::vector<Coord> vertList;

		Coord lo = Coord(0.8, 0.2, 0.4);
		Coord hi = Coord(0.9, 0.3, 0.6);
		Real distance = 0.005;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p = Coord(x, y, z);
					vertList.push_back(Coord(x, y, z));
				}
			}
		}

		auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->stateTopology()->getDataPtr());
		ptSet->setPoints(vertList);

		vertList.clear();

		//this->loadParticles(Vec3f(0.8, 0.2, 0.4), Vec3f(0.9, 0.3, 0.6), 0.005);

		auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		colorMapper->varMax()->setValue(5.0f);

		this->stateVector()->connect(calculateNorm->inVec());
		calculateNorm->outNorm()->connect(colorMapper->inScalar());

		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->setColor(Vec3f(1, 0, 0));
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
		ptRender->setColorMapRange(0, 5);

		this->stateTopology()->connect(ptRender->inPointSet());
		colorMapper->outColor()->connect(ptRender->inColor());
		
		this->graphicsPipeline()->pushModule(calculateNorm);
		this->graphicsPipeline()->pushModule(colorMapper);
		this->graphicsPipeline()->pushModule(ptRender);
	}

	template<typename TDataType>
	NodePortConnectionTest<TDataType>::~NodePortConnectionTest()
	{
		printf("GLPointVisualNode released \n");
	}

	template<typename TDataType>
	void NodePortConnectionTest<TDataType>::preUpdateStates()
	{
		if (this->getParticleSystem() == nullptr) {
			this->stateVector()->resize(0);

			auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->stateTopology()->getDataPtr());
			auto& pts = ptSet->getPoints();
			pts.resize(0);

			return;
		}

		auto& poss = this->getParticleSystem()->statePosition()->getData();
		auto& vels = this->getParticleSystem()->stateVelocity()->getData();

		this->stateVector()->resize(poss.size());

		auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->stateTopology()->getDataPtr());
		auto& pts = ptSet->getPoints();

		if (poss.size() != pts.size())
		{
			pts.resize(poss.size());
		}

		ptSet->setPoints(poss);
		this->stateVector()->getDataPtr()->assign(vels);
	}

	template<typename TDataType>
	void NodePortConnectionTest<TDataType>::resetStates()
	{
		this->update();
	}

	DEFINE_CLASS(NodePortConnectionTest);
}
