#include "StaticTriangularMesh.h"

#include "GLSurfaceVisualModule.h"

#include "Topology/TriangleSet.h"

namespace dyno
{
	IMPLEMENT_TCLASS(StaticTriangularMesh, TDataType)

	template<typename TDataType>
	StaticTriangularMesh<TDataType>::StaticTriangularMesh()
		: ParametricModel<TDataType>()
	{
		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		this->stateTopology()->setDataPtr(triSet);

		this->stateInitialTopology()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		auto surfaceRender = std::make_shared<GLSurfaceVisualModule>();
		surfaceRender->setColor(Color(0.8f, 0.52f, 0.25f));
		surfaceRender->setVisible(true);
		this->stateTopology()->connect(surfaceRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(surfaceRender);

		auto callFileLoader = std::make_shared<FCallBackFunc>(
			[=]() {
				auto initTopo = this->stateInitialTopology()->getDataPtr();
				auto curTopo = this->stateTopology()->getDataPtr();

				std::string fileName = this->varFileName()->getDataPtr()->string();

				if (fileName != "")
				{
					initTopo->loadObjFile(fileName);
					curTopo->copyFrom(*initTopo);

					curTopo->scale(this->varScale()->getData());
					curTopo->rotate(this->varRotation()->getData() * M_PI / 180);
					curTopo->translate(this->varLocation()->getData());
				}
			}
		);
		this->varFileName()->attach(callFileLoader);

		auto transform = std::make_shared<FCallBackFunc>(
			[=]() {
				auto initTopo = this->stateInitialTopology()->getDataPtr();
				auto curTopo = this->stateTopology()->getDataPtr();

				curTopo->copyFrom(*initTopo);
				curTopo->scale(this->varScale()->getData());
				curTopo->rotate(this->varRotation()->getData() * M_PI / 180);
				curTopo->translate(this->varLocation()->getData());
			}
		);
		this->varLocation()->attach(transform);
		this->varScale()->attach(transform);
		this->varRotation()->attach(transform);
	}

	DEFINE_CLASS(StaticTriangularMesh);
}