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
		this->stateTriangleSet()->setDataPtr(triSet);

		this->stateInitialTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		auto surfaceRender = std::make_shared<GLSurfaceVisualModule>();
		surfaceRender->setColor(Color(0.8f, 0.52f, 0.25f));
		surfaceRender->setVisible(true);
		this->stateTriangleSet()->connect(surfaceRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(surfaceRender);

		auto callFileLoader = std::make_shared<FCallBackFunc>(
			[=]() {
				auto initTopo = this->stateInitialTriangleSet()->getDataPtr();
				auto curTopo = this->stateTriangleSet()->getDataPtr();

				std::string fileName = this->varFileName()->getValue().string();

				if (fileName != "")
				{
					initTopo->loadObjFile(fileName);
					curTopo->copyFrom(*initTopo);

					curTopo->scale(this->varScale()->getValue());
					curTopo->rotate(this->varRotation()->getValue() * M_PI / 180);
					curTopo->translate(this->varLocation()->getValue());
				}
			}
		);
		this->varFileName()->attach(callFileLoader);

		auto transform = std::make_shared<FCallBackFunc>(
			[=]() {
				auto initTopo = this->stateInitialTriangleSet()->getDataPtr();
				auto curTopo = this->stateTriangleSet()->getDataPtr();

				curTopo->copyFrom(*initTopo);
				curTopo->scale(this->varScale()->getValue());
				curTopo->rotate(this->varRotation()->getValue() * M_PI / 180);
				curTopo->translate(this->varLocation()->getValue());
			}
		);
		this->varLocation()->attach(transform);
		this->varScale()->attach(transform);
		this->varRotation()->attach(transform);
	}

	DEFINE_CLASS(StaticTriangularMesh);
}