#include "StlLoader.h"

#include "Topology/TriangleSet.h"
#include <iostream>
#include <sys/stat.h>

#include "GLWireframeVisualModule.h"
#include "GLSurfaceVisualModule.h"
#include "GLPointVisualModule.h"
#include "helpers/stlreader_helper.h"



namespace dyno
{
	IMPLEMENT_TCLASS(StlLoader, TDataType)

		template<typename TDataType>
	StlLoader<TDataType>::StlLoader()
		: Node()
	{

		this->stateTopology()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		auto surfacerender = std::make_shared<GLSurfaceVisualModule>();
		surfacerender->setVisible(true);
		surfacerender->setColor(Color(0.8, 0.52, 0.25));

		this->stateTopology()->connect(surfacerender->inTriangleSet());
		this->graphicsPipeline()->pushModule(surfacerender);
	}


	template<typename TDataType>
	void StlLoader<TDataType>::resetStates()
	{
		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->stateTopology()->getDataPtr());

		if (this->varFileName()->getValue().string() == "")
			return;
		std::string filename = this->varFileName()->getValue().string();
		loadSTL(*triSet, filename);
		triSet->scale(this->varScale()->getValue());
		triSet->translate(this->varLocation()->getValue());
		triSet->rotate(this->varRotation()->getValue() * M_PI / 180);
		triSet->update();

		Node::resetStates();

		initPos.assign(triSet->getPoints());
	}


	template<typename TDataType>
	void StlLoader<TDataType>::loadSTL(TriangleSet<TDataType>& Triangleset, std::string filename)
	{
		std::vector<Coord> vertList;
		std::vector<TopologyModule::Triangle> faceList;
		dyno::loadStl(vertList,faceList,filename);

		Triangleset.setPoints(vertList);
		Triangleset.setTriangles(faceList);
		Triangleset.update();

	}


	DEFINE_CLASS(StlLoader);
}