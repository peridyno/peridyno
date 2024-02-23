#include "PointFromCurve.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"



namespace dyno
{
	template<typename TDataType>
	PointFromCurve<TDataType>::PointFromCurve()
		: ParametricModel<TDataType>()
	{

		this->varScale()->setRange(0.001f, 10.0f);

		this->statePointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		this->stateEdgeSet()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());

		this->statePointSet()->promoteOuput();

		auto pointGlModule = std::make_shared<GLPointVisualModule>();
		this->statePointSet()->connect(pointGlModule->inPointSet());
		this->graphicsPipeline()->pushModule(pointGlModule);
		pointGlModule->varPointSize()->setValue(0.01);

		auto wireframe = std::make_shared<GLWireframeVisualModule>();
		this->stateEdgeSet()->connect(wireframe->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireframe);
		wireframe->setColor(Color(1, 1, 0));
		wireframe->varLineWidth()->setValue(0.1);
		wireframe->varRenderMode()->setCurrentKey(1);

		this->statePointSet()->promoteOuput();
		auto curve = this->varCurve()->getValue();
		curve.setUseSquard(true);
		this->varCurve()->setValue(curve);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&PointFromCurve<TDataType>::varChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varUniformScale()->attach(callback);
		this->varCurve()->attach(callback);

	}

	template<typename TDataType>
	void PointFromCurve<TDataType>::resetStates()
	{
		varChanged();
	}

	template<typename TDataType>
	void PointFromCurve<TDataType>::varChanged()
	{
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();
		auto uniformScale = this->varUniformScale()->getData();
		auto pointSet = this->statePointSet()->getDataPtr();
		auto curve = this->varCurve()->getValue();
		auto floatCoord = curve.FinalCoord;
		int length = floatCoord.size();
		std::vector<Coord> vertices;


		Coord Location;
		if (length != 0)
		{
			for (size_t i = 0; i < length; i++)
			{
				Location = Coord(floatCoord[i].x, floatCoord[i].y, 0);

				vertices.push_back(Location);
			}
		}

		if (curve.curveClose && curve.resample == true && vertices.size()>=3)
		{

			vertices.erase(vertices.end() - 1);

		}

		//Transform Coord

		Quat<Real> q2 = computeQuaternion();

		q2.normalize();

		auto RVt = [&](const Coord& v)->Coord {
			return center + q2.rotate(v - center);
		};

		int numpt = vertices.size();

		for (int i = 0; i < numpt; i++)
		{

			vertices[i] = RVt(vertices[i] * scale * uniformScale + RVt(center));
		}

		int s = vertices.size();

		pointSet->setPoints(vertices);


	


		// create EdgeSet
		{
			std::vector<TopologyModule::Edge> edges;


			auto edgeSet = this->stateEdgeSet()->getDataPtr();

			int ptnum = vertices.size();

			
			//printf("vertices.size  %d \n", ptnum);
			if (ptnum >= 2)
			{
				for (int i = 0; i < ptnum - 1; i++)
				{
					edges.push_back(TopologyModule::Edge(i, i + 1));
					//printf(" %d  -- %d  \n", i, i + 1);
				}
			}
			//printf(" set \n");

			if (curve.curveClose == true && vertices.size()>=3) 
			{
				edges.push_back(TopologyModule::Edge(vertices.size()-1,0));
			}



			edgeSet->setPoints(vertices);
			edgeSet->setEdges(edges);


			vertices.clear();
			edges.clear();
			//printf(" clear  \n");
		
		
		}
		
	}

	DEFINE_CLASS(PointFromCurve);
}