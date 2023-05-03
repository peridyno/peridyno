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

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());


		glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Color(0.8f, 0.52f, 0.25f));
		glModule->setVisible(true);
		this->stateTriangleSet()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);


		auto wireframe = std::make_shared<GLWireframeVisualModule>();
		this->stateTriangleSet()->connect(wireframe->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireframe);

		this->stateTriangleSet()->promoteOuput();

		auto pointGlModule = std::make_shared<GLPointVisualModule>();
		this->stateTriangleSet()->connect(pointGlModule->inPointSet());
		this->graphicsPipeline()->pushModule(pointGlModule);
		pointGlModule->varPointSize()->setValue(0.02);

		this->stateTriangleSet()->promoteOuput();
	}

	template<typename TDataType>
	void PointFromCurve<TDataType>::resetStates()
	{

		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();
		auto uniformScale = this->varUniformScale()->getData();
		auto triangleSet = this->stateTriangleSet()->getDataPtr();
		auto Ramp = this->varCurveRamp()->getDataPtr();
		auto floatCoord = Ramp->MyCoord;
		int length = floatCoord.size();
		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> triangle;

		Coord Location;
		Vec3f Dir;
		if (length!= 0)
		{
			for (size_t i = 0; i < length; i++)
			{
				Location = Coord(floatCoord[i].x, floatCoord[i].y, 0);

				vertices.push_back(Location);
			}
		}




		//±ä»»

		Quat<Real> q2 = Quat<Real>(M_PI * rot[0] / 180, Coord(1, 0, 0))
			* Quat<Real>(M_PI * rot[1] / 180, Coord(0, 1, 0))
			* Quat<Real>(M_PI * rot[2] / 180, Coord(0, 0, 1));

		q2.normalize();

		auto RVt = [&](const Coord& v)->Coord {
			return center + q2.rotate(v - center);
		};

		int numpt = vertices.size();

		for (int i = 0; i < numpt; i++)
		{

			vertices[i] =RVt( vertices[i] * scale * uniformScale + RVt( center ));
		}

		int s = vertices.size();

		triangleSet->setPoints(vertices);
		triangleSet->setTriangles(triangle);

		triangleSet->update();



		vertices.clear();
		triangle.clear();
		

	}


	template<typename TDataType>
	void PointFromCurve<TDataType>::disableRender() {
		glModule->setVisible(false);
	};


	DEFINE_CLASS(PointFromCurve);
}