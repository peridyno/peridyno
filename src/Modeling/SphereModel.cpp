#include "SphereModel.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"



namespace dyno
{
	template<typename TDataType>
	SphereModel<TDataType>::SphereModel()
		: ParametricModel<TDataType>()
	{
		this->varRadius()->setRange(0.001f, 10.0f);

		this->varTheta()->setRange(0.001f, 1.5f);

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		


		glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Vec3f(0.8, 0.52, 0.25));
		glModule->setVisible(true);
		this->stateTriangleSet()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);


		auto wireframe = std::make_shared<GLWireframeVisualModule>();
		this->stateTriangleSet()->connect(wireframe->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireframe);


	}

	template<typename TDataType>
	void SphereModel<TDataType>::resetStates()
	{
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		auto radius = this->varRadius()->getData();
		auto theta = this->varTheta()->getData();


		radius *= (sqrt(scale[0]* scale[0] + scale[1]* scale[1] + scale[2]* scale[2]));

		TSphere3D<Real> ball;
		ball.center = center;
		ball.radius = radius;
		this->outSphere()->setValue(ball);


		auto triangleSet = this->stateTriangleSet()->getDataPtr();


		Real PI = 3.1415926535;

		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> triangle;

		Coord point(0.0f);

		for (float d_alpha = -PI/2 + theta/2; d_alpha < PI/2; d_alpha += theta )
		{
			point[1] = center[1] + radius * sin(d_alpha);

			//std::cout << "theta " << da << std::endl;
			for (float d_beta = 0.0f; d_beta < 2 * PI; d_beta += theta )
			{
				point[0] = center[0] + (cos(d_alpha) * radius) * sin(d_beta);
				point[2] = center[2] + (cos(d_alpha) * radius) * cos(d_beta);
				vertices.push_back(point);
			}

		}

		vertices.push_back(Coord(center[0], -radius + center[1], center[2]));
		vertices.push_back(Coord(center[0], radius + center[1], center[2]));


		int face_id = 0;
		for (float d_alpha = -PI / 2 + theta/2; d_alpha < PI / 2; d_alpha += theta)
		{


			for (float d_beta = 0.0f; d_beta < 2 * PI; d_beta += theta)
			{
				if ((d_beta + theta - 2 * PI < EPSILON)&&(d_alpha + theta < PI / 2))
				{
					int cd = 2 * PI / theta;
					triangle.push_back(TopologyModule::Triangle(face_id, face_id+1, face_id + cd + 1));
				}
				else if((d_beta + theta - 2 * PI >= EPSILON) && (d_alpha + theta < PI / 2))
				{
					int cd = 2 * PI / theta;
					triangle.push_back(TopologyModule::Triangle(face_id, face_id - cd , face_id + cd + 1));
				}
				else if ((d_beta + theta - 2 * PI < EPSILON) && (d_alpha + theta >= PI / 2))
				{
					triangle.push_back(TopologyModule::Triangle(face_id, face_id + 1, vertices.size() - 1));
				}
				else if ((d_beta + theta - 2 * PI >= EPSILON) && (d_alpha + theta >= PI / 2))
				{
					int cd = 2 * PI / theta;
					triangle.push_back(TopologyModule::Triangle(face_id, face_id - cd, vertices.size() - 1));
				}



				if ((d_beta + theta - 2 * PI < EPSILON) && (d_alpha - theta + PI / 2 > EPSILON ))
				{
					int cd = 2 * PI / theta;
					triangle.push_back(TopologyModule::Triangle(face_id, face_id + 1, face_id - cd));
				}
				else if ((d_beta + theta - 2 * PI > EPSILON) && (d_alpha - theta + PI / 2 > EPSILON))
				{
					int cd = 2 * PI / theta;
					triangle.push_back(TopologyModule::Triangle(face_id, face_id - cd, face_id - 2 * cd - 1));
				}
				else if ((d_beta + theta - 2 * PI < EPSILON) && (d_alpha - theta + PI / 2 < EPSILON))
				{
					triangle.push_back(TopologyModule::Triangle(face_id, face_id + 1, vertices.size() - 2));
				}
				else if ((d_beta + theta - 2 * PI >= EPSILON) && (d_alpha - theta + PI / 2 < EPSILON))
				{
					int cd = 2 * PI / theta;
					triangle.push_back(TopologyModule::Triangle(face_id, face_id - cd, vertices.size() - 2));
				}

				face_id++;
			}
		}

		triangleSet->setPoints(vertices);
		triangleSet->setTriangles(triangle);

		triangleSet->updateEdges();
		triangleSet->updateVertexNormal();


		triangleSet->update();

		vertices.clear();
		triangle.clear();
	}

	template<typename TDataType>
	void SphereModel<TDataType>::disableRender() {
		glModule->setVisible(false);
	};

	DEFINE_CLASS(SphereModel);
}