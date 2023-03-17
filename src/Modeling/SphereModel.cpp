#include "SphereModel.h"

#include "Primitive/Primitive3D.h"

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
		

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&SphereModel<TDataType>::varChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varRadius()->attach(callback);
		this->varTheta()->attach(callback);
		
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
		varChanged();
	}

	template<typename TDataType>
	void SphereModel<TDataType>::disableRender() {
		glModule->setVisible(false);
	};

	template<typename TDataType>
	NBoundingBox SphereModel<TDataType>::boundingBox()
	{
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		auto radius = this->varRadius()->getData();

		Coord length(radius);
		length[0] *= scale[0];
		length[1] *= scale[1];
		length[2] *= scale[2];

		Quat<Real> q = Quat<Real>(M_PI * rot[0] / 180, Coord(1, 0, 0))
			* Quat<Real>(M_PI * rot[1] / 180, Coord(0, 1, 0))
			* Quat<Real>(M_PI * rot[2] / 180, Coord(0, 0, 1));

		q.normalize();

		TOrientedBox3D<Real> box;
		box.center = center;
		box.u = q.rotate(Coord(1, 0, 0));
		box.v = q.rotate(Coord(0, 1, 0));
		box.w = q.rotate(Coord(0, 0, 1));
		box.extent = length;

		auto AABB = box.aabb();
		auto& v0 = AABB.v0;
		auto& v1 = AABB.v1;
		

		NBoundingBox ret;
		ret.lower = Vec3f(v0.x, v0.y, v0.z);
		ret.upper = Vec3f(v1.x, v1.y, v1.z);

		return ret;
	}


	template<typename TDataType>
	void SphereModel<TDataType>::varChanged()
	{
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		auto radius = this->varRadius()->getData();
		auto theta = this->varTheta()->getData();


		radius *= (sqrt(scale[0] * scale[0] + scale[1] * scale[1] + scale[2] * scale[2]));

		TSphere3D<Real> ball;
		ball.center = center;
		ball.radius = radius;
		this->outSphere()->setValue(ball);


		auto triangleSet = this->stateTriangleSet()->getDataPtr();


		Real PI = 3.1415926535;

		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> triangle;

		Coord point(0.0f);

		for (float d_alpha = -PI / 2 + theta / 2; d_alpha < PI / 2; d_alpha += theta)
		{
			point[1] = center[1] + radius * sin(d_alpha);

			//std::cout << "theta " << da << std::endl;
			for (float d_beta = 0.0f; d_beta < 2 * PI; d_beta += theta)
			{
				point[0] = center[0] + (cos(d_alpha) * radius) * sin(d_beta);
				point[2] = center[2] + (cos(d_alpha) * radius) * cos(d_beta);
				vertices.push_back(point);
			}

		}

		vertices.push_back(Coord(center[0], -radius + center[1], center[2]));
		vertices.push_back(Coord(center[0], radius + center[1], center[2]));

		int face_id = 0;
		for (float d_alpha = -PI / 2 + theta / 2; d_alpha < PI / 2; d_alpha += theta)
		{
			for (float d_beta = 0.0f; d_beta < 2 * PI; d_beta += theta)
			{
				if ((d_beta + theta - 2 * PI < EPSILON) && (d_alpha + theta < PI / 2))
				{
					int cd = 2 * PI / theta;
					triangle.push_back(TopologyModule::Triangle(face_id, face_id + 1, face_id + cd + 1));
				}
				else if ((d_beta + theta - 2 * PI >= EPSILON) && (d_alpha + theta < PI / 2))
				{
					int cd = 2 * PI / theta;
					triangle.push_back(TopologyModule::Triangle(face_id, face_id - cd, face_id + cd + 1));
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



				if ((d_beta + theta - 2 * PI < EPSILON) && (d_alpha - theta + PI / 2 > EPSILON))
				{
					int cd = 2 * PI / theta;
					triangle.push_back(TopologyModule::Triangle(face_id, face_id - cd, face_id + 1));
				}
				else if ((d_beta + theta - 2 * PI > EPSILON) && (d_alpha - theta + PI / 2 > EPSILON))
				{
					int cd = 2 * PI / theta;
					triangle.push_back(TopologyModule::Triangle(face_id, face_id - 2 * cd - 1, face_id - cd));
				}
				else if ((d_beta + theta - 2 * PI < EPSILON) && (d_alpha - theta + PI / 2 < EPSILON))
				{
					triangle.push_back(TopologyModule::Triangle(face_id, vertices.size() - 2, face_id + 1));
				}
				else if ((d_beta + theta - 2 * PI >= EPSILON) && (d_alpha - theta + PI / 2 < EPSILON))
				{
					int cd = 2 * PI / theta;
					triangle.push_back(TopologyModule::Triangle(face_id, vertices.size() - 2, face_id - cd));
				}

				face_id++;
			}
		}

		triangleSet->setPoints(vertices);
		triangleSet->setTriangles(triangle);

		triangleSet->update();

		vertices.clear();
		triangle.clear();
	}

	DEFINE_CLASS(SphereModel);
}