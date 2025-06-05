#include "SphereModel.h"

#include "Primitive/Primitive3D.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"

#include "Mapping/Extract.h"
#include "Subdivide.h"

namespace dyno
{
	template<typename TDataType>
	SphereModel<TDataType>::SphereModel()
		: BasicShape<TDataType>()
	{
		this->varRadius()->setRange(0.001f, 100.0f);

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		this->statePolygonSet()->setDataPtr(std::make_shared<PolygonSet<TDataType>>());
		
		this->varLatitude()->setRange(2, 10000);
		this->varLongitude()->setRange(3, 10000);
		this->varIcosahedronStep()->setRange(0,6);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&SphereModel<TDataType>::varChanged, this));


		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);
		this->varRadius()->attach(callback);
		this->varLatitude()->attach(callback);
		this->varLongitude()->attach(callback);
		this->varType()->attach(callback);

		auto tsRender = std::make_shared<GLSurfaceVisualModule>();
		this->stateTriangleSet()->connect(tsRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(tsRender);

		auto exES = std::make_shared<ExtractEdgeSetFromPolygonSet<TDataType>>();
		this->statePolygonSet()->connect(exES->inPolygonSet());
		this->graphicsPipeline()->pushModule(exES);

		auto esRender = std::make_shared<GLWireframeVisualModule>();
		esRender->varBaseColor()->setValue(Color(0, 0, 0));
		exES->outEdgeSet()->connect(esRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(esRender);

		this->stateTriangleSet()->promoteOuput();
		this->statePolygonSet()->promoteOuput();
	}

	template<typename TDataType>
	void SphereModel<TDataType>::resetStates()
	{
		varChanged();
	}

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

		Quat<Real> q = this->computeQuaternion();

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
	void SphereModel<TDataType>::standardSphere()
	{

		auto center = this->varLocation()->getValue();
		auto rot = this->varRotation()->getValue();
		auto scale = this->varScale()->getValue();

		auto radius = this->varRadius()->getValue();

		auto latitude = this->varLatitude()->getValue();
		auto longitude = this->varLongitude()->getValue();

		auto type = this->varType()->getValue();

		auto polySet = this->statePolygonSet()->getDataPtr();

		std::vector<Coord> vertices;


		Real deltaTheta = M_PI / latitude;
		Real deltaPhi = 2 * M_PI / longitude;

		//Setup vertices
		vertices.push_back(Coord(0, radius, 0));

		Real theta = 0;
		for (uint i = 0; i < latitude - 1; i++)
		{
			theta += deltaTheta;

			Real phi = 0;
			for (uint j = 0; j < longitude; j++)
			{
				phi += deltaPhi;

				Real y = radius * std::cos(theta);
				Real x = (std::sin(theta) * radius) * std::sin(phi);
				Real z = (std::sin(theta) * radius) * std::cos(phi);

				vertices.push_back(Coord(x, y, z));
			}
		}

		vertices.push_back(Coord(0, -radius, 0));

		//Setup polygon indices
		uint numOfPolygon = latitude * longitude;

		CArray<uint> counter(numOfPolygon);

		uint incre = 0;
		for (uint j = 0; j < longitude; j++)
		{
			counter[incre] = 3;
			incre++;
		}

		for (uint i = 0; i < latitude - 2; i++)
		{
			for (uint j = 0; j < longitude; j++)
			{
				counter[incre] = 4;
				incre++;
			}
		}

		for (uint j = 0; j < longitude; j++)
		{
			counter[incre] = 3;
			incre++;
		}

		CArrayList<uint> polygonIndices;
		polygonIndices.resize(counter);

		incre = 0;
		uint offset = 1;
		for (uint j = 0; j < longitude; j++)
		{
			auto& index = polygonIndices[incre];
			index.insert(0);
			index.insert(offset + j);
			index.insert(offset + (j + 1) % longitude);

			incre++;
		}

		for (uint i = 0; i < latitude - 2; i++)
		{
			for (uint j = 0; j < longitude; j++)
			{
				auto& index = polygonIndices[incre];
				index.insert(offset + j);
				index.insert(offset + j + longitude);
				index.insert(offset + (j + 1) % longitude + longitude);
				index.insert(offset + (j + 1) % longitude);

				incre++;
			}
			offset += longitude;
		}

		for (uint j = 0; j < longitude; j++)
		{
			auto& index = polygonIndices[incre];
			index.insert(offset + j);
			index.insert(vertices.size() - 1);
			index.insert(offset + (j + 1) % longitude);

			incre++;
		}

		//Apply transformation
		Quat<Real> q = this->computeQuaternion();

		auto RV = [&](const Coord& v)->Coord {
			return center + q.rotate(v - center);
		};

		int numpt = vertices.size();

		for (int i = 0; i < numpt; i++) {
			vertices[i] = RV(vertices[i] * scale + RV(center));
		}

		polySet->setPoints(vertices);
		polySet->setPolygons(polygonIndices);
		polySet->update();

		polygonIndices.clear();

		auto& ts = this->stateTriangleSet()->getData();
		polySet->turnIntoTriangleSet(ts);

		vertices.clear();
	}

	template<typename TDataType>
	void SphereModel<TDataType>::icosahedronSphere()
	{
		std::vector<Vec3f> vts;
		std::vector<TopologyModule::Triangle> trs;

		generateIcosahedron(vts, trs);
		float fixScale = this->varIcosahedronStep()->getValue() >= 2 ? 1.08 : 1;

		Quat<Real> q = this->computeQuaternion();
		auto center = this->varLocation()->getValue();
		auto scale = this->varScale()->getValue();

		auto RV = [&](const Coord& v)->Coord {
			return center + q.rotate(v - center);
		};

		for (int i = 0; i < vts.size(); i++) {
			vts[i] = RV(vts[i] * scale * fixScale + RV(center));
		}
		
		if (this->varIcosahedronStep()->getValue() >= 2) 
		{
			for (int i = 0; i < (int)this->varIcosahedronStep()->getValue() - 1; i++)
			{
				loopSubdivide(vts, trs);
			}
		}


		this->stateTriangleSet()->getDataPtr()->setPoints(vts);
		this->stateTriangleSet()->getDataPtr()->setTriangles(trs);
		this->stateTriangleSet()->getDataPtr()->update();

		this->statePolygonSet()->getDataPtr()->triangleSetToPolygonSet(this->stateTriangleSet()->getData());
		
		return;
	}



	template<typename TDataType>
	void SphereModel<TDataType>::varChanged()
	{
		auto center = this->varLocation()->getValue();
		auto rot = this->varRotation()->getValue();
		auto scale = this->varScale()->getValue();

		Real s;
		if (abs(scale.x - scale.y) < EPSILON) {
			s = scale.z;
		}
		else if (abs(scale.x - scale.z) < EPSILON){
			s = scale.y;
		}
		else
			s = scale.x;

		//To ensure all three components of varScale() have the same value
		this->varScale()->setValue(Coord(s), false);

		auto radius = this->varRadius()->getValue();

		//Setup a sphere primitive
		TSphere3D<Real> ball;
		ball.center = center;
		ball.radius = radius * s;
		this->outSphere()->setValue(ball);

		if (this->varType()->getDataPtr()->currentKey() == SphereType::Icosahedron)
		{
			icosahedronSphere();

			this->varLongitude()->setActive(false);
			this->varLatitude()->setActive(false);
			this->varIcosahedronStep()->setActive(true);
		}
		else {
			standardSphere();

			this->varLongitude()->setActive(true);
			this->varLatitude()->setActive(true);
			this->varIcosahedronStep()->setActive(false);
		}

	}


	template<typename TDataType>
	void SphereModel<TDataType>::generateIcosahedron(std::vector<Vec3f>& vertices, std::vector<TopologyModule::Triangle>& triangles) 
	{
		auto radius = this->varRadius()->getValue() * 2;

		if (this->varIcosahedronStep()->getValue() == 0)
		{
			float phi = (1.0 + std::sqrt(5.0)) / 2.0;

			vertices = {
				Vec3f(-1,  phi, 0), Vec3f(1,  phi, 0), Vec3f(-1, -phi, 0), Vec3f(1, -phi, 0),
				Vec3f(0, -1,  phi), Vec3f(0,  1,  phi), Vec3f(0, -1, -phi), Vec3f(0,  1, -phi),
				Vec3f(phi, 0, -1), Vec3f(phi, 0,  1), Vec3f(-phi, 0, -1), Vec3f(-phi, 0,  1)
			};

			triangles = {
				TopologyModule::Triangle(0, 11, 5), TopologyModule::Triangle(0, 5, 1), TopologyModule::Triangle(0, 1, 7), TopologyModule::Triangle(0, 7, 10), TopologyModule::Triangle(0, 10, 11),
				TopologyModule::Triangle(1, 5, 9), TopologyModule::Triangle(5, 11, 4), TopologyModule::Triangle(11, 10, 2), TopologyModule::Triangle(10, 7, 6), TopologyModule::Triangle(7, 1, 8),
				TopologyModule::Triangle(3, 9, 4), TopologyModule::Triangle(3, 4, 2), TopologyModule::Triangle(3, 2, 6), TopologyModule::Triangle(3, 6, 8), TopologyModule::Triangle(3, 8, 9),
				TopologyModule::Triangle(4, 9, 5), TopologyModule::Triangle(2, 4, 11), TopologyModule::Triangle(6, 2, 10), TopologyModule::Triangle(8, 6, 7), TopologyModule::Triangle(9, 8, 1)
			};

			for (auto& v : vertices) {
				v = v / M_PI * radius;
			}
		}
		else
		{
			vertices =
			{
				Vec3f(0,0, -0.5) * radius,
				Vec3f(0, 0.262865603, -0.425325394) * radius,
				Vec3f(0.249999985, 0.0812300518, -0.425325394) * radius,
				Vec3f(0, 0.44721368, -0.223606572) * radius,
				Vec3f(0.249999985, 0.344095677, -0.262865305) * radius,
				Vec3f(0.425325423, 0.138196841, -0.223606601) * radius,
				Vec3f(0.154508501, -0.212662712, -0.425325423) * radius,
				Vec3f(0.404508621, -0.131432697, -0.262865394) * radius,
				Vec3f(0.262865603, -0.361803472, -0.223606631) * radius,
				Vec3f(-0.154508501, -0.212662712, -0.425325423) * radius,
				Vec3f(0, -0.425325513, -0.262865365) * radius,
				Vec3f(-0.262865603, -0.361803472, -0.223606631) * radius,
				Vec3f(-0.249999985, 0.0812300518, -0.425325394) * radius,
				Vec3f(-0.404508621, -0.131432697, -0.262865394) * radius,
				Vec3f(-0.425325423, 0.138196841, -0.223606601) * radius,
				Vec3f(-0.249999985, 0.344095677, -0.262865305) * radius,
				Vec3f(-0.154508486, 0.4755283, 0) * radius,
				Vec3f(-0.404508412, 0.293892711, 0) * radius,
				Vec3f(-0.262865603, 0.361803472, 0.223606631) * radius,
				Vec3f(-0.5, 0, 0) * radius,
				Vec3f(-0.404508412, -0.293892711, 0) * radius,
				Vec3f(-0.425325423, -0.138196841, 0.223606601) * radius,
				Vec3f(-0.154508486, -0.4755283, 0) * radius,
				Vec3f(0.154508486, -0.4755283, 0) * radius,
				Vec3f(0, -0.44721368, 0.223606572) * radius,
				Vec3f(0.404508412, -0.293892711, 0) * radius,
				Vec3f(0.5, 0 ,0) * radius,
				Vec3f(0.425325423, -0.138196841, 0.223606601) * radius,
				Vec3f(0.404508412, 0.293892711, 0) * radius,
				Vec3f(0.154508486, 0.4755283, 0) * radius,
				Vec3f(0.262865603, 0.361803472 ,0.223606631) * radius,
				Vec3f(0, 0.425325513, 0.262865365) * radius,
				Vec3f(-0.404508621, 0.131432697, 0.262865394) * radius,
				Vec3f(-0.249999985, -0.344095677, 0.262865305) * radius,
				Vec3f(0.249999985, -0.344095677, 0.262865305) * radius,
				Vec3f(0.404508621, 0.131432697, 0.262865394) * radius,
				Vec3f(0, 0, 0.5) * radius,
				Vec3f(0.154508501, 0.212662712 ,0.425325423) * radius,
				Vec3f(-0.154508501, 0.212662712, 0.425325423) * radius,
				Vec3f(0.249999985, -0.0812300518, 0.425325394) * radius,
				Vec3f(0, -0.262865603, 0.425325394) * radius,
				Vec3f(-0.249999985, -0.0812300518, 0.425325394) * radius
			};

			triangles =
			{
				TopologyModule::Triangle(3, 1, 2),
				TopologyModule::Triangle(5,2 ,4),
				TopologyModule::Triangle(6, 3, 5),
				TopologyModule::Triangle(3, 2, 5),
				TopologyModule::Triangle(7, 1, 3),
				TopologyModule::Triangle(8, 3, 6),
				TopologyModule::Triangle(9, 7, 8),
				TopologyModule::Triangle(7, 3, 8),
				TopologyModule::Triangle(10, 1, 7),
				TopologyModule::Triangle(11, 7, 9),
				TopologyModule::Triangle(12, 10, 11),
				TopologyModule::Triangle(10, 7, 11),
				TopologyModule::Triangle(13, 1, 10),
				TopologyModule::Triangle(14, 10, 12),
				TopologyModule::Triangle(15, 13, 14),
				TopologyModule::Triangle(13, 10, 14),
				TopologyModule::Triangle(2 ,1, 13),
				TopologyModule::Triangle(16, 13, 15),
				TopologyModule::Triangle(4 ,2, 16),
				TopologyModule::Triangle(2 ,13, 16),
				TopologyModule::Triangle(17, 4, 16),
				TopologyModule::Triangle(18, 16, 15),
				TopologyModule::Triangle(19, 17, 18),
				TopologyModule::Triangle(17, 16, 18),
				TopologyModule::Triangle(20, 15, 14),
				TopologyModule::Triangle(21, 14, 12),
				TopologyModule::Triangle(22, 20, 21),
				TopologyModule::Triangle(20, 14, 21),
				TopologyModule::Triangle(23, 12, 11),
				TopologyModule::Triangle(24, 11, 9),
				TopologyModule::Triangle(25, 23, 24),
				TopologyModule::Triangle(23, 11, 24),
				TopologyModule::Triangle(26, 9, 8),
				TopologyModule::Triangle(27, 8, 6),
				TopologyModule::Triangle(28, 26, 27),
				TopologyModule::Triangle(26, 8, 27),
				TopologyModule::Triangle(29, 6, 5),
				TopologyModule::Triangle(30, 5, 4),
				TopologyModule::Triangle(31, 29, 30),
				TopologyModule::Triangle(29, 5, 30),
				TopologyModule::Triangle(30, 4, 17),
				TopologyModule::Triangle(32, 17, 19),
				TopologyModule::Triangle(31, 30, 32),
				TopologyModule::Triangle(30, 17, 32),
				TopologyModule::Triangle(18, 15, 20),
				TopologyModule::Triangle(33, 20, 22),
				TopologyModule::Triangle(19, 18, 33),
				TopologyModule::Triangle(18, 20, 33),
				TopologyModule::Triangle(21, 12, 23),
				TopologyModule::Triangle(34, 23, 25),
				TopologyModule::Triangle(22, 21, 34),
				TopologyModule::Triangle(21, 23, 34),
				TopologyModule::Triangle(24, 9, 26),
				TopologyModule::Triangle(35, 26, 28),
				TopologyModule::Triangle(25, 24, 35),
				TopologyModule::Triangle(24, 26, 35),
				TopologyModule::Triangle(27, 6, 29),
				TopologyModule::Triangle(36, 29, 31),
				TopologyModule::Triangle(28, 27, 36),
				TopologyModule::Triangle(27, 29, 36),
				TopologyModule::Triangle(39, 37, 38),
				TopologyModule::Triangle(32, 38, 31),
				TopologyModule::Triangle(19, 39, 32),
				TopologyModule::Triangle(39, 38, 32),
				TopologyModule::Triangle(38, 37, 40),
				TopologyModule::Triangle(36, 40, 28),
				TopologyModule::Triangle(31, 38, 36),
				TopologyModule::Triangle(38, 40, 36),
				TopologyModule::Triangle(40, 37, 41),
				TopologyModule::Triangle(35, 41, 25),
				TopologyModule::Triangle(28, 40, 35),
				TopologyModule::Triangle(40, 41, 35),
				TopologyModule::Triangle(41, 37, 42),
				TopologyModule::Triangle(34, 42, 22),
				TopologyModule::Triangle(25, 41, 34),
				TopologyModule::Triangle(41, 42, 34),
				TopologyModule::Triangle(42, 37, 39),
				TopologyModule::Triangle(33, 39, 19),
				TopologyModule::Triangle(22, 42, 33),
				TopologyModule::Triangle(42, 39, 33)
			};
			for (size_t i = 0; i < triangles.size(); i++)
			{
				triangles[i] = TopologyModule::Triangle(triangles[i][0] - 1, triangles[i][1] - 1, triangles[i][2] - 1);
			}

		}
	}




	DEFINE_CLASS(SphereModel);
}