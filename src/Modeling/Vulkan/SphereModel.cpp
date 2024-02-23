#include "SphereModel.h"

#include "Primitive/Primitive3D.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"

#include "Mapping/Extract.h"

namespace dyno
{
	template<typename TDataType>
	SphereModel<TDataType>::SphereModel()
		: ParametricModel<TDataType>()
	{
		this->varRadius()->setRange(0.001f, 100.0f);

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		this->statePolygonSet()->setDataPtr(std::make_shared<PolygonSet<TDataType>>());
		
		this->varLatitude()->setRange(2, 10000);
		this->varLongitude()->setRange(3, 10000);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&SphereModel<TDataType>::varChanged, this));


		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);
		this->varRadius()->attach(callback);
		this->varLatitude()->attach(callback);
		this->varLongitude()->attach(callback);
		this->varType()->attach(callback);

		auto tsRender = std::make_shared<GLSurfaceVisualModule>();
		tsRender->setColor(Color(0.8f, 0.52f, 0.25f));
		tsRender->setVisible(true);
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

		Quat<Real> q = computeQuaternion();

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
		auto center = this->varLocation()->getValue();
		auto rot = this->varRotation()->getValue();
		auto scale = this->varScale()->getValue();

		auto radius = this->varRadius()->getValue();

		auto latitude = this->varLatitude()->getValue();
		auto longitude = this->varLongitude()->getValue();

		auto type = this->varType()->getValue();

		//Setup a sphere primitive
		TSphere3D<Real> ball;
		ball.center = center;
		ball.radius = radius;
		this->outSphere()->setValue(ball);

		auto polySet = this->statePolygonSet()->getDataPtr();

		std::vector<Coord> vertices;

		if (type == SphereType::Standard)
		{
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
			Quat<Real> q = computeQuaternion();

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
		}
		else if(type == SphereType::Icosahedron)
		{
			//TODO: implementation
		}

		auto& ts = this->stateTriangleSet()->getData();
		polySet->turnIntoTriangleSet(ts);

		vertices.clear();
	}


	DEFINE_CLASS(SphereModel);
}