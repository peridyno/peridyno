#include "ConeModel.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"
#include "Mapping/Extract.h"


namespace dyno
{
	template<typename TDataType>
	ConeModel<TDataType>::ConeModel()
		: BasicShape<TDataType>()
	{
		this->varRow()->setRange(1, 50);
		this->varColumns()->setRange(3, 50);
		this->varRadius()->setRange(0.001f, 10.0f);
		this->varHeight()->setRange(0.001f, 10.0f);
		this->varEndSegment()->setRange(0,500);

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->statePolygonSet()->setDataPtr(std::make_shared<PolygonSet<TDataType>>());

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&ConeModel<TDataType>::varChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varColumns()->attach(callback);
		this->varRow()->attach(callback);
		this->varRadius()->attach(callback);
		this->varHeight()->attach(callback);

		auto tsRender = std::make_shared<GLSurfaceVisualModule>();
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

		this->statePolygonSet()->promoteOuput();
		this->stateTriangleSet()->promoteOuput();
	}

	template<typename TDataType>
	void ConeModel<TDataType>::resetStates()
	{
		varChanged();
	}

	template<typename TDataType>
	void ConeModel<TDataType>::varChanged()
	{
		auto center = this->varLocation()->getValue();
		auto rot = this->varRotation()->getValue();
		auto scale = this->varScale()->getValue();

		auto radius = this->varRadius()->getValue();
		auto row = this->varRow()->getValue();
		auto columns = this->varColumns()->getValue();
		auto height = this->varHeight()->getValue();
		auto end_segment = this->varEndSegment()->getValue();


		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> triangle;

		Coord Location;
		Real angle = M_PI / 180 * 360 / columns;
		Real temp_angle = angle;
		Real x, y, z;


		//Side
		for (int k = row - 1; k >= 0; k--)
		{
			Real ratio = float(k) / float(row);
			Real tempy = height * ratio;
			Real tempRadius = radius * (1 - ratio);

			for (int j = 0; j < columns; j++) {

				temp_angle = j * angle;

				Location = { sin(temp_angle) * tempRadius , tempy ,cos(temp_angle) * tempRadius };

				vertices.push_back(Location);
			}
		}

		for (int i = 1; i < end_segment; i++)
		{
			Real ratio = float(i) / float(end_segment);
			Real tempy = 0;
			Real tempRadius = radius * (1 - ratio);

			for (int p = 0; p < columns; p++)
			{
				temp_angle = p * angle;

				Coord buttompt = { sin(temp_angle) * tempRadius , tempy ,cos(temp_angle) * tempRadius };

				vertices.push_back(buttompt);
			}
		}


		vertices.push_back(Coord(0, height, 0));
		uint topCenter = vertices.size() - 1;
		uint buttomCenter = vertices.size();

		//if (end_segment > 0) 
			vertices.push_back(Coord(0, 0, 0));
		
		uint realRow = (int(row) - 1 > 0) ? int(row) - 1 : 0;
		uint realEndsegment = (int(end_segment) - 1 > 0) ? int(end_segment) - 1 : 0;

		uint numOfPolygon = columns * (row + end_segment);
		CArray<uint> counter(numOfPolygon);
		uint incre = 0;

		uint QuadNum = columns * realRow + columns * realEndsegment;
		uint TriangleNum;
		if(end_segment>0)
			TriangleNum= uint(columns * 2);
		else
			TriangleNum = uint(columns);




		for (uint j = 0; j < QuadNum; j++)
		{
			counter[incre] = 4;
			incre++;
		}

		for (uint j = 0; j < TriangleNum; j++)
		{
			counter[incre] = 3;
			incre++;
		}
		CArrayList<uint> polygonIndices;
		polygonIndices.resize(counter);



		//Quad
		incre = 0;

		

		for (uint i = 0; i < columns; i++)
		{
			for (uint j = 0; j < realRow + realEndsegment; j++)
			{
				auto& index = polygonIndices[incre];

				uint p1 = i + j * columns;
				uint p2 = (i + 1) % columns + j * columns;
				uint p3 = (i + 1) % columns + j * columns + columns;
				uint p4 = i + j * columns + columns;

				index.insert(p4);
				index.insert(p3);
				index.insert(p2);
				index.insert(p1);


				incre++;
			}
		}

		uint sidePtNum = incre;

		if (end_segment > 0)
		{
			//TriangleButtom
			for (uint i = 0; i < columns; i++)
			{
				auto& index = polygonIndices[incre];
				uint p1 = sidePtNum + i;
				uint p2 = sidePtNum + (i + 1) % columns;
				uint p3 = buttomCenter;

				index.insert(p3);
				index.insert(p2);
				index.insert(p1);


				incre++;
			}
		}
		//TriangleTop
		for (uint i = 0; i < columns; i++)
		{
			auto& index = polygonIndices[incre];
			uint p1 = i;
			uint p2 = (i + 1) % columns;
			uint p3 = topCenter;
			index.insert(p1);
			index.insert(p2);
			index.insert(p3);

			incre++;
		}

		//Transform
		Quat<Real> q = computeQuaternion();

		q.normalize();

		auto RV = [&](const Coord& v)->Coord {
			return center + q.rotate(v - center);
		};

		int numpt = vertices.size();

		for (int i = 0; i < numpt; i++)
		{
			vertices[i][1] -= 1 * height / 2;
			vertices[i] = RV(vertices[i] * scale + RV(center));
		}

		auto polySet = this->statePolygonSet()->getDataPtr();

		polySet->setPoints(vertices);
		polySet->setPolygons(polygonIndices);
		polySet->update();

		auto& ts = this->stateTriangleSet()->getData();
		polySet->turnIntoTriangleSet(ts);
		
		auto triangleSet = this->stateTriangleSet()->getDataPtr();


		polygonIndices.clear();
		vertices.clear();

		//Setup the geometric primitive
		TCone3D<Real> cone;
		cone.center = center;
		cone.height = height;
		cone.radius = radius;
		cone.scale = scale;
		cone.rotation = q;
		this->outCone()->setValue(cone);
	}

	template<typename TDataType>
	NBoundingBox ConeModel<TDataType>:: boundingBox() 
	{
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		auto radius = this->varRadius()->getData();
		auto height = this->varHeight()->getData();

		Coord length(Coord(radius, height, radius));
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



	DEFINE_CLASS(ConeModel);
}