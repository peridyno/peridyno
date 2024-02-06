#include "ConeModel.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"
#include "Mapping/Extract.h"


namespace dyno
{
	template<typename TDataType>
	ConeModel<TDataType>::ConeModel()
		: ParametricModel<TDataType>()
	{
		this->varRow()->setRange(2, 50);
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
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		auto radius = this->varRadius()->getData();
		auto row = this->varRow()->getData();
		auto columns = this->varColumns()->getData();
		auto height = this->varHeight()->getData();
		auto end_segment = this->varEndSegment()->getData();



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

		vertices.push_back(Coord(0, 0, 0));
		uint buttomCenter = vertices.size() - 1;
		uint topCenter = vertices.size();
		vertices.push_back(Coord(0, height, 0));
		
		uint numOfPolygon = columns * (row + end_segment);
		CArray<uint> counter(numOfPolygon);
		uint incre = 0;
		uint QuadNum = columns * uint(int(row) - 1)+ columns * uint(int(end_segment) - 1);
		for (uint j = 0; j < QuadNum; j++)
		{
			counter[incre] = 4;
			incre++;
		}

		uint TriangleNum = uint( columns * 2);

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
			for (uint j = 0; j < (row + end_segment -2); j++)
			{
				auto& index = polygonIndices[incre];

				uint p1 = i + j * columns;
				uint p2 = (i + 1) % columns + j * columns;
				uint p3 = (i + 1) % columns + j * columns + columns;
				uint p4 = i + j * columns + columns;

				index.insert(p1);
				index.insert(p2);
				index.insert(p3);
				index.insert(p4);

				incre++;
			}
		}

		uint sidePtNum = incre;

		//TriangleTop
		for (uint i = 0; i < columns; i++)
		{
			auto& index = polygonIndices[incre];
			uint p1 = sidePtNum + i;
			uint p2 = sidePtNum + (i + 1) % columns;
			uint p3 = buttomCenter;

			index.insert(p1);
			index.insert(p2);
			index.insert(p3);

			incre++;
		}

		//TriangleButtom
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
			vertices[i][1] -= 1 * height / 3;
			vertices[i] = RV(vertices[i] * scale + RV(center));
		}

		auto polySet = this->statePolygonSet()->getDataPtr();

		polySet->setPoints(vertices);
		polySet->setPolygons(polygonIndices);
		polySet->update();

		auto& ts = this->stateTriangleSet()->getData();
		polySet->turnIntoTriangleSet(ts);
		
		auto triangleSet = this->stateTriangleSet()->getDataPtr();

		std::cout<< triangleSet->getTriangles().size()<<std::endl;
		//std::cout << polySet-> << std::endl;

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

	DEFINE_CLASS(ConeModel);
}