#include "CylinderModel.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

#include "Mapping/Extract.h"


namespace dyno
{
	template<typename TDataType>
	CylinderModel<TDataType>::CylinderModel()
		: ParametricModel<TDataType>()
	{

		this->varRow()->setRange(1, 500);
		this->varColumns()->setRange(3, 500);
		this->varRadius()->setRange(0.001f, 20.0f);
		this->varHeight()->setRange(0.001f, 20.0f);
		this->varEndSegment()->setRange(0, 500);

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->statePolygonSet()->setDataPtr(std::make_shared<PolygonSet<TDataType>>());

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&CylinderModel<TDataType>::varChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varColumns()->attach(callback);
		this->varEndSegment()->attach(callback);
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
	void CylinderModel<TDataType>::resetStates()
	{
		varChanged();
	}

	template<typename TDataType>
	void CylinderModel<TDataType>::varChanged() 
	{
		auto center = this->varLocation()->getValue();
		auto rot = this->varRotation()->getValue();
		auto scale = this->varScale()->getValue();

		auto radius = this->varRadius()->getValue();
		auto row = this->varRow()->getValue();
		auto columns = this->varColumns()->getValue();
		auto height = this->varHeight()->getValue();
		auto end_segment = this->varEndSegment()->getValue();

		auto triangleSet = this->stateTriangleSet()->getDataPtr();
		auto polySet = this->statePolygonSet()->getDataPtr();

		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> triangle;




		Coord Location;
		Real angle = M_PI / 180 * 360 / columns;
		Real temp_angle = angle;

		//Side Point
		for (int k = 0; k <= row; k++)
		{
			Real tempy = height / row * k;

			for (int j = 0; j < columns; j++) {

				temp_angle = j * angle;

				Location = { sin(temp_angle) * radius , tempy ,cos(temp_angle) * radius };

				vertices.push_back(Location);
			}
		}

		//Top_Buttom Point

		int pt_side_len = vertices.size();

		for (int i = 1; i < end_segment; i++)
		{
			float offset = i / (float(end_segment) - i);

			for (int p = 0; p < columns; p++)
			{
				int top_start = pt_side_len - columns + p;

				Coord toppt = { (vertices[top_start][0] + offset * 0) / (1 + offset), (vertices[top_start][1] + offset * height) / (1 + offset), (vertices[top_start][2] + offset * 0) / (1 + offset) };

				vertices.push_back(toppt);
			}

		}

		for (int i = 1; i < end_segment; i++)
		{
			float offset = i / (float(end_segment) - i);

			for (int p = 0; p < columns; p++)
			{
				Coord buttompt = { (vertices[p][0] + offset * 0) / (1 + offset), (vertices[p][1] + offset * 0) / (1 + offset), (vertices[p][2] + offset * 0) / (1 + offset) };

				vertices.push_back(buttompt);
			}

		}

		vertices.push_back(Coord(0, 0, 0));
		uint buttomCenter = vertices.size() - 1;
		vertices.push_back(Coord(0, height, 0));
		uint topCenter = vertices.size() - 1;


		uint numOfPolygon = columns * row + 2 * columns * end_segment;
		CArray<uint> counter(numOfPolygon);

		uint incre = 0;
		uint endQuadNum = ((int(end_segment) - 1) * int(columns) < 0 ? 0 : (int(end_segment) - 1) * int(columns));

		uint QuadNum = columns * row + 2 * endQuadNum;
		for (uint j = 0; j < QuadNum; j++)
		{
			counter[incre] = 4;
			incre++;
		}

		uint TriangleNum = (int(end_segment) - 1) < 0 ? 0 : columns * 2;
		for (uint j = 0; j < TriangleNum; j++)
		{
			counter[incre] = 3;
			incre++;
		}

		CArrayList<uint> polygonIndices;
		polygonIndices.resize(counter);


		//side;
		incre = 0;
		for (uint i = 0; i < columns ; i++)
		{
			for (uint j = 0; j < row ; j++)
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

		//Top
		if (end_segment > 0) 
		{
			uint sidePtNum = columns * row;
			for (uint i = 0; i < columns; i++)
			{
				for (uint j = 0; j < end_segment - 1; j++)
				{
					auto& index = polygonIndices[incre];

					uint p1 = i + j * columns + (sidePtNum);
					uint p2 = (i + 1) % columns + j * columns + (sidePtNum);
					uint p3 = (i + 1) % columns + j * columns + columns + (sidePtNum);
					uint p4 = i + j * columns + columns + (sidePtNum);


					index.insert(p1);
					index.insert(p2);
					index.insert(p3);
					index.insert(p4);

					incre++;
				}
			}

			//Buttom
			uint sideTopPtNum = incre;
			for (uint i = 0; i < columns; i++)
			{
				for (uint j = 0; j < end_segment - 1; j++)
				{
					auto& index = polygonIndices[incre];

					uint temp = sideTopPtNum;
					if (j == 0)
					{
						temp = 0;
					}

					uint p1 = i + j * columns + (temp);
					uint p2 = (i + 1) % columns + j * columns + (temp);
					uint p3 = (i + 1) % columns + j * columns + columns + (sideTopPtNum);
					uint p4 = i + j * columns + columns + (sideTopPtNum);


					index.insert(p1);
					index.insert(p2);
					index.insert(p3);
					index.insert(p4);

					incre++;
				}
			}
			uint buttomPtNum = incre;

			//TriangleTop
			for (uint i = 0; i < columns; i++)
			{
				auto& index = polygonIndices[incre];
				uint p1 = sideTopPtNum + i;
				uint p2 = sideTopPtNum + (i + 1) % columns;
				uint p3 = topCenter;

				index.insert(p1);
				index.insert(p2);
				index.insert(p3);

				incre++;
			}

			//TriangleButtom
			if (end_segment == 1) 
			{
				buttomPtNum = 0;
			}
			for (uint i = 0; i < columns; i++)
			{
				auto& index = polygonIndices[incre];
				uint p1 = buttomPtNum + i;
				uint p2 = buttomPtNum + (i + 1) % columns;
				uint p3 = buttomCenter;

				index.insert(p1);
				index.insert(p2);
				index.insert(p3);

				incre++;
			}
		}
		


		//TransformModel

		Quat<Real> q = computeQuaternion();

		q.normalize();

		auto RV = [&](const Coord& v)->Coord {
			return center + q.rotate(v - center);
		};

		int numpt = vertices.size();

		for (int i = 0; i < numpt; i++)
		{
			vertices[i][1] -= height / 2;
			vertices[i] = RV(vertices[i] * scale + RV(center));
		}

		TCylinder3D<Real> cylinder;
		cylinder.center = center;
		cylinder.height = height;
		cylinder.radius = radius;
		cylinder.scale = scale;
		cylinder.rotation = q;
		this->outCylinder()->setValue(cylinder);


		polySet->setPoints(vertices);
		polySet->setPolygons(polygonIndices);
		polySet->update();

		polygonIndices.clear();

		auto& ts = this->stateTriangleSet()->getData();
		polySet->turnIntoTriangleSet(ts);

		vertices.clear();




	}

	template<typename TDataType>
	NBoundingBox CylinderModel<TDataType>::boundingBox()
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

	DEFINE_CLASS(CylinderModel);
}