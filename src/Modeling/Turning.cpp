#include "Turning.h"
#include "Topology/PointSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

#include "Mapping/QuadSetToTriangleSet.h"
#include "Mapping/Extract.h"

namespace dyno
{
	template<typename TDataType>
	TurningModel<TDataType>::TurningModel()
		: ParametricModel<TDataType>()
	{
		this->varColumns()->setRange(3, 50);
		this->varRadius()->setRange(-10.0f, 10.0f);
		this->varEndSegment()->setRange(0, 39);

		this->statePolygonSet()->setDataPtr(std::make_shared<PolygonSet<TDataType>>());
		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		this->inPointSet()->tagOptional(true);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&TurningModel<TDataType>::varChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varColumns()->attach(callback);
		this->varEndSegment()->attach(callback);
		this->varRadius()->attach(callback);
		this->varReverseNormal()->attach(callback);
		this->varUseRamp()->attach(callback);
		this->varCurve()->attach(callback);

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

		//this->statePolygonSet()->promoteOuput();
		this->stateTriangleSet()->promoteOuput();
		this->statePolygonSet()->promoteOuput();

		//Do not export the node
		this->allowExported(false);


		this->statePolygonSet()->promoteOuput();
		this->stateTriangleSet()->promoteOuput();
	}

	template<typename TDataType>
	void TurningModel<TDataType>::resetStates()
	{
		this->varChanged();
	}


	template<typename TDataType>
	void TurningModel<TDataType>::varChanged()
	{
		int pointsize = 0;;
		if (!this->inPointSet()->isEmpty())
		{
			pointsize = this->inPointSet()->getData().getPointSize();
		}

		auto useRamp = this->varUseRamp()->getValue();
		auto Ramp = this->varCurve()->getValue();
		auto floatCoordArray = Ramp.FinalCoord;


		PointSet<TDataType> s;
		if (!this->inPointSet()->isEmpty())
		{
			s.copyFrom(this->inPointSet()->getData());
		}
		DArray<Coord> sa = s.getPoints();
		CArray<Coord> c_sa;
		PointSet<TDataType> pointSet;

		if (useRamp)
		{
			std::vector<Coord> vertices;
			std::vector<TopologyModule::Triangle> triangle;
			pointsize = floatCoordArray.size();

			Coord Location;
			if (pointsize != 0)
			{
				for (size_t i = 0; i < pointsize; i++)
				{
					Location = Coord(floatCoordArray[i].x, floatCoordArray[i].y, 0);

					vertices.push_back(Location);
				}
			}
			pointSet.setPoints(vertices);
			c_sa.assign(pointSet.getPoints());
		}
		else
		{
			c_sa.assign(sa);
		}
	

		auto center = this->varLocation()->getValue();
		auto rot = this->varRotation()->getValue();
		auto scale = this->varScale()->getValue();

		auto radius = this->varRadius()->getValue();
		auto row = pointsize - 1;
		auto columns = this->varColumns()->getValue();
		auto end_segment = this->varEndSegment()->getValue();

		auto triangleSet = this->stateTriangleSet()->getDataPtr();
		auto polySet = this->statePolygonSet()->getDataPtr();

		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> triangle;


		Coord Location;
		Real angle = M_PI / 180 * 360 / columns;
		Real temp_angle = angle;

		//Side Point
		if (!this->inPointSet()->isEmpty() | (useRamp && pointsize > 0))
		{
			for (int k = 0; k <= row; k++)
			{
				Real tempy = c_sa[k][1];
				Real radius = c_sa[k][0] + this->varRadius()->getData();

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

					Coord toppt = { (vertices[top_start][0] + offset * 0) / (1 + offset),  c_sa[c_sa.size() - 1][1], (vertices[top_start][2] + offset * 0) / (1 + offset) };

					vertices.push_back(toppt);
				}

			}

			for (int i = 1; i < end_segment; i++)
			{
				float offset = i / (float(end_segment) - i);

				for (int p = 0; p < columns; p++)
				{
					Coord buttompt = { (vertices[p][0] + offset * 0) / (1 + offset), c_sa[0][1], (vertices[p][2] + offset * 0) / (1 + offset) };

					vertices.push_back(buttompt);
				}

			}

			vertices.push_back(Coord(0, c_sa[0][1], 0));
			uint buttomCenter = vertices.size() - 1;
			vertices.push_back(Coord(0, c_sa[c_sa.size() - 1][1], 0));
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
			for (uint i = 0; i < columns; i++)
			{
				for (uint j = 0; j < row; j++)
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

	}


	DEFINE_CLASS(TurningModel);
}