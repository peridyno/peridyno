#include "Extrude.h"
#include "Topology/PointSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"
#include "EarClipper.h"

namespace dyno
{
	template<typename TDataType>
	ExtrudeModel<TDataType>::ExtrudeModel()
		: ParametricModel<TDataType>()
	{
		this->varHeight()->setRange(0.001f, 10.0f);

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		auto glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Color(0.8f, 0.52f, 0.25f));
		glModule->setVisible(true);
		this->stateTriangleSet()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);

		auto glModule2 = std::make_shared<GLPointVisualModule>();
		glModule2->setColor(Color(1.0f, 1.0f, 1.0f));
		glModule2->varPointSize()->setValue(0.01);
		this->stateTriangleSet()->connect(glModule2->inPointSet());
		this->graphicsPipeline()->pushModule(glModule2);

		auto glModule3 = std::make_shared<GLWireframeVisualModule>();
		glModule3->setColor(Color(1.0f, 1.0f, 1.0f));
		this->stateTriangleSet()->connect(glModule3->inEdgeSet());
		this->graphicsPipeline()->pushModule(glModule3);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&ExtrudeModel<TDataType>::varChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varRow()->attach(callback);
		this->varHeight()->attach(callback);
		this->varReverseNormal()->attach(callback);
		this->varCurve()->attach(callback);


	}
	template<typename TDataType>
	void ExtrudeModel<TDataType>::resetStates()
	{
		varChanged();
	}

	template<typename TDataType>
	void ExtrudeModel<TDataType>::varChanged()
	{
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		enum PointMode
		{
			UseInput = 0,
			UseCurve = 1,
		};

		PointMode mPointMode = PointMode::UseCurve;

		int columns_i = 0;
		if (!this->inPointSet()->isEmpty()) 
		{
			columns_i = this->inPointSet()->getData().getPointSize();
			mPointMode = PointMode::UseInput;
		}
		else if (this->varCurve()->getValue().getPointSize()) 
		{
			columns_i = this->varCurve()->getValue().getPointSize();
			mPointMode = PointMode::UseCurve;
		}
		else
		{
			columns_i = 0;
		}
		
		if (columns_i >= 3)
		{
		//以下是侧面点的构建

			Real HeightValue = this->varHeight()->getData();
			Real RowValue = this->varRow()->getData();
			Real tempRow = 0;

			std::vector<Coord> vertices;
			CArray<Coord> capPoint;

			for (int i = 0; i <= RowValue; i++) 
			{
				Real tempy = HeightValue * i / RowValue;
				Vec3f position;

				if (mPointMode == PointMode::UseInput)
				{
					DArray<Coord> sa = this->inPointSet()->getData().getPoints();
					CArray<Coord> capPoint;
					capPoint.assign(capPoint);

					for (int k = 0; k < columns_i; k++)
					{
						position = { capPoint[k][0] , capPoint[k][1] + tempy ,capPoint[k][2] };

						vertices.push_back(position);
					}
				}
				else if (mPointMode == PointMode::UseCurve)
				{
					for (int k = 0; k < columns_i; k++)
					{
						auto curvePoint = this->varCurve()->getValue().getPoints();

						position = { float(curvePoint[k].x) , float(tempy) ,float(curvePoint[k].y) };
						vertices.push_back(position);

						if (i == 0 )
						{
							capPoint.pushBack(position);
						}
					}

				}
			}




			//以下是底部及上部点的构建
			std::vector<TopologyModule::Triangle> triangle;

			int pt_side_len = vertices.size();

			//以下是侧面的构建
			for (int rowl = 0; rowl <= RowValue - 1; rowl++)
			{

				for (int faceid = 0; faceid < columns_i; faceid++)
				{
					if (faceid != columns_i - 1)
					{
						triangle.push_back(TopologyModule::Triangle(columns_i + faceid + rowl * columns_i, 0 + faceid + rowl * columns_i, 1 + faceid + rowl * columns_i));
						triangle.push_back(TopologyModule::Triangle(columns_i + 1 + faceid + rowl * columns_i, columns_i + faceid + rowl * columns_i, 1 + faceid + rowl * columns_i));
					}
					else
					{
						triangle.push_back(TopologyModule::Triangle(1 + 2 * faceid + rowl * columns_i, 0 + faceid + rowl * columns_i, 0 + rowl * columns_i));
						triangle.push_back(TopologyModule::Triangle(1 + faceid + rowl * columns_i, 1 + 2 * faceid + rowl * columns_i, 0 + rowl * columns_i));
					}

				}
			}

			int pt_len = vertices.size() - 2;
			int top_pt_len = vertices.size() - 2 - pt_side_len;
			int addnum = 0;



			//transform


			Quat<Real> q = computeQuaternion();

			q.normalize();

			auto RV = [&](const Coord& v)->Coord {
				return center + q.rotate(v - center);
			};

			int numpt = vertices.size();

			for (int i = 0; i < numpt; i++)
			{
				//vertices[i][1] -= height / 2;
				vertices[i] = RV(vertices[i] * scale + RV(center));
			}



			if (this->varReverseNormal()->getData() == true)
			{
				int trinum = triangle.size();
				for (int i = 0; i < trinum; i++)
				{
					int temp;
					temp = triangle[i][0];
					triangle[i][0] = triangle[i][2];
					triangle[i][2] = temp;
				}
			}


			EarClipper<DataType3f> sab;
			std::vector<TopologyModule::Triangle> triangleCap;

			sab.polyClip(capPoint, triangleCap);
			int addnum2 = vertices.size() - capPoint.size();

			
			for (int i = 0; i < triangleCap.size(); i++)
			{
				triangle.push_back(triangleCap[i]);
				triangle.push_back(TopologyModule::Triangle(triangleCap[i][0] + addnum2, triangleCap[i][1] + addnum2, triangleCap[i][2] + addnum2));
			}

			auto triangleSet = this->stateTriangleSet()->getDataPtr();

			triangleSet->setPoints(vertices);
			triangleSet->setTriangles(triangle);

			triangleSet->update();

			vertices.clear();
			triangle.clear();
		}
		
	}


	DEFINE_CLASS(ExtrudeModel);
}