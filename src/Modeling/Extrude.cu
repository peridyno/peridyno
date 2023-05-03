#include "Extrude.h"
#include "Topology/PointSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"




namespace dyno
{
	template<typename TDataType>
	ExtrudeModel<TDataType>::ExtrudeModel()
		: ParametricModel<TDataType>()
	{

		//this->varRow()->setRange(2, 50);
		//this->varColumns()->setRange(3, 50);
		//this->varRadius()->setRange(-10.0f, 10.0f);
		//this->varHeight()->setRange(0.001f, 10.0f);
		//this->varEndSegment()->setRange(2, 39);

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());



		glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Color(0.8f, 0.52f, 0.25f));
		glModule->setVisible(true);
		this->stateTriangleSet()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);

		auto glModule2 = std::make_shared<GLPointVisualModule>();
		glModule2->setColor(Color(1, 0.1, 0.1));
		//glModule2->setVisible(false);
		glModule2->varPointSize()->setValue(0.01);
		this->stateTriangleSet()->connect(glModule2->inPointSet());
		this->graphicsPipeline()->pushModule(glModule2);

		auto glModule3 = std::make_shared<GLWireframeVisualModule>();
		glModule3->setColor(Color(0, 0, 0));
		//glModule3->setVisible(false);
		this->stateTriangleSet()->connect(glModule3->inEdgeSet());
		this->graphicsPipeline()->pushModule(glModule3);



	}

	template<typename TDataType>
	void ExtrudeModel<TDataType>::resetStates()
	{
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		//auto radius = this->varRadius()->getData();
		//auto row = this->varRow()->getData();
		auto columns = this->inPointSet()->getData().getPointSize();
		//auto height = this->varHeight()->getData();
		//auto end_segment = this->varEndSegment()->getData();

		int pointsize = this->inPointSet()->getData().getPointSize();
		std::cout << "输入点个数： " << pointsize << std::endl;


		Real PI = 3.1415926535;

		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> triangle;


		int columns_i = int(columns);
		int ptn = pointsize;

		uint counter = 0;
		Vec3f Location;

		//auto pointset = this->inPointSet()->getData().getPoints();
		PointSet<TDataType> s;
		s.copyFrom(this->inPointSet()->getData());
		//std::cout << "saaa" << std::endl;
		DArray<Coord> sa = s.getPoints();
		CArray<Coord> c_sa;
		c_sa.assign(sa);

		//std::cout << "sa" << sa << std::endl;
		//以下是侧面点的构建

		Real HeightValue = this->varHeight()->getData();
		Real RowValue = this->varRow()->getData();
		Real tempRow = 0;
		for (int i = 0; i <= RowValue; i++) {

			Real tempy = HeightValue * i / RowValue;

			for (int k = 0; k < ptn; k++)
			{
				Location = { c_sa[k][0] , c_sa[k][1] + tempy ,c_sa[k][2] };

				vertices.push_back(Location);	
			}
		}
		//以下是底部及上部点的构建

		int pt_side_len = vertices.size();


		//以下是底部圆心及上部圆心点的构建
		Real buttom = c_sa[0][1];
		Real height = this->varHeight()->getData();
		vertices.push_back(Coord(0, buttom, 0));
		vertices.push_back(Coord(0, height, 0));

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
		//以下是底面和顶面的构建

		for (int i = 0; i < ptn; i++) 
		{
			if (i != columns - 1) 
			{
				triangle.push_back(TopologyModule::Triangle(i, i + 1, pt_side_len));
				triangle.push_back(TopologyModule::Triangle(columns * RowValue + i, columns * RowValue + 1 + i, pt_side_len + 1));
			}
			else 
			{
				triangle.push_back(TopologyModule::Triangle(i, 0, pt_side_len));
				triangle.push_back(TopologyModule::Triangle(columns * RowValue + i, columns * RowValue, pt_side_len + 1));

			}
		}
		//以下是底面和顶面的构建
		
		//侧面原有的点数，pt_side_len,

		int pt_len = vertices.size() - 2;
		int top_pt_len = vertices.size() - 2 - pt_side_len;
		int addnum = 0;



		//变换

		Quat<Real> q = Quat<Real>(M_PI * rot[0] / 180, Coord(1, 0, 0))
			* Quat<Real>(M_PI * rot[1] / 180, Coord(0, 1, 0))
			* Quat<Real>(M_PI * rot[2] / 180, Coord(0, 0, 1));

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



		auto triangleSet = this->stateTriangleSet()->getDataPtr();

		triangleSet->setPoints(vertices);
		triangleSet->setTriangles(triangle);


		triangleSet->update();



		vertices.clear();
		triangle.clear();


	}


	template<typename TDataType>
	void ExtrudeModel<TDataType>::disableRender() {
		glModule->setVisible(false);
	};




	DEFINE_CLASS(ExtrudeModel);
}