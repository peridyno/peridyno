#include "Turning.h"
#include "Topology/PointSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"



namespace dyno
{
	template<typename TDataType>
	TurningModel<TDataType>::TurningModel()
		: ParametricModel<TDataType>()
	{

		//this->varRow()->setRange(2, 50);
		this->varColumns()->setRange(3, 50);
		this->varRadius()->setRange(-10.0f, 10.0f);
		//this->varHeight()->setRange(0.001f, 10.0f);
		this->varEndSegment()->setRange(2, 39);

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());



		glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Vec3f(0.8, 0.52, 0.25));
		glModule->setVisible(true);
		this->stateTriangleSet()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);


		//auto wireframe = std::make_shared<GLWireframeVisualModule>();
		//this->stateTriangleSet()->connect(wireframe->inEdgeSet());
		//this->graphicsPipeline()->pushModule(wireframe);


	}

	template<typename TDataType>
	void TurningModel<TDataType>::resetStates()
	{
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		//auto radius = this->varRadius()->getData();
		//auto row = this->varRow()->getData();
		auto columns = this->varColumns()->getData();
		//auto height = this->varHeight()->getData();
		auto end_segment = this->varEndSegment()->getData();

		this->inPointSet()->getData().getPointSize();
		int pointsize = this->inPointSet()->getData().getPointSize();
		std::cout << "输入点个数： " << pointsize << std::endl;


		//TCylinder3D<Real> tube;
		//tube.row = row;
		//tube.columns = columns;
		//tube.radius = radius;
		//tube.height = height;
		//tube.end_segment = end_segment;

		//this->outCylinder()->setValue(tube);

		auto triangleSet = this->stateTriangleSet()->getDataPtr();

		Real PI = 3.1415926535;
		
		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> triangle;


		int columns_i = int(columns);
		int row_i = pointsize;

		uint counter = 0;
		Vec3f Location;
		Real angle = PI / 180 * 360 / columns_i;
		Real temp_angle = angle;
		
		//auto pointset = this->inPointSet()->getData().getPoints();
		PointSet<TDataType> s;
		s.copyFrom(this->inPointSet()->getData());
		//std::cout << "saaa" << std::endl;
		DArray<Coord> sa = s.getPoints();
		CArray<Coord> c_sa;
		c_sa.assign(sa);
	
		//std::cout << "sa" << sa << std::endl;
		//以下是侧面点的构建

		for (int k = 0; k < row_i; k++)
		{
			Real tempy = c_sa[k][1];
			Real radius = c_sa[k][0]+this->varRadius()->getData();

			for (int j = 0; j < columns_i; j++) {

				temp_angle = j * angle;

				Location = { sin(temp_angle) * radius , tempy ,cos(temp_angle) * radius };

				vertices.push_back(Location);
			}
		}

		//以下是底部及上部点的构建
		
		int pt_side_len = vertices.size();

		for (int i = 1; i < end_segment; i++)
		{
			float offset = i / (float(end_segment) - i);

			for (int p = 0; p < columns; p++)
			{
				Coord buttompt = { (vertices[p][0] + offset * 0) / (1 + offset), c_sa[0][1], (vertices[p][2] + offset * 0) / (1 + offset) };

				vertices.push_back(buttompt);
			}

		}

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


		//以下是底部圆心及上部圆心点的构建
		Real buttom = c_sa[0][1];
		Real height = c_sa[c_sa.size() - 1][1];
		vertices.push_back(Coord(0, buttom, 0));
		vertices.push_back(Coord(0, height, 0));

		//以下是侧面的构建
		for (int rowl = 0; rowl < row_i - 1; rowl++)
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

		//以下是底面和顶面的构建
		//侧面原有的点数，pt_side_len,

		int pt_len = vertices.size() - 2;
		int top_pt_len = vertices.size() - 2 - pt_side_len;
		int addnum = 0;

		for (int s = 0; s < end_segment; s++)
		{


		}
		for (int s = 0; s < end_segment; s++)  //内部循环遍历每一圈每一列
		{
			int temp = 0;
			//****************是否是外圈，是外圈使用四个点围成两个三角形*****************//
			if (s != end_segment - 1)
			{
				for (int i = 0; i < columns; i++)
				{
					//****************先判断是否是最外一圈，是的话与侧面序号相接*****************//
					if (s == 0)
					{
						temp = i;  //i为0-columns的序号，“+ x * (pt_side_len - columns)”作为侧面序号的变化量，最终得出侧面 上、下一圈的序号
						addnum = pt_side_len;
					}
					else
					{
						temp = pt_side_len + i + unsigned(s - 1) * columns;
						addnum = columns;
					}


					//****************是否是最后一列，是的话首尾序号相接，防止连接点换行*****************//
					if (i != columns - 1)
					{
						triangle.push_back(TopologyModule::Triangle(addnum + temp, temp + 1, temp));	//生成底面
						triangle.push_back(TopologyModule::Triangle(addnum + temp, addnum + temp + 1, temp + 1));
					}
					else
					{
						triangle.push_back(TopologyModule::Triangle(addnum + temp, temp - columns + 1, temp));	//生成底面最后一列

						if (s == 0)		triangle.push_back(TopologyModule::Triangle(addnum + temp, temp - columns + addnum + 1, temp - columns + 1));
						else			triangle.push_back(TopologyModule::Triangle(addnum + temp, temp + 1, temp - columns + 1));

					}

				}
			}
			//****************是否是最内圈，是最内圈使用周长连接圆心*****************//
			else
			{

				for (int z = 0; z < columns; z++)
				{
					temp = pt_side_len + z + unsigned(s - 1) * columns;
					if (z != columns - 1)
					{
						triangle.push_back(TopologyModule::Triangle(temp + 1, temp, pt_len));	//生成底面最内圈

					}
					else
					{
						triangle.push_back(TopologyModule::Triangle(temp - columns + 1, temp, pt_len));	//生成底面最内圈最后一个面

					}
					 
				}
			}

		}
		//*************************上部************************//

		for (int s = 0; s < end_segment; s++)  //内部循环遍历每一圈每一列
		{
			int temp = 0;
			//****************是否是外圈，是外圈使用四个点围成两个三角形*****************//
			if (s != end_segment - 1)
			{
				for (int i = 0; i < columns; i++)
				{
					//****************先判断是否是最外一圈，是的话与侧面序号相接*****************//
					if (s == 0)
					{
						temp = i + pt_side_len - columns;  //i为0-columns的序号，“+ x * (pt_side_len - columns)”作为侧面序号的变化量，最终得出侧面 上、下一圈的序号
						addnum = end_segment * columns;;
					}
					else
					{
						temp = pt_side_len + i + unsigned(s - 1) * columns + columns * (end_segment - 1);
						addnum = columns;
					}
					//****************是否是最后一列，是的话首尾序号相接，防止连接点换行*****************//
					if (i != columns - 1)
					{
						triangle.push_back(TopologyModule::Triangle(temp, temp + 1, addnum + temp));	//生成底面
						triangle.push_back(TopologyModule::Triangle(temp + 1, addnum + temp + 1, addnum + temp));
					}
					else
					{
						triangle.push_back(TopologyModule::Triangle(temp, temp - columns + 1, addnum + temp));	//生成底面最后一列

						if (s == 0)		triangle.push_back(TopologyModule::Triangle(temp - columns + 1, temp - columns + addnum + 1, addnum + temp));
						else			triangle.push_back(TopologyModule::Triangle(temp - columns + 1, temp + 1, addnum + temp));

					}

				}
			}
			//****************是否是最内圈，是最内圈使用周长连接圆心*****************//
			else
			{

				for (int z = 0; z < columns; z++)
				{
					temp = pt_side_len + z + unsigned(s - 1) * columns + columns * (end_segment - 1);
					if (z != columns - 1)
					{
						triangle.push_back(TopologyModule::Triangle(pt_len + 1, temp, temp + 1));	//生成底面最内圈

					}
					else
					{
						triangle.push_back(TopologyModule::Triangle(pt_len + 1, temp, temp - columns + 1));	//生成底面最内圈最后一个面

					}

				}
			}

		}


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
			vertices[i] =RV( vertices[i] * scale + RV( center ));
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




		triangleSet->setPoints(vertices);
		triangleSet->setTriangles(triangle);

		//triangleSet->updateEdges();
		//triangleSet->updateVertexNormal();


		triangleSet->update();



		vertices.clear();
		triangle.clear();
		

	}


	template<typename TDataType>
	void TurningModel<TDataType>::disableRender() {
		glModule->setVisible(false);
	};




	DEFINE_CLASS(TurningModel);
}