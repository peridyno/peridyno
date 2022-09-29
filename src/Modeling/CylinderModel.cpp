#include "CylinderModel.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"



namespace dyno
{
	template<typename TDataType>
	CylinderModel<TDataType>::CylinderModel()
		: ParametricModel<TDataType>()
	{

		this->varRow()->setRange(2, 50);
		this->varColumns()->setRange(3, 50);
		this->varRadius()->setRange(0.001f, 10.0f);
		this->varHeight()->setRange(0.001f, 10.0f);

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());



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
	void CylinderModel<TDataType>::resetStates()
	{
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		auto radius = this->varRadius()->getData();
		auto row = this->varRow()->getData();
		auto columns = this->varColumns()->getData();
		auto height = this->varHeight()->getData();

		radius *= (sqrt(scale[0] * scale[0] + scale[1] * scale[1] + scale[2] * scale[2]));

		//TCylinder3D<Real> tour;
		//tour.row = row;
		//tour.columns = columns;
		//tour.radius = radius;
		//tour.height = height;

		//this->outCylinder()->setValue(tour);

		auto triangleSet = this->stateTriangleSet()->getDataPtr();

		Real PI = 3.1415926535;
		
		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> triangle;


		int columns_i = int(columns);
		int row_i = int(row);

		uint counter = 0;
		Real x, y, z;

		Real angle = PI / 180 * 360 / columns_i;
		Real temp_angle = angle;



		//以下是侧面点的构建
		for (int k = 0; k <= row_i; k++)
		{
			Real tempy = height / row_i * k;

			for (int j = 0; j < columns_i; j++) {

				temp_angle = j * angle;

				x = sin(temp_angle) * radius;
				y = tempy;
				z = cos(temp_angle) * radius;

				vertices.push_back(Coord(x, y, z));
			}
		}
		//以下是底部圆心及上部圆心点的构建
		vertices.push_back(Coord(0, 0, 0));
		vertices.push_back(Coord(0, height, 0));

		//以下是侧面的构建
		for (int rowl = 0; rowl <= row_i - 1; rowl++)
		{


			for (int faceid = 0; faceid < columns_i; faceid++)
			{

				if (faceid != columns_i - 1)
				{
					
					triangle.push_back(TopologyModule::Triangle(1 + faceid + rowl * columns_i, 0 + faceid + rowl * columns_i, columns_i + faceid + rowl * columns_i));
					triangle.push_back(TopologyModule::Triangle(1 + faceid + rowl * columns_i, columns_i + faceid + rowl * columns_i, columns_i + 1 + faceid + rowl * columns_i));
				}
				else
				{
					triangle.push_back(TopologyModule::Triangle(0 + rowl * columns_i, 0 + faceid + rowl * columns_i, 1 + 2 * faceid + rowl * columns_i));
					triangle.push_back(TopologyModule::Triangle(0 + rowl * columns_i, 1 + 2 * faceid + rowl * columns_i, 1 + faceid + rowl * columns_i));
				}

			}
		}

		//以下是底面和顶面的构建

		int pt_len = vertices.size() - 2;

		for (int i = 0; i < columns_i; i++)
		{
			if (i != columns_i - 1)
			{
				triangle.push_back(TopologyModule::Triangle(pt_len, i, i + 1));	//生成底面
				triangle.push_back(TopologyModule::Triangle(pt_len + 1, pt_len - i - 1, pt_len - i - 2));		//生成顶面
			}
			else
			{
				triangle.push_back(TopologyModule::Triangle(pt_len, i, 0));	//生成最后一个底面
				triangle.push_back(TopologyModule::Triangle(pt_len + 1, pt_len - i - 1, pt_len - 1));	//生成最后一个顶面
			}

		}


		triangleSet->setPoints(vertices);
		triangleSet->setTriangles(triangle);

		triangleSet->updateEdges();
		triangleSet->updateVertexNormal();


		triangleSet->update();

		vertices.clear();
		triangle.clear();
	}

	template<typename TDataType>
	void CylinderModel<TDataType>::disableRender() {
		glModule->setVisible(false);
	};

	DEFINE_CLASS(CylinderModel);
}