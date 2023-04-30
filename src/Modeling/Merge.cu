#include "Merge.h"
#include "Topology/PointSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"



namespace dyno
{
	template<typename TDataType>
	Merge<TDataType>::Merge()

	{

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Color(0.8f, 0.3f, 0.25f));
		glModule->setVisible(true);
		this->stateTriangleSet()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);

		this->inTriangleSet01()->tagOptional(true);
		this->inTriangleSet02()->tagOptional(true);
		this->inTriangleSet03()->tagOptional(true);
		this->inTriangleSet04()->tagOptional(true);

		auto ptModule = std::make_shared<GLPointVisualModule>();
		ptModule->setVisible(false);
		this->stateTriangleSet()->connect(ptModule->inPointSet());
		this->graphicsPipeline()->pushModule(ptModule);
		ptModule->varPointSize()->setValue(0.01);
		
		this->stateTriangleSet()->promoteOuput();
	}

	template<typename TDataType>
	void Merge<TDataType>::resetStates()
	{
		MergeCPU();

	}
	template<typename TDataType>
	void Merge<TDataType>::preUpdateStates()
	{
		Node::preUpdateStates();
		if (this->varUpdateMode()->getData() == UpdateMode::Tick) 
		{
			MergeCPU();
		}

	
	}
	template<typename TDataType>
	void Merge<TDataType>::MergeCPU() 
	{
		TriangleSet<TDataType> TriangleSet01;
		TriangleSet<TDataType> TriangleSet02;
		TriangleSet<TDataType> TriangleSet03;
		TriangleSet<TDataType> TriangleSet04;

		DArray<TopologyModule::Triangle> d_triangle01;
		CArray<TopologyModule::Triangle> c_triangle01;
		DArray<TopologyModule::Triangle> d_triangle02;
		CArray<TopologyModule::Triangle> c_triangle02;
		DArray<TopologyModule::Triangle> d_triangle03;
		CArray<TopologyModule::Triangle> c_triangle03;
		DArray<TopologyModule::Triangle> d_triangle04;
		CArray<TopologyModule::Triangle> c_triangle04;

		DArray<Coord> d_point01;
		CArray<Coord> c_point01;
		DArray<Coord> d_point02;
		CArray<Coord> c_point02;
		DArray<Coord> d_point03;
		CArray<Coord> c_point03;
		DArray<Coord> d_point04;
		CArray<Coord> c_point04;
		std::vector<TopologyModule::Triangle> triangle;
		std::vector<Coord>point;
		printf("初始化变量\n");

		auto triangleSet = this->stateTriangleSet()->getDataPtr();
		printf("auto triangleSet\n");

		int addtri = 0;
		int addpt = 0;

		if (!this->inTriangleSet01()->isEmpty())
		{
			TriangleSet01.copyFrom(this->inTriangleSet01()->getData());
		}
		if (!this->inTriangleSet02()->isEmpty())
		{
			TriangleSet02.copyFrom(this->inTriangleSet02()->getData());
		}
		if (!this->inTriangleSet03()->isEmpty())
		{
			TriangleSet03.copyFrom(this->inTriangleSet03()->getData());
		}
		if (!this->inTriangleSet04()->isEmpty())
		{
			TriangleSet04.copyFrom(this->inTriangleSet04()->getData());
		}
		printf("TriangleSet.copyFrom\n");

		if (!this->inTriangleSet01()->isEmpty())
		{
			d_triangle01 = this->inTriangleSet01()->getData().getTriangles();
			d_point01 = this->inTriangleSet01()->getData().getPoints();

			c_point01.assign(d_point01);
			c_triangle01.assign(d_triangle01);

			d_triangle01 = TriangleSet01.getTriangles();
			d_point01 = TriangleSet01.getPoints();


			int ptsize01 = d_point01.size();
			for (int i = 0; i < ptsize01; i++)
			{
				point.push_back(c_point01[i]);
			}



			int trisize01 = d_triangle01.size();
			for (int i = 0; i < trisize01; i++)
			{
				triangle.push_back(c_triangle01[i]);
			}

			addpt = addpt + ptsize01;
			addtri = addtri + trisize01;

		}

		if (!this->inTriangleSet02()->isEmpty())
		{
			d_triangle02 = this->inTriangleSet02()->getData().getTriangles();
			d_point02 = this->inTriangleSet02()->getData().getPoints();

			c_point02.assign(d_point02);
			c_triangle02.assign(d_triangle02);

			d_triangle02 = TriangleSet02.getTriangles();
			d_point02 = TriangleSet02.getPoints();


			int ptsize02 = d_point02.size();
			for (int i = 0; i < ptsize02; i++)
			{
				point.push_back(c_point02[i]);
			}



			int trisize02 = d_triangle02.size();
			for (int i = 0; i < trisize02; i++)
			{
				triangle.push_back(TopologyModule::Triangle(c_triangle02[i][0] + addpt, c_triangle02[i][1] + addpt, c_triangle02[i][2] + addpt));//c_triangle02[i][0] + addpt, c_triangle02[i][1] + addpt, c_triangle02[i][2] + addpt
			}

			addpt = addpt + ptsize02;
			addtri = addtri + trisize02;

		}

		if (!this->inTriangleSet03()->isEmpty())
		{
			d_triangle03 = this->inTriangleSet03()->getData().getTriangles();
			d_point03 = this->inTriangleSet03()->getData().getPoints();

			c_point03.assign(d_point03);
			c_triangle03.assign(d_triangle03);

			d_triangle03 = TriangleSet03.getTriangles();
			d_point03 = TriangleSet03.getPoints();


			int ptsize03 = d_point03.size();
			for (int i = 0; i < ptsize03; i++)
			{
				point.push_back(c_point03[i]);
			}



			int trisize03 = d_triangle03.size();
			for (int i = 0; i < trisize03; i++)
			{
				triangle.push_back(TopologyModule::Triangle(c_triangle03[i][0] + addpt, c_triangle03[i][1] + addpt, c_triangle03[i][2] + addpt));//c_triangle02[i][0] + addpt, c_triangle02[i][1] + addpt, c_triangle02[i][2] + addpt
			}

			addpt = addpt + ptsize03;
			addtri = addtri + trisize03;

		}

		if (!this->inTriangleSet04()->isEmpty())
		{
			d_triangle04 = this->inTriangleSet04()->getData().getTriangles();
			d_point04 = this->inTriangleSet04()->getData().getPoints();

			c_point04.assign(d_point04);
			c_triangle04.assign(d_triangle04);

			d_triangle04 = TriangleSet04.getTriangles();
			d_point04 = TriangleSet04.getPoints();


			int ptsize04 = d_point04.size();
			for (int i = 0; i < ptsize04; i++)
			{
				point.push_back(c_point04[i]);
			}



			int trisize04 = d_triangle04.size();
			for (int i = 0; i < trisize04; i++)
			{
				triangle.push_back(TopologyModule::Triangle(c_triangle04[i][0] + addpt, c_triangle04[i][1] + addpt, c_triangle04[i][2] + addpt));//c_triangle02[i][0] + addpt, c_triangle02[i][1] + addpt, c_triangle02[i][2] + addpt
			}

			addpt = addpt + ptsize04;
			addtri = addtri + trisize04;

		}

		printf("update PT\n");
		triangleSet->setPoints(point);
		triangleSet->setTriangles(triangle);
		printf("Set\n");


		//triangleSet->update();

		printf("update\n");


		point.clear();
		triangle.clear();
		c_point01.clear();
		c_point02.clear();
		c_point03.clear();
		c_point04.clear();
		c_triangle01.clear();
		c_triangle02.clear();
		c_triangle03.clear();
		c_triangle04.clear();
		printf("clear\n");

	}

	template<typename TDataType>
	void Merge<TDataType>::disableRender() {
		glModule->setVisible(false);
	};




	DEFINE_CLASS(Merge);
}