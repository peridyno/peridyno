#include "Sweep.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"



namespace dyno
{
	template<typename TDataType>
	SweepModel<TDataType>::SweepModel()
		: ParametricModel<TDataType>()
	{

		this->varRadius()->setRange(0.001f, 10.0f);
		this->varRadius()->setValue(1);


		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());



		glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Color(0.8f, 0.52f, 0.25f));
		glModule->setVisible(true);
		this->stateTriangleSet()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);


		auto wireframe = std::make_shared<GLWireframeVisualModule>();
		this->stateTriangleSet()->connect(wireframe->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireframe);

		this->stateTriangleSet()->promoteOuput();

		auto pointGlModule = std::make_shared<GLPointVisualModule>();
		this->stateTriangleSet()->connect(pointGlModule->inPointSet());
		this->graphicsPipeline()->pushModule(pointGlModule);
		pointGlModule->varPointSize()->setValue(0.02);

		this->stateTriangleSet()->promoteOuput();
	}

	template<typename TDataType>
	void SweepModel<TDataType>::resetStates()
	{

		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		auto radius = this->varRadius()->getData();
		
		auto VertexIn = this->inSpline()->getData().getPoints();

		auto triangleSet = this->stateTriangleSet()->getDataPtr();

		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> triangle;

		//Curve1路径曲线 曲线点
		CArray<Coord> c_point1;
		c_point1.assign(VertexIn);

		int lengthV = VertexIn.size();
		totalIndex = lengthV;
		//Curve2环形曲线 曲线点
		auto VertexIn2 = this->inCurve()->getData().getPoints();

		CArray<Coord> c_point2;
		c_point2.assign(VertexIn2);

		int lengthV2 = VertexIn2.size();

		unsigned CopyNumber = lengthV;

		Real PI = 3.1415926535;

		uint counter = 0;
		Coord Location1;
		Coord Location2;
		Coord LocationTemp1 = {0,1,0};
		Coord LocationTemp2 = {0,1,0};

		Vec3f Dir;
		//建立四元数以进行递归变换


		for (size_t i = 0; i < lengthV; i++) 
		{
			currentIndex = i;
			Location2 = { c_point1[i][0], c_point1[i][1], c_point1[i][2] };
			
			int next = i + 1;
			if (i == lengthV - 1) { next = i - 1; }

			LocationTemp1 = { c_point1[next][0], c_point1[next][1], c_point1[next][2] };

			Vec3f vb = Vec3f(0,1,0);
			Vec3f va = LocationTemp1 - Location2;
			if (i == lengthV - 1) { va = Location2 - LocationTemp1; }

			va.normalize();
			vb.normalize();

			Vec3f v =va.cross(vb);
			Vec3f vs = va.cross(vb);
			v.normalize();

			float ca = vb.dot(va);

			float scale = 1 - ca;
			
			Vec3f vt = Vec3f(v[0]*scale ,v[1]*scale ,v[2]*scale);

			SquareMatrix<Real, 3> rotationMatrix;
			rotationMatrix(0,0) = vt[0] * v[0] + ca;
			rotationMatrix(1,1) = vt[1] * v[1] + ca;
			rotationMatrix(2,2) = vt[2] * v[2] + ca;
			vt[0] *= v[1];
			vt[2] *= v[0];
			vt[1] *= v[2];

			rotationMatrix(0,1) = vt[0] - vs[2];
			rotationMatrix(0,2) = vt[2] + vs[1];
			rotationMatrix(1,0) = vt[0] + vs[2];
			rotationMatrix(1,2) = vt[1] - vs[0];
			rotationMatrix(2,0) = vt[2] - vs[1];
			rotationMatrix(2,1) = vt[1] + vs[0];


			Quat<Real> q = Quat<Real>(rotationMatrix);


			auto RV = [&](const Coord& v)->Coord
			{
				return q.rotate(v);//
			};


			for (size_t k = 0; k < lengthV2; k++ ) 
			{

				Location1 = { c_point2[k][0], c_point2[k][1], c_point2[k][2] };
				
				


				Location1 = RV(Location1 * RealScale()) + Location2;//

				vertices.push_back(Location1);

			}

		}


		vertices.push_back(Coord(c_point1[0][0], c_point1[0][1], c_point1[0][2] ));
		vertices.push_back(Coord(c_point1[lengthV-1][0], c_point1[lengthV - 1][1], c_point1[lengthV - 1][2]));

		unsigned ptnum = vertices.size();
	
		for (int rowl = 0; rowl < lengthV - 1; rowl++)
		{
			for (int faceid = 0; faceid < lengthV2; faceid++)
			{
				if (faceid != lengthV2 - 1)
				{

					triangle.push_back(TopologyModule::Triangle(lengthV2 + faceid + rowl * lengthV2, 0 + faceid + rowl * lengthV2, 1 + faceid + rowl * lengthV2));
					triangle.push_back(TopologyModule::Triangle(lengthV2 + 1 + faceid + rowl * lengthV2, lengthV2 + faceid + rowl * lengthV2, 1 + faceid + rowl * lengthV2));
				}
				else
				{
					triangle.push_back(TopologyModule::Triangle(1 + 2 * faceid + rowl * lengthV2, 0 + faceid + rowl * lengthV2, 0 + rowl * lengthV2));
					triangle.push_back(TopologyModule::Triangle(1 + faceid + rowl * lengthV2, 1 + 2 * faceid + rowl * lengthV2, 0 + rowl * lengthV2));
				}
			}

		}


		for (int i = 0; i < lengthV2; i++) 
		{
			if (i < lengthV2 - 1) 
			{
				triangle.push_back(TopologyModule::Triangle(i, i + 1, ptnum - 2));
				triangle.push_back(TopologyModule::Triangle(ptnum - 3 - lengthV2 + i +1, ptnum - 3 - lengthV2 + i + 2, ptnum - 1));
			}
			else 
			{
				triangle.push_back(TopologyModule::Triangle(i, i - lengthV2 + 1, ptnum - 2));
				triangle.push_back(TopologyModule::Triangle(ptnum - 3 - lengthV2 + i + 1, ptnum - 3 - lengthV2 + i + 2 - lengthV2, ptnum - 1));
			}
		}



		//变换

		Quat<Real> q2 = computeQuaternion();

		q2.normalize();

		auto RVt = [&](const Coord& v)->Coord {
			return center + q2.rotate(v - center);
		};

		int numpt = vertices.size();

		for (int i = 0; i < numpt; i++)
		{

			vertices[i] =RVt( vertices[i] * scale + RVt( center ));
		}

		int s = vertices.size();

		triangleSet->setPoints(vertices);
		triangleSet->setTriangles(triangle);

		triangleSet->update();



		vertices.clear();
		triangle.clear();
		

	}


	template<typename TDataType>
	void SweepModel<TDataType>::disableRender() {
		glModule->setVisible(false);
	};

	template<typename TDataType>
	Vec3f SweepModel<TDataType>::RealScale() 
	{
		auto radius = this->varRadius()->getData();
		//后续可以修改这个变换以起到使sweep细节更丰富的目的
		Vec3f s = Vec3f(1*radius ,1* radius ,1 *radius);
		if (this->varuseRamp()->getData() == true) 
		{
			float pr = this->varCurveRamp()->getValue().getCurveValueByX(currentIndex / totalIndex);
			std::cout << "采样值为" << pr << std::endl;
			if (pr != -1) { s = s * pr;}
		}
		return s;
	}



	DEFINE_CLASS(SweepModel);
}