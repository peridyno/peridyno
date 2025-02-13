#include "Sweep.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"
#include "EarClipper.h"



namespace dyno
{
	template<typename TDataType>
	SweepModel<TDataType>::SweepModel()
		: ParametricModel<TDataType>()
	{
		this->varRadius()->setRange(0.001f, 10.0f);

		this->varRadius()->setValue(1);


		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());


		auto glModule = std::make_shared<GLSurfaceVisualModule>();
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
		pointGlModule->varPointSize()->setValue(0.005);

		this->stateTriangleSet()->promoteOuput();

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&SweepModel<TDataType>::varChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varRadius()->attach(callback);
		this->varCurveRamp()->attach(callback);
		this->varReverseNormal()->attach(callback);

		auto displayCallback = std::make_shared<FCallBackFunc>(std::bind(&SweepModel<TDataType>::displayChanged, this));

		this->varDisplayPoints()->attach(displayCallback);
		this->varDisplaySurface()->attach(displayCallback);
		this->varDisplayWireframe()->attach(displayCallback);

	}

	template<typename TDataType>
	void SweepModel<TDataType>::resetStates()
	{
		varChanged();
		displayChanged();
		printf("resetStates  sweep \n");
	}

	template<typename TDataType>
	void SweepModel<TDataType>::varChanged()
	{
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		auto radius = this->varRadius()->getData();
		
		if (this->inSpline()->isEmpty())
		{ 
			printf("inSpline  empty \n");
			return; 
		}
		if (this->inCurve()->isEmpty()) 
		{ 
			printf("inCurve  empty \n");
			return; 
		}

		auto SplineIn = this->inSpline()->getData().getPoints();

		auto triangleSet = this->stateTriangleSet()->getDataPtr();

		auto VertexIn2 = this->inCurve()->getData().getPoints();

		if (SplineIn.size() == 0) { return; }
		if (VertexIn2.size() == 0) { return; }

		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> triangle;

		//Spline路径曲线 曲线点
		CArray<Coord> c_point1;

		c_point1.assign(SplineIn);

		int lengthV = SplineIn.size();
		totalIndex = lengthV;
		//Curve2环形曲线 曲线点

		CArray<Coord> c_point2;
		c_point2.assign(VertexIn2);

		int lengthV2 = VertexIn2.size();

		unsigned CopyNumber = lengthV;

		Real PI = 3.1415926535;

		uint counter = 0;

		Coord LocationCurrent;
		Coord LocationNext = {0,1,0};
		Coord LocationTemp2 = {0,1,0};


		//建立四元数以进行递归变换


		for (size_t i = 0; i < lengthV; i++) 
		{
			currentIndex = i;

			int current = i;
			int next = i + 1;

			if (i == lengthV - int(1))
			{ 
				next = i ; 
				current = i - 1;
			}

			LocationCurrent = { c_point1[current][0], c_point1[current][1], c_point1[current][2] };
			LocationNext = { c_point1[next][0], c_point1[next][1], c_point1[next][2] };

			Vec3f vb = Vec3f(0, 1, 0);
			Vec3f va =LocationNext - LocationCurrent;
			va.normalize();
			vb.normalize();

			Vec3f Axis = va.cross(vb);
			if (Axis == Vec3f(0, 0, 0)) 
			{
				Axis = Vec3f(0,1,0);

			}
			Axis.normalize();

			Real angle =-1 * acos(va.dot(vb));

			//DYN_FUNC Quat(Real rot, const Vector<Real, 3> &axis);  //init from the rotation axis and angle(in radian)

			Quat<Real> q = Quat<Real>(angle ,Axis);


			auto RV = [&](const Coord& v)->Coord
			{
				return q.rotate(v);//
			};

			Coord LocationCurvePoint;
			Coord Offest;
			for (size_t k = 0; k < lengthV2; k++ ) 
			{
				LocationCurvePoint = { c_point2[k][0], c_point2[k][1], c_point2[k][2] };
				Offest = c_point1[i];
				LocationCurvePoint = RV(LocationCurvePoint * RealScale()) + Offest;//
				vertices.push_back(LocationCurvePoint);
			}

		}

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

		//fill cap
		// get triangleCap by EarClipper
		EarClipper<DataType3f> sab;
		std::vector<TopologyModule::Triangle> triangleCap;

		sab.polyClip(VertexIn2, triangleCap);
		int addnum2 = vertices.size() - VertexIn2.size();



		for (int i = 0; i < triangleCap.size(); i++)
		{
			triangle.push_back(TopologyModule::Triangle(triangleCap[i][2], triangleCap[i][1], triangleCap[i][0]));
			triangle.push_back(TopologyModule::Triangle(triangleCap[i][0] + addnum2, triangleCap[i][1] + addnum2, triangleCap[i][2] + addnum2));
		}

		//ReverseNormal
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

		//Transform

		Quat<Real> q2 = this->computeQuaternion();

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
	Vec3f SweepModel<TDataType>::RealScale() 
	{
		auto radius = this->varRadius()->getData();
		Vec3f s = Vec3f(1*radius ,1* radius ,1 *radius);

		float pr = this->varCurveRamp()->getValue().getCurveValueByX(currentIndex / (totalIndex - 1));
		//std::cout << "current : "<< currentIndex <<"  totalIndex : " << totalIndex << "sampler" << pr << std::endl;
		if (pr != -1) { s = s * pr;}

		return s;
	}

	template<typename TDataType>
	void SweepModel<TDataType>::displayChanged()
	{
		auto SurfaceModule = this->graphicsPipeline()->template findFirstModule<GLSurfaceVisualModule>();
		SurfaceModule->setVisible(this->varDisplaySurface()->getValue());

		auto wireModule = this->graphicsPipeline()->template findFirstModule<GLWireframeVisualModule>();
		wireModule->setVisible(this->varDisplayWireframe()->getValue());
	
		auto pointModule = this->graphicsPipeline()->template findFirstModule<GLPointVisualModule>();
		pointModule->setVisible(this->varDisplayPoints()->getValue());
	}

	DEFINE_CLASS(SweepModel);
}