#include "CopyModel.h"
#include "Topology/PointSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"



namespace dyno
{
	template<typename TDataType>
	CopyModel<TDataType>::CopyModel()
		: ParametricModel<TDataType>()
	{
		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		auto glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Color(0.8f, 0.52f, 0.25f));
		glModule->setVisible(true);
		this->stateTriangleSet()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);

		auto glModule2 = std::make_shared<GLPointVisualModule>();
		glModule2->setColor(Color(1.0f, 0.1f, 0.1f));
		glModule2->setVisible(false);
		glModule2->varPointSize()->setValue(0.01);
		this->stateTriangleSet()->connect(glModule2->inPointSet());
		this->graphicsPipeline()->pushModule(glModule2);

		auto glModule3 = std::make_shared<GLWireframeVisualModule>();
		glModule3->setColor(Color(0.0f, 0.0f, 0.0f));
		glModule3->setVisible(false);
		this->stateTriangleSet()->connect(glModule3->inEdgeSet());
		this->graphicsPipeline()->pushModule(glModule3);
	}

	template<typename TDataType>
	void CopyModel<TDataType>::resetStates()
	{
		auto center = this->varCopyTransform()->getData();
		auto rot = this->varCopyRotation()->getData();
		auto scale = this->varCopyScale()->getData();

		auto triangleSet = this->stateTriangleSet()->getDataPtr();

		auto VertexIn = this->inTriangleSetIn()->getData().getPoints();
		auto TriangleIn = this->inTriangleSetIn()->getData().getTriangles();

		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> triangle;

		CArray<TopologyModule::Triangle> c_triangle;
		CArray<Coord> c_point;
		c_point.assign(VertexIn);
		c_triangle.assign(TriangleIn);
		
		int lengthV = VertexIn.size();
		int lengthT = TriangleIn.size();
		unsigned CopyNumber = this->varTotalNumber()->getData() - 1 ;

		Coord Location;
		//构建vertices及triangle
		for (int i = 0; i < lengthV; i++)
		{
			Location = { c_point[i][0], c_point[i][1], c_point[i][2] };
			vertices.push_back( Location );
		}
		for (int i = 0; i < lengthT; i++)
		{
			triangle.push_back(TopologyModule::Triangle(c_triangle[i][0], c_triangle[i][1], c_triangle[i][2]));
		}


		//构建Copy多边形
		//顶点
		if (this->varTotalNumber()->getValue() > 1) 
		{

			for(int i = 0;i < CopyNumber; i++)
			{
				//建立四元数以进行递归变换
				Quat<Real> q = Quat<Real>(M_PI * rot[0] / 180 * (i + 1), Coord(1, 0, 0))
					* Quat<Real>(M_PI * rot[1] / 180 * (i + 1), Coord(0, 1, 0))
					* Quat<Real>(M_PI * rot[2] / 180 * (i + 1), Coord(0, 0, 1));
				q.normalize();

				auto RV = [&](const Coord& v)->Coord {
					return q.rotate(v );//center + q.rotate(v - center)
				};

				Vec3f RealScale = scale;
				if (this->varScaleMode()->getData() == 0) 
				{
					for (int k = 0; k < i; k++)
					{
						RealScale *= scale;
					}
				}
				if (this->varScaleMode()->getData() == 1)
				{
					RealScale = scale * 1 / (i + 1);
				}

				for (int j = 0; j < lengthV; j++)
				{
					Location = { c_point[j][0], c_point[j][1], c_point[j][2] };

					Location = RV(Location * RealScale) + center * (i+1);//添加变换RV(Location * RealScale + RV(center * (i + 1)))
					vertices.push_back(Location);
				}
			}
			
			//面
			for (int i = 0; i < CopyNumber; i++) 
			{
				for (int j = 0; j < lengthT; j++)
				{
					;
					triangle.push_back(TopologyModule::Triangle(c_triangle[j][0] + (i + 1) * lengthV, c_triangle[j][1] + (i + 1) * lengthV, c_triangle[j][2] + (i + 1) * lengthV));

				}
		
			}
		}

		//整体变换
		
		auto centert = this->varLocation()->getData();
		auto rott = this->varRotation()->getData();
		auto scalet= this->varScale()->getData();

		Quat<Real> qt = Quat<Real>(M_PI * rott[0] / 180, Coord(1, 0, 0))
			* Quat<Real>(M_PI * rott[1] / 180, Coord(0, 1, 0))
			* Quat<Real>(M_PI * rott[2] / 180, Coord(0, 0, 1));

		qt.normalize();

		auto RVt = [&](const Coord& v)->Coord {
			return centert + qt.rotate(v - centert);
		};

		int numptt = vertices.size();

		for (int i = 0; i < numptt; i++)
		{
			vertices[i] = RVt(vertices[i] * scalet + RVt(centert));
		}

		triangleSet->setPoints(vertices);
		triangleSet->setTriangles(triangle);


		triangleSet->update();

		vertices.clear();
		triangle.clear();
	}

	DEFINE_CLASS(CopyModel);
}