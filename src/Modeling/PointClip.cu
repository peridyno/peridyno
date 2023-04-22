#include "PointClip.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"



namespace dyno
{
	template<typename TDataType>
	PointClip<TDataType>::PointClip()
	{
		this->stateClipPlane()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->statePointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		auto plane = this->stateClipPlane()->getDataPtr();
		auto size = this->varPlaneSize()->getData();

		std::vector<TopologyModule::Triangle> triangles;

		planeVertices.push_back(Coord(0, size, size));
		planeVertices.push_back(Coord(0, size, -size));
		planeVertices.push_back(Coord(0, -size, -size));
		planeVertices.push_back(Coord(0, -size, size));

		triangles.push_back(TopologyModule::Triangle(0,1,2));
		triangles.push_back(TopologyModule::Triangle(2,3,0));

		plane->setPoints(planeVertices);
		plane->setTriangles(triangles);
		
		surface = std::make_shared<GLSurfaceVisualModule>();
		glpoint = std::make_shared<GLPointVisualModule>();

		surface->setAlpha(0.4);
		surface->setColor(Vec3f(0,0,0));
		this->stateClipPlane()->connect(surface->inTriangleSet());
;
		glpoint->setColor(this->varPointColor()->getData());
		glpoint->varPointSize()->setValue(this->varPointSize()->getData());
		this->statePointSet()->connect(glpoint->inPointSet());

		this->graphicsPipeline()->pushModule(surface);
		this->graphicsPipeline()->pushModule(glpoint);
		this->statePointSet()->promoteOuput();


	}

	template<typename TDataType>
	void PointClip<TDataType>::resetStates()
	{
		transformPlane();
		clip();
		showPlane();
		glpoint->setColor(this->varPointColor()->getData());

	}

	template<typename TDataType>
	void PointClip<TDataType>::updateStates()
	{
		transformPlane();
		clip();
		showPlane();
	}

	template<typename TDataType>
	void PointClip<TDataType>::clip() 
	{
		auto plane = this->stateClipPlane()->getDataPtr();

		auto pointSet = this->inPointSet()->getDataPtr();
		auto pointSetState = this->statePointSet()->getDataPtr();

		CArray<Coord> planePoint;
		planePoint.assign(plane->getPoints());
		Vec3f planeV1 = planePoint[0] - planePoint[1];
		Vec3f planeV2 = planePoint[1] - planePoint[2];
		planeV1.normalize();
		planeV2.normalize();
		Normal = planeV1.cross(planeV2);
		Vec3f planeV3;


		CArray<Coord> c_point;
		std::vector<Coord> vertices;
		c_point.assign(pointSet->getPoints());

		for (size_t i = 0; i < c_point.size(); i++) 
		{
			if (this->varReverse()->getData()) { planeV3 = c_point[i]-planePoint[0];}
			else { planeV3 = planePoint[0] - c_point[i]; }
			if (planeV3.dot(Normal) >= 0)
			{
				vertices.push_back(c_point[i]);
			}
		}
		pointSetState->getPoints().clear();
		pointSetState->setPoints(vertices);
		pointSetState->update();
		vertices.clear();
		planePoint.clear();
		c_point.clear();
	}


	template<typename TDataType>
	void PointClip<TDataType>::transformPlane() 
	{
		auto plane = this->stateClipPlane()->getDataPtr();

		auto scale = this->varScale()->getData();
		auto rot = this->varRotation()->getData();
		auto center = this->varLocation()->getData();

		std::vector<Coord> vertices;

		for (size_t i = 0; i < planeVertices.size(); i++) 
		{
			vertices.push_back(planeVertices[i]);
		}


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
			vertices[i] = RV(vertices[i] * scale + RV(center));
		}
		plane->getPoints().clear();
		plane->setPoints(vertices);
		vertices.clear();

	}

	template<typename TDataType>
	void PointClip<TDataType>::showPlane()
	{
		surface->setVisible(this->varShowPlane()->getData());
	}
	DEFINE_CLASS(PointClip);
}