#include "PlaneModel.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	PlaneModel<TDataType>::PlaneModel()
		: ParametricModel<TDataType>()
	{
		this->varLengthX()->setRange(0.01, 50.0f);
		this->varLengthZ()->setRange(0.01, 50.0f);

		this->varSegmentX()->setRange(1, 100);
		this->varSegmentZ()->setRange(1, 100);

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->stateQuadSet()->setDataPtr(std::make_shared<QuadSet<TDataType>>());

		auto glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Color(0.8f, 0.52f, 0.25f));
		glModule->setVisible(true);
		this->stateTriangleSet()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);

		auto wireframe = std::make_shared<GLWireframeVisualModule>();
		this->stateTriangleSet()->connect(wireframe->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireframe);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&PlaneModel<TDataType>::varChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varSegmentX()->attach(callback);
		this->varSegmentZ()->attach(callback);

		this->varLengthX()->attach(callback);
		this->varLengthZ()->attach(callback);

		this->stateTriangleSet()->promoteOuput();
		this->stateQuadSet()->promoteOuput();
	}

	struct Index2DPlane
	{
		Index2DPlane() : x(), y() {}
		Index2DPlane(int x, int y) : x(x), y(y) {}
		int x, y;
	};

	bool operator<(const Index2DPlane& lhs, const Index2DPlane& rhs)
	{
		return lhs.x != rhs.x ? lhs.x < rhs.x : lhs.y < rhs.y;
	}

	template<typename TDataType>
	NBoundingBox PlaneModel<TDataType>::boundingBox()
	{
		auto center = this->varLocation()->getValue();
		auto rot = this->varRotation()->getValue();
		auto scale = this->varScale()->getValue();

		Coord length;
		length[0] *= scale[0];
		length[1] = 1;
		length[2] *= scale[2];

		Quat<Real> q = computeQuaternion();

		q.normalize();

		TOrientedBox3D<Real> box;
		box.center = center;
		box.u = q.rotate(Coord(1, 0, 0));
		box.v = q.rotate(Coord(0, 1, 0));
		box.w = q.rotate(Coord(0, 0, 1));
		box.extent = length;

		auto AABB = box.aabb();
		auto& v0 = AABB.v0;
		auto& v1 = AABB.v1;


		NBoundingBox ret;
		ret.lower = Vec3f(v0.x, v0.y, v0.z);
		ret.upper = Vec3f(v1.x, v1.y, v1.z);

		return ret;
	}

	template<typename TDataType>
	void PlaneModel<TDataType>::resetStates()
	{
		varChanged();
	}

	template<typename TDataType>
	void PlaneModel<TDataType>::varChanged()
	{
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		auto segmentX = this->varSegmentX()->getData();
		auto segmentZ = this->varSegmentZ()->getData();

		auto lengthX = this->varLengthX()->getData();
		auto lengthZ = this->varLengthZ()->getData();

		Vec3f length = Vec3f(lengthX,1,lengthZ);
		Vec3i segments = Vec3i(segmentX, 1, segmentZ);

		lengthX *= scale[0];
		lengthZ *= scale[2];

		Quat<Real> q = computeQuaternion();

		q.normalize();

		std::vector<Coord> vertices;
		std::vector<TopologyModule::Quad> quads;
		std::vector<TopologyModule::Triangle> triangles;

		Real dx = lengthX / segmentX;
		Real dz = lengthZ / segmentZ;

		//A lambda function to rotate a vertex
		auto RV = [&](const Coord& v)->Coord {
			return center + q.rotate(v);
		};


		Real x, y, z;
		//Bottom
		for (int nz = 0; nz <= segmentZ; nz++)
		{
			for (int nx = 0; nx <= segmentX; nx++)
			{
				x = nx * dx - lengthX / 2;
				z = nz * dz - lengthZ / 2;
				vertices.push_back(RV(Coord(x, Real(0), z)));
			}
		}

		int v0;
		int v1;
		int v2;
		int v3;

		for (int nz = 0; nz < segmentZ; nz++)
		{
			for (int nx = 0; nx < segmentX; nx++)
			{
				v0 = nx + nz * (segmentX + 1);
				v1 = nx + 1 + nz * (segmentX + 1);;
				v2 = nx + 1 + (nz + 1) * (segmentX + 1);;
				v3 = nx + (nz + 1) * (segmentX + 1);;

				quads.push_back(TopologyModule::Quad(v0, v1, v2, v3));

				if ((nx + nz) % 2 == 0){
					triangles.push_back(TopologyModule::Triangle(v0, v1, v2));
					triangles.push_back(TopologyModule::Triangle(v0, v2, v3));
				}
				else{
					triangles.push_back(TopologyModule::Triangle(v0, v1, v3));
					triangles.push_back(TopologyModule::Triangle(v1, v2, v3));
				}
			}
		}

		auto qs = this->stateQuadSet()->getDataPtr();
		qs->setPoints(vertices);
		qs->setQuads(quads);
		qs->update();

		auto ts = this->stateTriangleSet()->getDataPtr();
		ts->setPoints(vertices);
		ts->setTriangles(triangles);
		ts->update();

		vertices.clear();
		quads.clear();
		triangles.clear();
	}

	DEFINE_CLASS(PlaneModel);
}