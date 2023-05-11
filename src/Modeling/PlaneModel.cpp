#include "PlaneModel.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	PlaneModel<TDataType>::PlaneModel()
		: ParametricModel<TDataType>()
	{
		this->varLength()->setRange(0.01, 100.0f);
		this->varSegments()->setRange(1, 100);

		this->stateQuadSet()->setDataPtr(std::make_shared<QuadSet<TDataType>>());

		auto glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Color(0.8f, 0.52f, 0.25f));
		glModule->setVisible(true);
		this->stateQuadSet()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);

		auto wireframe = std::make_shared<GLWireframeVisualModule>();
		this->stateQuadSet()->connect(wireframe->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireframe);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&PlaneModel<TDataType>::varChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varSegments()->attach(callback);
		this->varLength()->attach(callback);

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
		NBoundingBox bound;

		auto box = this->outCube()->getData();
		auto aabb = box.aabb();

		Coord v0 = aabb.v0;
		Coord v1 = aabb.v1;

		bound.lower = Vec3f(v0.x, v0.y, v0.z);
		bound.upper = Vec3f(v1.x, v1.y, v1.z);

		return bound;
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

		auto length = this->varLength()->getData();

		length[0] *= scale[0];
		length[1] *= scale[1];
		length[2] *= scale[2];

		Quat<Real> q = Quat<Real>(M_PI * rot[0] / 180, Coord(1, 0, 0))
			* Quat<Real>(M_PI * rot[1] / 180, Coord(0, 1, 0))
			* Quat<Real>(M_PI * rot[2] / 180, Coord(0, 0, 1));

		q.normalize();

		TOrientedBox3D<Real> box;
		box.center = center;
		box.u = q.rotate(Coord(1, 0, 0));
		box.v = q.rotate(Coord(0, 1, 0));
		box.w = q.rotate(Coord(0, 0, 1));
		box.extent = 0.5 * length;

		this->outCube()->setValue(box);

		auto segments = this->varSegments()->getData();

		auto quadSet = this->stateQuadSet()->getDataPtr();

		uint nyz = 2 * (segments[1] + segments[2]);

		std::vector<Coord> vertices;
		std::vector<TopologyModule::Quad> quads;

		Real dx = length[0] / segments[0];
		Real dy = length[1] / segments[1];
		Real dz = length[2] / segments[2];

		std::map<Index2DPlane, uint> indexTop;
		std::map<Index2DPlane, uint> indexBottom;

		//A lambda function to rotate a vertex
		auto RV = [&](const Coord& v)->Coord {
			return center + q.rotate(v - center);
		};


		uint counter = 0;
		Real x, y, z;
		

		//Bottom
		y = center[1];
		for (int nx = 0; nx < segments[0]+1; nx++)
		{
			for (int nz = 0; nz < segments[2]+1; nz++)
			{
				indexBottom[Index2DPlane(nx, nz)] = vertices.size();

				x = nx * dx - length[0] / 2;
				z = nz * dz - length[2] / 2;
				vertices.push_back(RV(Coord(x, y, z)));
			}
		}

		int v0;
		int v1;
		int v2;
		int v3;

		for (int nx = 0; nx < segments[0]; nx++)
		{
			for (int nz = 0; nz < segments[2]; nz++)
			{
				v0 = indexBottom[Index2DPlane(nx, nz)];
				v1 = indexBottom[Index2DPlane(nx + 1, nz)];
				v2 = indexBottom[Index2DPlane(nx + 1, nz + 1)];
				v3 = indexBottom[Index2DPlane(nx, nz + 1)];

				quads.push_back(TopologyModule::Quad(v3, v2, v1, v0));
			}
		}


		quadSet->setPoints(vertices);
		quadSet->setQuads(quads);

		//quadSet->updateTriangles();
		quadSet->update();

		indexTop.clear();
		vertices.clear();
		quads.clear();
	
	}


	DEFINE_CLASS(PlaneModel);
}