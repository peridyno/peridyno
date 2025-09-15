#include "PlaneModel.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

#include "Mapping/QuadSetToTriangleSet.h"
#include "Mapping/Extract.h"

namespace dyno
{
	template<typename TDataType>
	PlaneModel<TDataType>::PlaneModel()
		: BasicShape<TDataType>()
	{
		this->varLengthX()->setRange(0.01, 100.0f);
		this->varLengthZ()->setRange(0.01, 100.0f);

		this->varSegmentX()->setRange(1, 100);
		this->varSegmentZ()->setRange(1, 100);


		this->statePolygonSet()->setDataPtr(std::make_shared<PolygonSet<TDataType>>());

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		this->stateQuadSet()->setDataPtr(std::make_shared<QuadSet<TDataType>>());


		auto callback = std::make_shared<FCallBackFunc>(std::bind(&PlaneModel<TDataType>::varChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varSegmentX()->attach(callback);
		this->varSegmentZ()->attach(callback);

		this->varLengthX()->attach(callback);
		this->varLengthZ()->attach(callback);


		auto tsRender = std::make_shared<GLSurfaceVisualModule>();
		this->stateTriangleSet()->connect(tsRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(tsRender);

		auto exES = std::make_shared<ExtractEdgeSetFromPolygonSet<TDataType>>();
		this->statePolygonSet()->connect(exES->inPolygonSet());
		this->graphicsPipeline()->pushModule(exES);

		auto esRender = std::make_shared<GLWireframeVisualModule>();
		esRender->varBaseColor()->setValue(Color(0, 0, 0));
		//exES->outEdgeSet()->connect(esRender->inEdgeSet());
		this->stateTriangleSet()->connect(esRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(esRender);

		//this->statePolygonSet()->promoteOuput();
		this->stateTriangleSet()->promoteOuput();
		this->stateQuadSet()->promoteOuput();

		this->stateTriangleSet()->promoteOuput();
		this->stateQuadSet()->promoteOuput();
		this->statePolygonSet()->promoteOuput();
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

		Quat<Real> q = this->computeQuaternion();

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

		Quat<Real> q = this->computeQuaternion();

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

		uint numOfPolygon = segments[0] * segments[2];

		CArray<uint> counter2(numOfPolygon);

		uint incre = 0;

		for (uint j = 0; j < numOfPolygon; j++)
		{
			counter2[incre] = 4;
			incre++;
		}

		CArrayList<uint> polygonIndices;
		polygonIndices.resize(counter2);

		incre = 0;

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

				auto& quads = polygonIndices[incre];

				if ((nx + nz) % 2 == 0) {
					quads.insert(v3);
					quads.insert(v2);
					quads.insert(v1);
					quads.insert(v0);
				}
				else {
					quads.insert(v2);
					quads.insert(v1);
					quads.insert(v0);
					quads.insert(v3);
				}
				

				incre++;
			}
		}

		auto polySet = this->statePolygonSet()->getDataPtr();

		polySet->setPoints(vertices);
		polySet->setPolygons(polygonIndices);
		polySet->update();

		polygonIndices.clear();

		auto& ts = this->stateTriangleSet()->getData();
		polySet->turnIntoTriangleSet(ts);

		auto& qs = this->stateQuadSet()->getData();
		polySet->extractQuadSet(qs);

		vertices.clear();

	}

	DEFINE_CLASS(PlaneModel);
}