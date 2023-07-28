#include "CubeModel.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

#include "Mapping/QuadSetToTriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	CubeModel<TDataType>::CubeModel()
		: ParametricModel<TDataType>()
	{
		this->varLength()->setRange(0.01, 100.0f);
		this->varSegments()->setRange(1, 100);

		this->stateQuadSet()->setDataPtr(std::make_shared<QuadSet<TDataType>>());

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		auto q2t = std::make_shared<QuadSetToTriangleSet<TDataType>>();
		this->stateQuadSet()->connect(q2t->inQuadSet());
		this->graphicsPipeline()->pushModule(q2t);

		auto glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Color(0.8f, 0.52f, 0.25f));
		glModule->setVisible(true);
		q2t->outTriangleSet()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);

		auto wireframe = std::make_shared<GLWireframeVisualModule>();
		this->stateQuadSet()->connect(wireframe->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireframe);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&CubeModel<TDataType>::varChanged, this));

		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		this->varSegments()->attach(callback);
		this->varLength()->attach(callback);

	 	this->stateQuadSet()->promoteOuput();
		this->stateTriangleSet()->promoteOuput();

		//Do not export the node
		this->allowExported(false);
	}

	struct Index2D
	{
		Index2D() : x(), y() {}
		Index2D(int x, int y) : x(x), y(y) {}
		int x, y;
	};

	bool operator<(const Index2D& lhs, const Index2D& rhs)
	{
		return lhs.x != rhs.x ? lhs.x < rhs.x : lhs.y < rhs.y;
	}

	template<typename TDataType>
	NBoundingBox CubeModel<TDataType>::boundingBox()
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
	void CubeModel<TDataType>::resetStates()
	{
		varChanged();
	}

	template<typename TDataType>
	void CubeModel<TDataType>::varChanged() 
	{
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		auto length = this->varLength()->getData();

		length[0] *= scale[0];
		length[1] *= scale[1];
		length[2] *= scale[2];

		Quat<Real> q = computeQuaternion();

		q.normalize();

		TOrientedBox3D<Real> box;
		box.center = center;
		box.u = q.rotate(Coord(1, 0, 0));
		box.v = q.rotate(Coord(0, 1, 0));
		box.w = q.rotate(Coord(0, 0, 1));
		box.extent = 0.5 * length;

		this->outCube()->setValue(box);

		auto segments = this->varSegments()->getData();

		uint nyz = 2 * (segments[1] + segments[2]);

		std::vector<Coord> vertices;
		std::vector<TopologyModule::Quad> quads;
		std::vector<TopologyModule::Triangle> triangles;

		Real dx = length[0] / segments[0];
		Real dy = length[1] / segments[1];
		Real dz = length[2] / segments[2];

		std::map<Index2D, uint> indexTop;
		std::map<Index2D, uint> indexBottom;

		//A lambda function to rotate a vertex
		auto RV = [&](const Coord& v)->Coord {
			return center + q.rotate(v - center);
		};

		int v0, v1, v2, v3;

		//Rim
		uint counter = 0;
		Real x, y, z;
		for (int nx = 0; nx < segments[0] + 1; nx++)
		{
			x = center[0] - length[0] / 2 + nx * dx;

			z = center[2] - length[2] / 2;
			for (int ny = 0; ny < segments[1]; ny++)
			{
				if (nx == 0)
					indexBottom[Index2D(ny, 0)] = vertices.size();

				if (nx == segments[0])
					indexTop[Index2D(ny, 0)] = vertices.size();

				Real y = center[1] - length[1] / 2 + dy * ny;
				vertices.push_back(RV(Coord(x, y, z)));
			}

			y = center[1] + length[1] / 2;
			for (int nz = 0; nz < segments[2]; nz++)
			{
				if (nx == 0)
					indexBottom[Index2D(segments[1], nz)] = vertices.size();

				if (nx == segments[0])
					indexTop[Index2D(segments[1], nz)] = vertices.size();

				Real z = center[2] - length[2] / 2 + dz * nz;
				vertices.push_back(RV(Coord(x, y, z)));
			}

			z = center[2] + length[2] / 2;
			for (int ny = segments[1]; ny > 0; ny--)
			{
				if (nx == 0)
					indexBottom[Index2D(ny, segments[2])] = vertices.size();

				if (nx == segments[0])
					indexTop[Index2D(ny, segments[2])] = vertices.size();

				Real y = center[1] - length[1] / 2 + dy * ny;
				vertices.push_back(RV(Coord(x, y, z)));
			}

			y = center[1] - length[1] / 2;
			for (int nz = segments[2]; nz > 0; nz--)
			{
				if (nx == 0)
					indexBottom[Index2D(0, nz)] = vertices.size();

				if (nx == segments[0])
					indexTop[Index2D(0, nz)] = vertices.size();

				Real z = center[2] - length[2] / 2 + dz * nz;
				vertices.push_back(RV(Coord(x, y, z)));
			}

			if (nx < segments[0])
			{
				for (int t = 0; t < nyz - 1; t++)
				{
					v0 = counter + t;
					v1 = counter + t + 1;
					v2 = counter + t + nyz + 1;
					v3 = counter + t + nyz;

					quads.push_back(TopologyModule::Quad(v0, v1, v2, v3));

					triangles.push_back(TopologyModule::Triangle(v0, v1, v2));
					triangles.push_back(TopologyModule::Triangle(v0, v2, v3));
				}

				v0 = counter + nyz - 1;
				v1 = counter;
				v2 = counter + nyz;
				v3 = counter + 2 * nyz - 1;

				quads.push_back(TopologyModule::Quad(v0, v1, v2, v3));

				triangles.push_back(TopologyModule::Triangle(v0, v1, v2));
				triangles.push_back(TopologyModule::Triangle(v0, v2, v3));
			}

			counter += nyz;
		}

		//Top
		x = center[0] + length[0] / 2;
		for (int ny = 1; ny < segments[1]; ny++)
		{
			for (int nz = 1; nz < segments[2]; nz++)
			{
				indexTop[Index2D(ny, nz)] = vertices.size();

				y = center[1] - length[1] / 2 + ny * dy;
				z = center[2] - length[2] / 2 + nz * dz;
				vertices.push_back(RV(Coord(x, y, z)));
			}
		}

		for (int ny = 0; ny < segments[1]; ny++)
		{
			for (int nz = 0; nz < segments[2]; nz++)
			{
				v0 = indexTop[Index2D(ny, nz)];
				v1 = indexTop[Index2D(ny + 1, nz)];
				v2 = indexTop[Index2D(ny + 1, nz + 1)];
				v3 = indexTop[Index2D(ny, nz + 1)];

				quads.push_back(TopologyModule::Quad(v0, v1, v2, v3));

				triangles.push_back(TopologyModule::Triangle(v0, v1, v2));
				triangles.push_back(TopologyModule::Triangle(v0, v2, v3));
			}
		}

		//Bottom
		x = center[0] - length[0] / 2;
		for (int ny = 1; ny < segments[1]; ny++)
		{
			for (int nz = 1; nz < segments[2]; nz++)
			{
				indexBottom[Index2D(ny, nz)] = vertices.size();

				y = center[1] - length[1] / 2 + ny * dy;
				z = center[2] - length[2] / 2 + nz * dz;
				vertices.push_back(RV(Coord(x, y, z)));
			}
		}

		for (int ny = 0; ny < segments[1]; ny++)
		{
			for (int nz = 0; nz < segments[2]; nz++)
			{
				v0 = indexBottom[Index2D(ny, nz)];
				v1 = indexBottom[Index2D(ny + 1, nz)];
				v2 = indexBottom[Index2D(ny + 1, nz + 1)];
				v3 = indexBottom[Index2D(ny, nz + 1)];

				quads.push_back(TopologyModule::Quad(v3, v2, v1, v0));

				triangles.push_back(TopologyModule::Triangle(v0, v1, v2));
				triangles.push_back(TopologyModule::Triangle(v0, v2, v3));
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

		indexTop.clear();
		indexBottom.clear();
		vertices.clear();
		quads.clear();
		triangles.clear();
	}


	DEFINE_CLASS(CubeModel);
}