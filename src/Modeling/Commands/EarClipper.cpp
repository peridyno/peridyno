#include "EarClipper.h"
#include "Topology/PointSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"
#include "EarClipper.h"
#include <numeric>

namespace dyno
{
	template<typename TDataType>
	EarClipper<TDataType>::EarClipper()
		: ModelEditing<TDataType>()
	{

	}

	template<typename TDataType>
	void EarClipper<TDataType>::resetStates()
	{
		varChanged();
	}

	template<typename TDataType>
	void EarClipper<TDataType>::varChanged()
	{
		auto d_coords = this->inPointSet()->getDataPtr()->getPoints();
		CArray<Coord> c_coords;
		c_coords.assign(d_coords);
		std::vector<DataType3f::Coord> vts;
		for (size_t i = 0; i < c_coords.size(); i++)
		{
			vts.push_back(c_coords[i]);
		}
		std::vector<TopologyModule::Triangle> outTriangles;
		polyClip(vts,outTriangles);


	}

	Vec3f crossProduct(const Vec3f& a, const Vec3f& b) {
		return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
	}

	float triangleArea(const Vec3f& a, const Vec3f& b, const Vec3f& c) {
		Vec3f ab = b- a;
		Vec3f ac = c- a;
		Vec3f cross = crossProduct(ab, ac);
		return 0.5f * std::sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);
	}


	Vec3f projectPointToTriangle(const Vec3f& A, const Vec3f& B, const Vec3f& C, const Vec3f& P) {
		Vec3f AB = B - A;
		Vec3f AC = C - A;
		Vec3f normal = AB.cross(AC); // 计算法向量

		// 计算点到平面的距离
		float distance = std::fabs(normal.dot(P - A)) / normal.norm();

		// 计算投影点
		Vec3f projection = P - (distance * normal.normalize());
		return projection;
	}

	// Check the point is inside the triangle
	bool isPointInTriangle(const Vec3f& pt, const Vec3f& a, const Vec3f& b, const Vec3f& c) {

		Vec3f projectP = projectPointToTriangle(a,b,c,pt);

		float areaABC = triangleArea(a, b, c);
		float areaABP = triangleArea(a, b, projectP);
		float areaBCP = triangleArea(b, c, projectP);
		float areaCAP = triangleArea(c, a, projectP);

		// Check that the triangles are equal in area
		return std::fabs(areaABC - (areaABP + areaBCP + areaCAP)) < 1e-6;
	}

	// Check poly ear
	bool isEar(const std::vector<Vec3f>& vertices, const TopologyModule::Triangle& triangle,Vec3f n) {
		const Vec3f& a = vertices[triangle[0]];
		const Vec3f& b = vertices[triangle[1]];
		const Vec3f& c = vertices[triangle[2]];
		// Checke convex vertices
		Vec3f ab = b- a;
		Vec3f ac = c - a;
		Vec3f cross = crossProduct(ab, ac);
		if (cross.dot(n) < 0)
			return false;
		// Checke vertices in the triangle
		for (size_t i = 0; i < vertices.size(); ++i) {
			if (i == triangle[0] || i == triangle[1] || i == triangle[2]) {
				continue; //is current triangle vetex,skip it
			}
			if (isPointInTriangle(vertices[i], a, b, c)) {
				return false;
			}
		}

		return true;
	}
	
	// Ear clip
	std::vector<TopologyModule::Triangle> earClipping(const std::vector<Vec3f>& vertices) {
		std::vector<TopologyModule::Triangle> triangles;
		std::vector<int> indices(vertices.size());
		std::iota(indices.begin(), indices.end(), 0); // init 0,1,2...
		
		Vec3f stand_N;
		for (size_t i = 0; i < vertices.size(); i++)
		{
			int prev = indices[(i + indices.size() - 1) % indices.size()];
			int curr = indices[i];
			int next = indices[(i + 1) % indices.size()];

			stand_N += (vertices[prev] - vertices[curr]).cross(vertices[curr] - vertices[next]);
		}
		stand_N.normalize();

		while (indices.size() > 2) {
			bool earFound = false;
			for (size_t i = 0; i < indices.size(); ++i) {
				int prev = indices[(i + indices.size() - 1) % indices.size()];
				int curr = indices[i];
				int next = indices[(i + 1) % indices.size()];

				TopologyModule::Triangle triangle(prev, curr, next);
				if (isEar(vertices, triangle, stand_N)) {
					triangles.push_back(triangle); // add this ear triangle;
					indices.erase(indices.begin() + i); // delete vertex from vertices;
					earFound = true;
					break; 
				}
			}
			if (!earFound) {
				for (int i = 0; i < indices.size() - 2; i++)
				{
					triangles.push_back(TopologyModule::Triangle(indices[0], indices[i + 1], indices[i + 2]));
				}

				break; 
			}
		}

		return triangles;
	}



	template<typename TDataType>
	void EarClipper<TDataType>::polyClip(std::vector<DataType3f::Coord> vts, std::vector<TopologyModule::Triangle>& outTriangles)
	{
		
		outTriangles = earClipping(vts);


	}

	DEFINE_CLASS(EarClipper);
}