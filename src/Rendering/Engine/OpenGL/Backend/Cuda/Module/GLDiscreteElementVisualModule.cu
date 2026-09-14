/**
 * Copyright 2026 Yuzhong Guo
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "GLDiscreteElementVisualModule.h"

#include <glad/glad.h>
#include <math.h>

#include "ShaderStruct.h"
#include "Utility.h"

#include "surface.vert.h"
#include "surface.frag.h"
#include "surface.geom.h"

namespace dyno
{
	typedef typename ::dyno::TOrientedBox3D<Real> Box3D;

	namespace
	{
		constexpr uint MEDIAL_RENDER_RESOLUTION = 24;
		constexpr float MEDIAL_PI = 3.14159265358979323846f;
		constexpr float MEDIAL_EPSILON = 1.0e-6f;

		DYN_FUNC inline bool normalizeVector(Vec3f& out, const Vec3f& value)
		{
			float lengthSquared = dot(value, value);
			if (lengthSquared <= MEDIAL_EPSILON * MEDIAL_EPSILON)
			{
				out = Vec3f(0);
				return false;
			}

			out = value / sqrtf(lengthSquared);
			return true;
		}

		DYN_FUNC inline Vec3f normalizedOr(const Vec3f& value, const Vec3f& fallback)
		{
			Vec3f ret;
			return normalizeVector(ret, value) ? ret : fallback;
		}

		DYN_FUNC inline float medialAngle(float r0, float r1, const Vec3f& c01)
		{
			float dr = r0 - r1;
			float dr2 = dr * dr;
			if (dr2 <= MEDIAL_EPSILON * MEDIAL_EPSILON)
			{
				return 0.5f * MEDIAL_PI;
			}

			float lengthSquared = dot(c01, c01);
			float ratio = fmaxf(lengthSquared - dr2, 0.0f) / dr2;
			float phi = atanf(sqrtf(ratio));
			return r0 < r1 ? MEDIAL_PI - phi : phi;
		}

		DYN_FUNC inline Vec3f rotateAroundAxis(const Vec3f& value, const Vec3f& axis, float angle)
		{
			float cosAngle = cosf(angle);
			float sinAngle = sinf(angle);
			return value * cosAngle + cross(axis, value) * sinAngle + axis * (dot(axis, value) * (1.0f - cosAngle));
		}

		DYN_FUNC inline Vec3f tangentStartDirection(const Vec3f& axis)
		{
			Vec3f up = fabsf(dot(axis, Vec3f(0, 1, 0))) > 0.999f ? Vec3f(1, 0, 0) : Vec3f(0, 1, 0);
			return normalizedOr(cross(axis, up), Vec3f(0, 0, 1));
		}

		DYN_FUNC inline Vec3f medialConeSurfacePoint(
			const Vec3f& p0,
			const Vec3f& p1,
			float r0,
			float r1,
			uint ringId,
			uint endpointId,
			uint resolution)
		{
			Vec3f axis = normalizedOr(p1 - p0, Vec3f(1, 0, 0));
			Vec3f startDir = tangentStartDirection(axis);
			float angle = 2.0f * MEDIAL_PI * float(ringId) / float(resolution);
			Vec3f radialDir = rotateAroundAxis(startDir, axis, angle);
			float phi = medialAngle(r0, r1, p1 - p0);
			Vec3f sideDir = axis * cosf(phi) + radialDir * sinf(phi);

			Vec3f center = endpointId == 0 ? p0 : p1;
			float radius = endpointId == 0 ? r0 : r1;

			return radius <= MEDIAL_EPSILON ? center : center + sideDir * radius;
		}

		DYN_FUNC inline bool planeLineIntersection(
			Vec3f& out,
			const Vec3f& normal,
			const Vec3f& pointOnPlane,
			const Vec3f& lineDirection,
			const Vec3f& pointOnLine)
		{
			float denominator = dot(lineDirection, normal);
			if (fabsf(denominator) < 1.0e-10f)
			{
				out = pointOnLine;
				return false;
			}

			float t = (dot(pointOnPlane, normal) - dot(pointOnLine, normal)) / denominator;
			out = pointOnLine + lineDirection * t;
			return true;
		}

		DYN_FUNC inline bool intersectPointOfCones(
			Vec3f& out0,
			Vec3f& out1,
			const Vec3f& v0,
			float r0,
			const Vec3f& v1,
			float r1,
			const Vec3f& v2,
			float r2,
			const Vec3f& normal)
		{
			if (r0 < 1.0e-3f)
			{
				out0 = v0;
				out1 = v0;
				return true;
			}

			Vec3f dir01;
			Vec3f dir02;
			if (!normalizeVector(dir01, v1 - v0) || !normalizeVector(dir02, v2 - v0))
			{
				out0 = v0;
				out1 = v0;
				return false;
			}

			float phi01 = medialAngle(r0, r1, v1 - v0);
			float phi02 = medialAngle(r0, r2, v2 - v0);
			Vec3f p01 = v0 + dir01 * (cosf(phi01) * r0);
			Vec3f p02 = v0 + dir02 * (cosf(phi02) * r0);
			Vec3f lineDirection = rotateAroundAxis(normal, dir01, 0.5f * MEDIAL_PI);

			Vec3f intersectPoint;
			if (!planeLineIntersection(intersectPoint, v2 - v0, p02, lineDirection, p01))
			{
				out0 = v0;
				out1 = v0;
				return false;
			}

			Vec3f v0p = intersectPoint - v0;
			float scale = sqrtf(fmaxf(r0 * r0 - dot(v0p, v0p), 1.0e-5f));
			out0 = intersectPoint + normal * scale;
			out1 = intersectPoint - normal * scale;
			return true;
		}

		DYN_FUNC inline bool generateMedialSlabCap(
			Vec3f& p0,
			Vec3f& p1,
			Vec3f& p2,
			Vec3f& p3,
			Vec3f& p4,
			Vec3f& p5,
			const MedialSlab3D& slab)
		{
			Vec3f normal;
			if (!normalizeVector(normal, cross(slab.v[0] - slab.v[1], slab.v[0] - slab.v[2])))
			{
				p0 = p1 = p2 = p3 = p4 = p5 = slab.v[0];
				return false;
			}

			Vec3f tangent0A;
			Vec3f tangent0B;
			Vec3f tangent1A;
			Vec3f tangent1B;
			Vec3f tangent2A;
			Vec3f tangent2B;

			bool valid = intersectPointOfCones(
				tangent0A, tangent0B,
				slab.v[0], slab.radius[0],
				slab.v[1], slab.radius[1],
				slab.v[2], slab.radius[2],
				normal);
			valid = intersectPointOfCones(
				tangent1A, tangent1B,
				slab.v[1], slab.radius[1],
				slab.v[0], slab.radius[0],
				slab.v[2], slab.radius[2],
				normal) && valid;
			valid = intersectPointOfCones(
				tangent2A, tangent2B,
				slab.v[2], slab.radius[2],
				slab.v[0], slab.radius[0],
				slab.v[1], slab.radius[1],
				normal) && valid;

			const float threshold = 1.0e-4f;
			float d0 = sqrtf(dot(tangent0A - slab.v[0], tangent0A - slab.v[0])) - slab.radius[0];
			float d1 = sqrtf(dot(tangent1A - slab.v[1], tangent1A - slab.v[1])) - slab.radius[1];
			float d2 = sqrtf(dot(tangent2A - slab.v[2], tangent2A - slab.v[2])) - slab.radius[2];
			valid = valid && d0 <= threshold && d1 <= threshold && d2 <= threshold;

			if (!valid)
			{
				p0 = p1 = p2 = p3 = p4 = p5 = slab.v[0];
				return false;
			}

			p0 = tangent0A;
			p1 = tangent1A;
			p2 = tangent2A;
			p3 = tangent0B;
			p4 = tangent1B;
			p5 = tangent2B;
			return true;
		}
	}

	// Element instance transform kernels (NEW)
	__global__ void GenerateBoxTransforms(
		DArray<Box3D> boxes,
		DArray<Transform3f> transforms)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= boxes.size()) return;

		Box3D box = boxes[tId];

		Transform3f tf;
		tf.translation() = Vec3f(float(box.center[0]), float(box.center[1]), float(box.center[2]));

		SquareMatrix<float, 3>& rot = tf.rotation();
		rot(0, 0) = float(box.u[0]); rot(0, 1) = float(box.v[0]); rot(0, 2) = float(box.w[0]);
		rot(1, 0) = float(box.u[1]); rot(1, 1) = float(box.v[1]); rot(1, 2) = float(box.w[1]);
		rot(2, 0) = float(box.u[2]); rot(2, 1) = float(box.v[2]); rot(2, 2) = float(box.w[2]);

		tf.scale() = Vec3f(
			float(box.extent[0]) * 2.0f,
			float(box.extent[1]) * 2.0f,
			float(box.extent[2]) * 2.0f);

		transforms[tId] = tf;
	}

	__global__ void GenerateSphereTransforms(
		DArray<Sphere3D> spheres,
		DArray<Transform3f> transforms)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= spheres.size()) return;

		Sphere3D sphere = spheres[tId];

		Transform3f tf;
		tf.translation() = Vec3f(float(sphere.center[0]), float(sphere.center[1]), float(sphere.center[2]));

		SquareMatrix<Real, 3> rot = sphere.rotation.toMatrix3x3();
		SquareMatrix<float, 3>& tfRot = tf.rotation();
		tfRot(0, 0) = float(rot(0, 0)); tfRot(0, 1) = float(rot(0, 1)); tfRot(0, 2) = float(rot(0, 2));
		tfRot(1, 0) = float(rot(1, 0)); tfRot(1, 1) = float(rot(1, 1)); tfRot(1, 2) = float(rot(1, 2));
		tfRot(2, 0) = float(rot(2, 0)); tfRot(2, 1) = float(rot(2, 1)); tfRot(2, 2) = float(rot(2, 2));

		float r = float(sphere.radius);
		tf.scale() = Vec3f(r * 2.0f, r * 2.0f, r * 2.0f);

		transforms[tId] = tf;
	}

	__global__ void GenerateCapsuleTransforms(
		DArray<Capsule3D> capsules,
		DArray<Transform3f> transforms)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= capsules.size()) return;

		Capsule3D capsule = capsules[tId];

		Transform3f tf;
		tf.translation() = Vec3f(float(capsule.center[0]), float(capsule.center[1]), float(capsule.center[2]));

		SquareMatrix<Real, 3> rot = capsule.rotation.toMatrix3x3();
		SquareMatrix<float, 3>& tfRot = tf.rotation();
		tfRot(0, 0) = float(rot(0, 0)); tfRot(0, 1) = float(rot(0, 1)); tfRot(0, 2) = float(rot(0, 2));
		tfRot(1, 0) = float(rot(1, 0)); tfRot(1, 1) = float(rot(1, 1)); tfRot(1, 2) = float(rot(1, 2));
		tfRot(2, 0) = float(rot(2, 0)); tfRot(2, 1) = float(rot(2, 1)); tfRot(2, 2) = float(rot(2, 2));

		float r = float(capsule.radius);
		float h = float(capsule.halfLength);
		tf.scale() = Vec3f(r * 2.0f, h * 2.0f, r * 2.0f);

		transforms[tId] = tf;
	}

	__global__ void FillInstanceColor(
		DArray<Vec3f> colors,
		Vec3f color)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= colors.size()) return;
		colors[tId] = color;
	}

	// Medial/Tet/Triangle kernels (copied from DiscreteElementsToTriangleSet.cu)
	template<typename Triangle>
	__global__ void SetupTetInstances(
		DArray<Vec3f> vertices,
		DArray<Triangle> indices,
		DArray<Tet3D> tets,
		uint pointOffset,
		uint indexOffset,
		uint tetOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tets.size()) return;

		int idx = tId;
		Tet3D tet = tets[idx];

		Vec3f v0 = tet.v[0];
		Vec3f v1 = tet.v[1];
		Vec3f v2 = tet.v[2];
		Vec3f v3 = tet.v[3];

		vertices[pointOffset + idx * 4] = v0;
		vertices[pointOffset + idx * 4 + 1] = v1;
		vertices[pointOffset + idx * 4 + 2] = v2;
		vertices[pointOffset + idx * 4 + 3] = v3;

		uint offset = idx * 4 + pointOffset;

		indices[indexOffset + idx * 4] = Triangle(offset + 0, offset + 1, offset + 2);
		indices[indexOffset + idx * 4 + 1] = Triangle(offset + 0, offset + 1, offset + 3);
		indices[indexOffset + idx * 4 + 2] = Triangle(offset + 1, offset + 2, offset + 3);
		indices[indexOffset + idx * 4 + 3] = Triangle(offset + 0, offset + 2, offset + 3);
	}

	template<typename Triangle>
	__global__ void SetupTriangleInstances(
		DArray<Vec3f> vertices,
		DArray<Triangle> indices,
		DArray<Triangle3D> triangles,
		uint pointOffset,
		uint indexOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triangles.size()) return;

		int idx = tId;
		Triangle3D tri = triangles[idx];

		Vec3f v0 = tri.v[0];
		Vec3f v1 = tri.v[1];
		Vec3f v2 = tri.v[2];

		vertices[pointOffset + idx * 3] = v0;
		vertices[pointOffset + idx * 3 + 1] = v1;
		vertices[pointOffset + idx * 3 + 2] = v2;

		uint offset = idx * 3 + pointOffset;
		indices[indexOffset + idx] = Triangle(offset + 0, offset + 1, offset + 2);
	}

	__global__ void SetupVerticesForMedialConeSurfaces(
		DArray<Vec3f> vertices,
		DArray<MedialCone3D> coneInstances,
		uint resolution,
		uint pointOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		uint verticesPerCone = resolution * 2;
		if (tId >= coneInstances.size() * verticesPerCone) return;

		uint instanceId = tId / verticesPerCone;
		uint localId = tId % verticesPerCone;
		uint ringId = localId / 2;
		uint endpointId = localId % 2;
		MedialCone3D cone = coneInstances[instanceId];

		vertices[pointOffset + tId] = medialConeSurfacePoint(
			cone.v[0],
			cone.v[1],
			cone.radius[0],
			cone.radius[1],
			ringId,
			endpointId,
			resolution);
	}

	template<typename Triangle>
	__global__ void SetupIndicesForMedialConeSurfaces(
		DArray<Triangle> indices,
		DArray<MedialCone3D> coneInstances,
		uint resolution,
		uint vertexOffset,
		uint indexOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		uint trianglesPerCone = resolution * 2;
		if (tId >= coneInstances.size() * trianglesPerCone) return;

		uint instanceId = tId / trianglesPerCone;
		uint localId = tId % trianglesPerCone;
		uint segmentId = localId / 2;
		uint nextSegmentId = (segmentId + 1) % resolution;
		uint baseVertex = vertexOffset + instanceId * resolution * 2;

		uint v0 = baseVertex + segmentId * 2;
		uint v1 = v0 + 1;
		uint v2 = baseVertex + nextSegmentId * 2;
		uint v3 = v2 + 1;

		indices[indexOffset + tId] = (localId % 2 == 0)
			? Triangle(v0, v3, v1)
			: Triangle(v0, v2, v3);
	}

	__global__ void SetupVerticesForMedialSlabEdgeSurfaces(
		DArray<Vec3f> vertices,
		DArray<MedialSlab3D> slabInstances,
		uint resolution,
		uint pointOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		uint verticesPerEdge = resolution * 2;
		uint edgeInstanceCount = slabInstances.size() * 3;
		if (tId >= edgeInstanceCount * verticesPerEdge) return;

		uint edgeInstanceId = tId / verticesPerEdge;
		uint slabId = edgeInstanceId / 3;
		uint edgeId = edgeInstanceId % 3;
		uint localId = tId % verticesPerEdge;
		uint ringId = localId / 2;
		uint endpointId = localId % 2;

		MedialSlab3D slab = slabInstances[slabId];
		uint v0Id = edgeId;
		uint v1Id = (edgeId + 1) % 3;

		vertices[pointOffset + tId] = medialConeSurfacePoint(
			slab.v[v0Id],
			slab.v[v1Id],
			slab.radius[v0Id],
			slab.radius[v1Id],
			ringId,
			endpointId,
			resolution);
	}

	template<typename Triangle>
	__global__ void SetupIndicesForMedialSlabEdgeSurfaces(
		DArray<Triangle> indices,
		DArray<MedialSlab3D> slabInstances,
		uint resolution,
		uint vertexOffset,
		uint indexOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		uint trianglesPerEdge = resolution * 2;
		uint edgeInstanceCount = slabInstances.size() * 3;
		if (tId >= edgeInstanceCount * trianglesPerEdge) return;

		uint edgeInstanceId = tId / trianglesPerEdge;
		uint localId = tId % trianglesPerEdge;
		uint segmentId = localId / 2;
		uint nextSegmentId = (segmentId + 1) % resolution;
		uint baseVertex = vertexOffset + edgeInstanceId * resolution * 2;

		uint v0 = baseVertex + segmentId * 2;
		uint v1 = v0 + 1;
		uint v2 = baseVertex + nextSegmentId * 2;
		uint v3 = v2 + 1;

		indices[indexOffset + tId] = (localId % 2 == 0)
			? Triangle(v0, v3, v1)
			: Triangle(v0, v2, v3);
	}

	__global__ void SetupVerticesForMedialConeSphereInstances(
		DArray<Vec3f> vertices,
		DArray<Vec3f> sphereVertices,
		DArray<MedialCone3D> coneInstances,
		uint pointOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		uint sphereVertexCount = sphereVertices.size();
		uint sphereInstanceCount = coneInstances.size() * 2;
		if (tId >= sphereInstanceCount * sphereVertexCount) return;

		uint sphereInstanceId = tId / sphereVertexCount;
		uint vertexId = tId % sphereVertexCount;
		uint coneId = sphereInstanceId / 2;
		uint endpointId = sphereInstanceId % 2;
		MedialCone3D cone = coneInstances[coneId];

		vertices[pointOffset + tId] = cone.v[endpointId] + sphereVertices[vertexId] * cone.radius[endpointId];
	}

	template<typename Triangle>
	__global__ void SetupIndicesForMedialConeSphereInstances(
		DArray<Triangle> indices,
		DArray<Triangle> sphereIndices,
		DArray<MedialCone3D> coneInstances,
		uint sphereVertexCount,
		uint vertexOffset,
		uint indexOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		uint sphereIndexCount = sphereIndices.size();
		uint sphereInstanceCount = coneInstances.size() * 2;
		if (tId >= sphereInstanceCount * sphereIndexCount) return;

		uint sphereInstanceId = tId / sphereIndexCount;
		uint indexId = tId % sphereIndexCount;
		uint baseVertex = vertexOffset + sphereInstanceId * sphereVertexCount;
		Triangle tIndex = sphereIndices[indexId];

		indices[indexOffset + tId] = Triangle(tIndex[0] + baseVertex, tIndex[1] + baseVertex, tIndex[2] + baseVertex);
	}

	__global__ void SetupVerticesForMedialSlabSphereInstances(
		DArray<Vec3f> vertices,
		DArray<Vec3f> sphereVertices,
		DArray<MedialSlab3D> slabInstances,
		uint pointOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		uint sphereVertexCount = sphereVertices.size();
		uint sphereInstanceCount = slabInstances.size() * 3;
		if (tId >= sphereInstanceCount * sphereVertexCount) return;

		uint sphereInstanceId = tId / sphereVertexCount;
		uint vertexId = tId % sphereVertexCount;
		uint slabId = sphereInstanceId / 3;
		uint endpointId = sphereInstanceId % 3;
		MedialSlab3D slab = slabInstances[slabId];

		vertices[pointOffset + tId] = slab.v[endpointId] + sphereVertices[vertexId] * slab.radius[endpointId];
	}

	template<typename Triangle>
	__global__ void SetupIndicesForMedialSlabSphereInstances(
		DArray<Triangle> indices,
		DArray<Triangle> sphereIndices,
		DArray<MedialSlab3D> slabInstances,
		uint sphereVertexCount,
		uint vertexOffset,
		uint indexOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		uint sphereIndexCount = sphereIndices.size();
		uint sphereInstanceCount = slabInstances.size() * 3;
		if (tId >= sphereInstanceCount * sphereIndexCount) return;

		uint sphereInstanceId = tId / sphereIndexCount;
		uint indexId = tId % sphereIndexCount;
		uint baseVertex = vertexOffset + sphereInstanceId * sphereVertexCount;
		Triangle tIndex = sphereIndices[indexId];

		indices[indexOffset + tId] = Triangle(tIndex[0] + baseVertex, tIndex[1] + baseVertex, tIndex[2] + baseVertex);
	}

	__global__ void SetupVerticesForMedialSlabCaps(
		DArray<Vec3f> vertices,
		DArray<MedialSlab3D> slabInstances,
		uint pointOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= slabInstances.size() * 6) return;

		uint slabId = tId / 6;
		uint localId = tId % 6;
		Vec3f p0;
		Vec3f p1;
		Vec3f p2;
		Vec3f p3;
		Vec3f p4;
		Vec3f p5;
		generateMedialSlabCap(p0, p1, p2, p3, p4, p5, slabInstances[slabId]);

		Vec3f point = p0;
		if (localId == 1) point = p1;
		else if (localId == 2) point = p2;
		else if (localId == 3) point = p3;
		else if (localId == 4) point = p4;
		else if (localId == 5) point = p5;

		vertices[pointOffset + tId] = point;
	}

	template<typename Triangle>
	__global__ void SetupIndicesForMedialSlabCaps(
		DArray<Triangle> indices,
		DArray<MedialSlab3D> slabInstances,
		uint vertexOffset,
		uint indexOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= slabInstances.size() * 2) return;

		uint slabId = tId / 2;
		uint localId = tId % 2;
		uint baseVertex = vertexOffset + slabId * 6;

		indices[indexOffset + tId] = localId == 0
			? Triangle(baseVertex + 5, baseVertex + 3, baseVertex + 4)
			: Triangle(baseVertex + 2, baseVertex + 1, baseVertex);
	}

	// Constructor
	template<typename TDataType>
	GLDiscreteElementVisualModule<TDataType>::GLDiscreteElementVisualModule()
	{
		this->setName("discrete_element_visual");

		// Create unit box [-0.5, 0.5]^3: 8 vertices, 12 triangles
		std::vector<Vec3f> pts;
		pts.resize(8);
		pts[0] = Vec3f(-0.5f, -0.5f, -0.5f);
		pts[1] = Vec3f(0.5f, -0.5f, -0.5f);
		pts[2] = Vec3f(0.5f, 0.5f, -0.5f);
		pts[3] = Vec3f(-0.5f, 0.5f, -0.5f);
		pts[4] = Vec3f(-0.5f, -0.5f, 0.5f);
		pts[5] = Vec3f(0.5f, -0.5f, 0.5f);
		pts[6] = Vec3f(0.5f, 0.5f, 0.5f);
		pts[7] = Vec3f(-0.5f, 0.5f, 0.5f);
		mStandardBox.setPoints(pts);
		pts.clear();

		std::vector<Topology::Triangle> tris;
		tris.resize(12);
		tris[0]  = Topology::Triangle(0, 1, 2);
		tris[1]  = Topology::Triangle(0, 2, 3);
		tris[2]  = Topology::Triangle(0, 4, 5);
		tris[3]  = Topology::Triangle(0, 5, 1);
		tris[4]  = Topology::Triangle(4, 7, 6);
		tris[5]  = Topology::Triangle(4, 6, 5);
		tris[6]  = Topology::Triangle(1, 5, 6);
		tris[7]  = Topology::Triangle(1, 6, 2);
		tris[8]  = Topology::Triangle(2, 6, 7);
		tris[9]  = Topology::Triangle(2, 7, 3);
		tris[10] = Topology::Triangle(0, 3, 7);
		tris[11] = Topology::Triangle(0, 7, 4);
		mStandardBox.setTriangles(tris);
		tris.clear();

		// Load sphere and capsule from OBJ files
		mStandardSphere.loadObjFile(getAssetPath() + "standard/standard_icosahedron.obj");
		mStandardCapsule.loadObjFile(getAssetPath() + "standard/standard_capsule.obj");

		this->varBaseColor()->setValue(Color(0.5, 0.721, 1));
	}

	template<typename TDataType>
	std::string GLDiscreteElementVisualModule<TDataType>::caption()
	{
		return "GLDiscreteElementVisualModule";
	}

	// updateImpl()
	template<typename TDataType>
	void GLDiscreteElementVisualModule<TDataType>::updateImpl()
	{
		auto topo = this->inDiscreteElements()->constDataPtr();

		// Get element arrays
		auto& boxInGlobal = topo->boxesInGlobal();
		auto& sphereInGlobal = topo->spheresInGlobal();
		auto& tetInGlobal = topo->tetsInGlobal();
		auto& capsuleInGlobal = topo->capsulesInGlobal();
		auto& triangleInGlobal = topo->trianglesInGlobal();
		auto& medialConeInGlobal = topo->medialConesInGlobal();
		auto& medialSlabInGlobal = topo->medialSlabsInGlobal();

		// 1. Generate element instance transforms (apply visibility filters)
		uint boxCount = this->varShowBox()->getValue() ? boxInGlobal.size() : 0;
		uint sphereCount = this->varShowSphere()->getValue() ? sphereInGlobal.size() : 0;
		uint capsuleCount = this->varShowCapsule()->getValue() ? capsuleInGlobal.size() : 0;

		// Element color from base class BaseColor (Color -> Vec3f)
		auto bc = this->varBaseColor()->getValue();
		Vec3f elementColor(bc.r, bc.g, bc.b);

		mBoxTransforms.resize(boxCount);
		mSphereTransforms.resize(sphereCount);
		mCapsuleTransforms.resize(capsuleCount);

		if (boxCount > 0)
		{
			cuExecuteNoSync(boxCount,
				GenerateBoxTransforms,
				boxInGlobal, mBoxTransforms);

			mBoxColors.resize(boxCount);
			cuExecuteNoSync(boxCount, FillInstanceColor, mBoxColors, elementColor);
		}
		else
		{
			mBoxColors.clear();
		}

		if (sphereCount > 0)
		{
			cuExecuteNoSync(sphereCount,
				GenerateSphereTransforms,
				sphereInGlobal, mSphereTransforms);

			mSphereColors.resize(sphereCount);
			cuExecuteNoSync(sphereCount, FillInstanceColor, mSphereColors, elementColor);
		}
		else
		{
			mSphereColors.clear();
		}

		if (capsuleCount > 0)
		{
			cuExecuteNoSync(capsuleCount,
				GenerateCapsuleTransforms,
				capsuleInGlobal, mCapsuleTransforms);

			mCapsuleColors.resize(capsuleCount);
			cuExecuteNoSync(capsuleCount, FillInstanceColor, mCapsuleColors, elementColor);
		}
		else
		{
			mCapsuleColors.clear();
		}

		// 2. Generate TriangleSet for MedialCone/MedialSlab/Tet/Triangle
		if (mStandardSphere.isEmpty())
		{
			mStandardSphere.loadObjFile(getAssetPath() + "standard/standard_icosahedron.obj");
			mStandardCapsule.loadObjFile(getAssetPath() + "standard/standard_capsule.obj");
		}

		auto& sphereVertices = mStandardSphere.getPoints();
		auto& sphereIndices = mStandardSphere.triangleIndices();

		int numOfTets = this->varShowTet()->getValue() ? tetInGlobal.size() : 0;
		int numOfTriangles = this->varShowTriangle()->getValue() ? triangleInGlobal.size() : 0;
		int numOfMedialCones = this->varShowMedialCone()->getValue() ? medialConeInGlobal.size() : 0;
		int numOfMedialSlabs = this->varShowMedialSlab()->getValue() ? medialSlabInGlobal.size() : 0;

		int medialConeSurfaceVertices = numOfMedialCones * MEDIAL_RENDER_RESOLUTION * 2;
		int medialConeSurfaceTriangles = numOfMedialCones * MEDIAL_RENDER_RESOLUTION * 2;
		int medialConeSphereVertices = numOfMedialCones * 2 * sphereVertices.size();
		int medialConeSphereTriangles = numOfMedialCones * 2 * sphereIndices.size();

		int medialSlabSurfaceVertices = numOfMedialSlabs * 3 * MEDIAL_RENDER_RESOLUTION * 2;
		int medialSlabSurfaceTriangles = numOfMedialSlabs * 3 * MEDIAL_RENDER_RESOLUTION * 2;
		int medialSlabSphereVertices = numOfMedialSlabs * 3 * sphereVertices.size();
		int medialSlabSphereTriangles = numOfMedialSlabs * 3 * sphereIndices.size();
		int medialSlabCapVertices = numOfMedialSlabs * 6;
		int medialSlabCapTriangles = numOfMedialSlabs * 2;

		int numOfVertices = 4 * numOfTets + 3 * numOfTriangles
			+ medialConeSurfaceVertices + medialConeSphereVertices
			+ medialSlabSurfaceVertices + medialSlabSphereVertices + medialSlabCapVertices;
		int numOfTrianglesTotal = 4 * numOfTets + numOfTriangles
			+ medialConeSurfaceTriangles + medialConeSphereTriangles
			+ medialSlabSurfaceTriangles + medialSlabSphereTriangles + medialSlabCapTriangles;

		mTriVertices.resize(numOfVertices);
		mTriIndices.resize(numOfTrianglesTotal);

		uint vertexOffset = 0;
		uint indexOffset = 0;

		// Setup tets
		if (numOfTets > 0)
		{
			cuExecuteNoSync(numOfTets,
				SetupTetInstances,
				mTriVertices,
				mTriIndices,
				tetInGlobal,
				vertexOffset,
				indexOffset,
				0);

			vertexOffset += numOfTets * 4;
			indexOffset += numOfTets * 4;
		}

		// Setup triangles
		if (numOfTriangles > 0)
		{
			cuExecuteNoSync(numOfTriangles,
				SetupTriangleInstances,
				mTriVertices,
				mTriIndices,
				triangleInGlobal,
				vertexOffset,
				indexOffset);

			vertexOffset += numOfTriangles * 3;
			indexOffset += numOfTriangles;
		}

		// Setup medial cones (surface)
		if (medialConeSurfaceVertices > 0)
		{
			cuExecuteNoSync(medialConeSurfaceVertices,
				SetupVerticesForMedialConeSurfaces,
				mTriVertices,
				medialConeInGlobal,
				MEDIAL_RENDER_RESOLUTION,
				vertexOffset);

			cuExecuteNoSync(medialConeSurfaceTriangles,
				SetupIndicesForMedialConeSurfaces,
				mTriIndices,
				medialConeInGlobal,
				MEDIAL_RENDER_RESOLUTION,
				vertexOffset,
				indexOffset);

			vertexOffset += medialConeSurfaceVertices;
			indexOffset += medialConeSurfaceTriangles;
		}

		// Setup medial cones (sphere caps)
		if (medialConeSphereVertices > 0)
		{
			cuExecuteNoSync(medialConeSphereVertices,
				SetupVerticesForMedialConeSphereInstances,
				mTriVertices,
				sphereVertices,
				medialConeInGlobal,
				vertexOffset);

			cuExecuteNoSync(medialConeSphereTriangles,
				SetupIndicesForMedialConeSphereInstances,
				mTriIndices,
				sphereIndices,
				medialConeInGlobal,
				sphereVertices.size(),
				vertexOffset,
				indexOffset);

			vertexOffset += medialConeSphereVertices;
			indexOffset += medialConeSphereTriangles;
		}

		// Setup medial slabs (edge surfaces)
		if (medialSlabSurfaceVertices > 0)
		{
			cuExecuteNoSync(medialSlabSurfaceVertices,
				SetupVerticesForMedialSlabEdgeSurfaces,
				mTriVertices,
				medialSlabInGlobal,
				MEDIAL_RENDER_RESOLUTION,
				vertexOffset);

			cuExecuteNoSync(medialSlabSurfaceTriangles,
				SetupIndicesForMedialSlabEdgeSurfaces,
				mTriIndices,
				medialSlabInGlobal,
				MEDIAL_RENDER_RESOLUTION,
				vertexOffset,
				indexOffset);

			vertexOffset += medialSlabSurfaceVertices;
			indexOffset += medialSlabSurfaceTriangles;
		}

		// Setup medial slabs (sphere caps)
		if (medialSlabSphereVertices > 0)
		{
			cuExecuteNoSync(medialSlabSphereVertices,
				SetupVerticesForMedialSlabSphereInstances,
				mTriVertices,
				sphereVertices,
				medialSlabInGlobal,
				vertexOffset);

			cuExecuteNoSync(medialSlabSphereTriangles,
				SetupIndicesForMedialSlabSphereInstances,
				mTriIndices,
				sphereIndices,
				medialSlabInGlobal,
				sphereVertices.size(),
				vertexOffset,
				indexOffset);

			vertexOffset += medialSlabSphereVertices;
			indexOffset += medialSlabSphereTriangles;
		}

		// Setup medial slabs (caps)
		if (medialSlabCapVertices > 0)
		{
			cuExecuteNoSync(medialSlabCapVertices,
				SetupVerticesForMedialSlabCaps,
				mTriVertices,
				medialSlabInGlobal,
				vertexOffset);

			cuExecuteNoSync(medialSlabCapTriangles,
				SetupIndicesForMedialSlabCaps,
				mTriIndices,
				medialSlabInGlobal,
				vertexOffset,
				indexOffset);

			vertexOffset += medialSlabCapVertices;
			indexOffset += medialSlabCapTriangles;
		}

		// 3. Synchronize all kernel launches before loading data into buffers
		cuSynchronize();

		// 4. Load TriangleSet data into buffers
		mVertexIndex.load(mTriIndices);
		mVertexPosition.load(mTriVertices);
	}

	// ---------------------------------------------------------------------------
	// GL functions
	// ---------------------------------------------------------------------------

	template<typename TDataType>
	bool GLDiscreteElementVisualModule<TDataType>::initializeGL()
	{
		// create vertex buffer and vertex array object for TriangleSet pass
		mVAO.create();

		mVertexIndex.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mVertexPosition.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);

		// create shader program
		mShaderProgram = Program::createProgramSPIRV(
			SURFACE_VERT, sizeof(SURFACE_VERT),
			SURFACE_FRAG, sizeof(SURFACE_FRAG),
			SURFACE_GEOM, sizeof(SURFACE_GEOM));

		// create shader uniform buffer
		mRenderParamsUBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		mPBRMaterialUBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);

		// create instance passes
		auto initPass = [](InstancePass& pass) {
			pass.vao.create();
			pass.vertexIndex.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
			pass.vertexPosition.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
			pass.instanceTransform.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
			pass.instanceColor.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		};
		initPass(mBoxPass);
		initPass(mSpherePass);
		initPass(mCapsulePass);
		return true;
	}

	template<typename TDataType>
	void GLDiscreteElementVisualModule<TDataType>::releaseGL()
	{
		// release surface rendering resources
		if (mShaderProgram) {
			mShaderProgram->release();
			delete mShaderProgram;
			mShaderProgram = nullptr;
		}
		mVAO.release();
		mVertexIndex.release();
		mVertexPosition.release();
		mRenderParamsUBlock.release();
		mPBRMaterialUBlock.release();

		// release instance passes
		auto releasePass = [](InstancePass& pass) {
			pass.vao.release();
			pass.vertexIndex.release();
			pass.vertexPosition.release();
			pass.instanceTransform.release();
			pass.instanceColor.release();
		};
		releasePass(mBoxPass);
		releasePass(mSpherePass);
		releasePass(mCapsulePass);
	}

	template<typename TDataType>
	void GLDiscreteElementVisualModule<TDataType>::updateGL()
	{
		// 1. Update TriangleSet pass
		mNumTriangles = mVertexIndex.count();
		if (mNumTriangles > 0) {
			mVertexIndex.updateGL();
			mVertexPosition.updateGL();
		}

		// 2. Load template mesh data once (into instance passes)
		if (!mTemplateLoaded)
		{
			bool allLoaded = true;
			auto loadTemplate = [&allLoaded](InstancePass& pass,
				DArray<Topology::Triangle>& templateIndices,
				DArray<Vec3f>& templateVertices)
			{
				pass.numTriangles = templateIndices.size();
				if (pass.numTriangles == 0) { allLoaded = false; return; }
				pass.vertexIndex.load(templateIndices);
				pass.vertexPosition.load(templateVertices);
				pass.vertexIndex.updateGL();
				pass.vertexPosition.updateGL();
			};
			loadTemplate(mBoxPass,    mStandardBox.triangleIndices(),    mStandardBox.getPoints());
			loadTemplate(mSpherePass, mStandardSphere.triangleIndices(), mStandardSphere.getPoints());
			loadTemplate(mCapsulePass, mStandardCapsule.triangleIndices(), mStandardCapsule.getPoints());
			if (allLoaded) mTemplateLoaded = true;
		}

		// 3. Update instance data (transforms + colors) every frame
		auto updateInstance = [](InstancePass& pass,
			DArray<Transform3f>& transforms,
			DArray<Vec3f>& colors)
		{
			pass.instanceCount = transforms.size();
			if (pass.instanceCount == 0 || pass.numTriangles == 0) return;

			pass.instanceTransform.load(transforms);
			pass.instanceColor.load(colors);
			pass.instanceTransform.updateGL();
			pass.instanceColor.updateGL();

			// Re-bind VAO only when instance buffer changes
			pass.vao.bind();
			pass.vertexIndex.bind();
			glEnableVertexAttribArray(0);
			glVertexAttribIPointer(0, 1, GL_INT, sizeof(int), (void*)0);
			pass.vao.bindVertexBuffer(&pass.instanceTransform, 3, 3, GL_FLOAT, sizeof(Transform3f), 0, 1);
			pass.vao.bindVertexBuffer(&pass.instanceTransform, 4, 3, GL_FLOAT, sizeof(Transform3f), sizeof(Vec3f), 1);
			pass.vao.bindVertexBuffer(&pass.instanceTransform, 5, 3, GL_FLOAT, sizeof(Transform3f), 2 * sizeof(Vec3f), 1);
			pass.vao.bindVertexBuffer(&pass.instanceTransform, 6, 3, GL_FLOAT, sizeof(Transform3f), 3 * sizeof(Vec3f), 1);
			pass.vao.bindVertexBuffer(&pass.instanceTransform, 7, 3, GL_FLOAT, sizeof(Transform3f), 4 * sizeof(Vec3f), 1);
			pass.vao.bindVertexBuffer(&pass.instanceColor, 8, 3, GL_FLOAT, sizeof(Vec3f), 0, 1);
			pass.vao.unbind();
		};

		updateInstance(mBoxPass,    mBoxTransforms,    mBoxColors);
		updateInstance(mSpherePass, mSphereTransforms, mSphereColors);
		updateInstance(mCapsulePass, mCapsuleTransforms, mCapsuleColors);

		glCheckError();
	}

	template<typename TDataType>
	void GLDiscreteElementVisualModule<TDataType>::paintGL(const RenderParams& rparams)
	{
		if (mNumTriangles == 0 && mBoxPass.instanceCount == 0 && mSpherePass.instanceCount == 0
			&& mCapsulePass.instanceCount == 0)
			return;

		mShaderProgram->use();

		mShaderProgram->setInt("uVertexNormal", 0);
		mShaderProgram->setInt("uColorMode", 0);

		mRenderParamsUBlock.load((void*)&rparams, sizeof(RenderParams));
		mRenderParamsUBlock.bindBufferBase(0);

		// Helper: load PBR material with given alpha
		auto loadPBR = [&](float alpha) {
			PBRMaterial pbr;
			auto color = this->varBaseColor()->getValue();
			pbr.color = glm::vec3{ color.r, color.g, color.b };
			pbr.metallic = this->varMetallic()->getValue();
			pbr.roughness = this->varRoughness()->getValue();
			pbr.alpha = alpha;
			pbr.useAOTex = 0;
			pbr.useMetallicTex = 0;
			pbr.useRoughnessTex = 0;
			mPBRMaterialUBlock.load((void*)&pbr, sizeof(pbr));
			mPBRMaterialUBlock.bindBufferBase(1);
		};

		// 1. Draw TriangleSet (Tet, MedialCone, MedialSlab, Triangle) with varAlpha
		if (mNumTriangles > 0)
		{
			loadPBR(float(this->varAlpha()->getValue()));
			mVertexPosition.bindBufferBase(8);
			mVAO.bind();
			mShaderProgram->setInt("uInstanced", 0);
			// setup attribute 0
			mVertexIndex.bind();
			glEnableVertexAttribArray(0);
			glVertexAttribIPointer(0, 1, GL_INT, sizeof(int), (void*)0);
			// disable other attributes
			glDisableVertexAttribArray(1);
			glDisableVertexAttribArray(2);
			glDisableVertexAttribArray(3);
			glDisableVertexAttribArray(4);
			glDisableVertexAttribArray(5);
			glDisableVertexAttribArray(6);
			glDisableVertexAttribArray(7);
			glDisableVertexAttribArray(8);
			auto c = this->varBaseColor()->getData();
			glVertexAttrib3f(8, c.r, c.g, c.b);

			glDrawArrays(GL_TRIANGLES, 0, mNumTriangles * 3);
			mVAO.unbind();
		}

		// 2. Draw element instance passes (Box, Sphere, Capsule) with varAlpha
		{
			loadPBR(float(this->varAlpha()->getValue()));
			auto drawPass = [&](InstancePass& pass) {
				if (pass.instanceCount == 0 || pass.numTriangles == 0) return;
				pass.vertexPosition.bindBufferBase(8);
				pass.vao.bind();
				mShaderProgram->setInt("uInstanced", 1);
				glDrawArraysInstanced(GL_TRIANGLES, 0, pass.numTriangles * 3, pass.instanceCount);
				pass.vao.unbind();
			};

			drawPass(mBoxPass);
			drawPass(mSpherePass);
			drawPass(mCapsulePass);
		}

		glCheckError();
	}

	DEFINE_CLASS(GLDiscreteElementVisualModule);

}
