#include "DiscreteElementsToTriangleSet.h"

#include <math.h>

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

	template<typename TDataType>
	DiscreteElementsToTriangleSet<TDataType>::DiscreteElementsToTriangleSet()
		: TopologyMapping()
	{

	}

	template<typename Triangle>
	__global__ void SetupCubeInstances(
		DArray<Vec3f> vertices,
		DArray<Triangle> indices,
		DArray<Box3D> boxes,
		uint pointOffset,
		uint indexOffset,
		uint cubeOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boxes.size()) return;
		
		int idx = tId;
		Box3D box = boxes[idx];

		Vec3f hx = box.u * box.extent[0];
		Vec3f hy = box.v * box.extent[1];
		Vec3f hz = box.w * box.extent[2];

		Vec3f hyz = hy + hz;
		Vec3f hxy = hx + hy;
		Vec3f hxz = hx + hz;

		Vec3f c = box.center;

		Vec3f v0 = c - hx - hyz;
		Vec3f v1 = c + hx - hyz;
		Vec3f v2 = c + hxz - hy;
		Vec3f v3 = c - hxy + hz;

		Vec3f v4 = c - hxz + hy;
		Vec3f v5 = c + hxy - hz;
		Vec3f v6 = c + hx + hyz;
		Vec3f v7 = c - hx + hyz;

		vertices[pointOffset + idx * 8] = v0;
		vertices[pointOffset + idx * 8 + 1] = v1;
		vertices[pointOffset + idx * 8 + 2] = v2;
		vertices[pointOffset + idx * 8 + 3] = v3;
		vertices[pointOffset + idx * 8 + 4] = v4;
		vertices[pointOffset + idx * 8 + 5] = v5;
		vertices[pointOffset + idx * 8 + 6] = v6;
		vertices[pointOffset + idx * 8 + 7] = v7;

		uint offset = idx * 8 + pointOffset;

		indices[indexOffset + idx * 12] = Triangle(offset + 0, offset + 1, offset + 2);
		indices[indexOffset + idx * 12 + 1] = Triangle(offset + 0, offset + 2, offset + 3);

		indices[indexOffset + idx * 12 + 2] = Triangle(offset + 0, offset + 4, offset + 5);
		indices[indexOffset + idx * 12 + 3] = Triangle(offset + 0, offset + 5, offset + 1);

		indices[indexOffset + idx * 12 + 4] = Triangle(offset + 4, offset + 7, offset + 6);
		indices[indexOffset + idx * 12 + 5] = Triangle(offset + 4, offset + 6, offset + 5);

		indices[indexOffset + idx * 12 + 6] = Triangle(offset + 1, offset + 5, offset + 6);
		indices[indexOffset + idx * 12 + 7] = Triangle(offset + 1, offset + 6, offset + 2);

		indices[indexOffset + idx * 12 + 8] = Triangle(offset + 2, offset + 6, offset + 7);
		indices[indexOffset + idx * 12 + 9] = Triangle(offset + 2, offset + 7, offset + 3);

		indices[indexOffset + idx * 12 + 10] = Triangle(offset + 0, offset + 3, offset + 7);
		indices[indexOffset + idx * 12 + 11] = Triangle(offset + 0, offset + 7, offset + 4);
	}

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

	__global__ void SetupVerticesForSphereInstances(
		DArray<Vec3f> vertices,
		DArray<Vec3f> sphereVertices,
		DArray<Sphere3D> sphereInstances,
		uint pointOffset,
		uint sphereOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= sphereInstances.size() * sphereVertices.size()) return;

		uint instanceId = tId / sphereVertices.size();
		uint vertexId = tId % sphereVertices.size();

		Sphere3D sphere = sphereInstances[instanceId];

		Vec3f v = sphereVertices[vertexId];
		vertices[pointOffset + tId] = sphere.center + sphere.radius * sphere.rotation.rotate(v);
	}

	template<typename Triangle>
	__global__ void SetupIndicesForSphereInstances(
		DArray<Triangle> indices,
		DArray<Triangle> sphereIndices,
		DArray<Sphere3D> sphereInstances,
		uint vertexSize,						//vertex size of the instance sphere 
		uint vertexOffset,
		uint indexOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= sphereInstances.size() * sphereIndices.size()) return;

		uint instanceId = tId / sphereIndices.size();
		uint indexId = tId % sphereIndices.size();

		uint curVertexOffset = vertexOffset + instanceId * vertexSize;
		
		Triangle tIndex = sphereIndices[indexId];
		indices[indexOffset + tId] = Triangle(tIndex[0] + curVertexOffset, tIndex[1] + curVertexOffset, tIndex[2] + curVertexOffset);
	}

	__global__ void SetupVerticesForCapsuleInstances(
		DArray<Vec3f> vertices,
		DArray<Vec3f> capsuleVertices,
		DArray<Capsule3D> capsuleInstances,
		uint pointOffset,
		uint capsuleOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= capsuleInstances.size() * capsuleVertices.size()) return;

		uint instanceId = tId / capsuleVertices.size();
		uint vertexId = tId % capsuleVertices.size();

		Capsule3D capsule = capsuleInstances[instanceId];
		float r = capsule.radius;
		float h = capsule.halfLength;
		auto rot = capsule.rotation.toMatrix3x3();
		Vec3f center = capsule.center; 

		Vec3f v = capsuleVertices[vertexId];
		Vec3f orignZ = Vec3f(0, 1, 0);
		Vec3f newZ = Vec3f(0, h, 0);

		if (v.y >= 1)
		{
			vertices[pointOffset + tId] = rot * ((v - orignZ) * r + newZ) + center;
		}
		else if (v.y <= -1) 
		{
			vertices[pointOffset + tId] = rot * ((v + orignZ) * r - newZ) + center;
		}
		else
		{
			vertices[pointOffset + tId] = rot * (v * Vec3f(r, h, r)) + center;
		}	
	}

	template<typename Triangle>
	__global__ void SetupIndicesForCapsuleInstances(
		DArray<Triangle> indices,
		DArray<Triangle> capsuleIndices,
		DArray<Capsule3D> capsuleInstances,
		uint vertexSize,						//vertex size of the instance sphere 
		uint vertexOffset,
		uint indexOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= capsuleInstances.size() * capsuleIndices.size()) return;

		uint instanceId = tId / capsuleIndices.size();
		uint indexId = tId % capsuleIndices.size();

		vertexOffset += instanceId * vertexSize;
		
		Triangle tIndex = capsuleIndices[indexId];
		indices[indexOffset + tId] = Triangle(tIndex[0] + vertexOffset, tIndex[1] + vertexOffset, tIndex[2] + vertexOffset);
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

	template<typename TDataType>
	bool DiscreteElementsToTriangleSet<TDataType>::apply()
	{
		if (this->outTriangleSet()->isEmpty())
		{
			this->outTriangleSet()->allocate();
		}

		if (mStandardSphere.isEmpty())
		{
			mStandardSphere.loadObjFile(getAssetPath() + "standard/standard_icosahedron.obj");
			mStandardCapsule.loadObjFile(getAssetPath() + "standard/standard_capsule.obj");
		}
		auto inTopo = this->inDiscreteElements()->constDataPtr();

		DArray<Box3D>& boxInGlobal = inTopo->boxesInGlobal();
		DArray<Sphere3D>& sphereInGlobal = inTopo->spheresInGlobal();
		DArray<Tet3D>& tetInGlobal = inTopo->tetsInGlobal();
		DArray<Capsule3D>& capsuleInGlobal = inTopo->capsulesInGlobal();
		DArray<MedialCone3D>& medialConeInGlobal = inTopo->medialConesInGlobal();
		DArray<MedialSlab3D>& medialSlabInGlobal = inTopo->medialSlabsInGlobal();

		ElementOffset elementOffset = inTopo->calculateElementOffset();

		int numOfSpheres = sphereInGlobal.size();
		int numofCaps = capsuleInGlobal.size();
		int numOfBoxes = boxInGlobal.size();
		int numOfTets = tetInGlobal.size();
		int numOfMedialCones = medialConeInGlobal.size();
		int numOfMedialSlabs = medialSlabInGlobal.size();
		
		auto triSet = this->outTriangleSet()->getDataPtr();

		auto& vertices = triSet->getPoints();
		auto& indices = triSet->triangleIndices();

		auto& sphereVertices = mStandardSphere.getPoints();
		auto& sphereIndices = mStandardSphere.triangleIndices();

		auto& capsuleVertices = mStandardCapsule.getPoints();
		auto& capsuleIndices = mStandardCapsule.triangleIndices();

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
		
		int numOfVertices = 8 * numOfBoxes + 4 * numOfTets + sphereVertices.size() * numOfSpheres + capsuleVertices.size() * numofCaps
			+ medialConeSurfaceVertices + medialConeSphereVertices
			+ medialSlabSurfaceVertices + medialSlabSphereVertices + medialSlabCapVertices;
		int numOfTriangles = 12 * numOfBoxes + 4 * numOfTets + sphereIndices.size() * numOfSpheres + capsuleIndices.size() * numofCaps
			+ medialConeSurfaceTriangles + medialConeSphereTriangles
			+ medialSlabSurfaceTriangles + medialSlabSphereTriangles + medialSlabCapTriangles;

		vertices.resize(numOfVertices);
		indices.resize(numOfTriangles);

		uint vertexOffset = 0;
		uint indexOffset = 0;

		//Setup spheres
		cuExecute(numOfSpheres * sphereVertices.size(),
			SetupVerticesForSphereInstances,
			vertices,
			sphereVertices,
			sphereInGlobal,
			vertexOffset,
			elementOffset.sphereIndex());

		cuExecute(numOfSpheres * sphereIndices.size(),
			SetupIndicesForSphereInstances,
			indices,
			sphereIndices,
			sphereInGlobal,
			sphereVertices.size(),
			vertexOffset,
			indexOffset);

		vertexOffset += numOfSpheres * sphereVertices.size();
		indexOffset += numOfSpheres * sphereIndices.size();

		//Setup boxes
		cuExecute(numOfBoxes,
			SetupCubeInstances,
			vertices,
			indices,
			boxInGlobal,
			vertexOffset,
			indexOffset,
			elementOffset.boxIndex());

		vertexOffset += numOfBoxes * 8;
		indexOffset += numOfBoxes * 12;

		//Setup tets
		cuExecute(numOfTets,
			SetupTetInstances,
			vertices,
			indices,
			tetInGlobal,
			vertexOffset,
			indexOffset,
			elementOffset.tetIndex());

		vertexOffset += numOfTets * 4;
		indexOffset += numOfTets * 4;

		cuExecute(numofCaps * capsuleVertices.size(),
			SetupVerticesForCapsuleInstances,
			vertices,
			capsuleVertices,
			capsuleInGlobal,
			vertexOffset,
			elementOffset.capsuleIndex());

		cuExecute(numofCaps * capsuleIndices.size(),
			SetupIndicesForCapsuleInstances,
			indices,
			capsuleIndices,
			capsuleInGlobal,
			capsuleVertices.size(),
			vertexOffset,
			indexOffset);

		vertexOffset += numofCaps * capsuleVertices.size();
		indexOffset += numofCaps * capsuleIndices.size();

		cuExecute(medialConeSurfaceVertices,
			SetupVerticesForMedialConeSurfaces,
			vertices,
			medialConeInGlobal,
			MEDIAL_RENDER_RESOLUTION,
			vertexOffset);

		cuExecute(medialConeSurfaceTriangles,
			SetupIndicesForMedialConeSurfaces,
			indices,
			medialConeInGlobal,
			MEDIAL_RENDER_RESOLUTION,
			vertexOffset,
			indexOffset);

		vertexOffset += medialConeSurfaceVertices;
		indexOffset += medialConeSurfaceTriangles;

		cuExecute(medialConeSphereVertices,
			SetupVerticesForMedialConeSphereInstances,
			vertices,
			sphereVertices,
			medialConeInGlobal,
			vertexOffset);

		cuExecute(medialConeSphereTriangles,
			SetupIndicesForMedialConeSphereInstances,
			indices,
			sphereIndices,
			medialConeInGlobal,
			sphereVertices.size(),
			vertexOffset,
			indexOffset);

		vertexOffset += medialConeSphereVertices;
		indexOffset += medialConeSphereTriangles;

		cuExecute(medialSlabSurfaceVertices,
			SetupVerticesForMedialSlabEdgeSurfaces,
			vertices,
			medialSlabInGlobal,
			MEDIAL_RENDER_RESOLUTION,
			vertexOffset);

		cuExecute(medialSlabSurfaceTriangles,
			SetupIndicesForMedialSlabEdgeSurfaces,
			indices,
			medialSlabInGlobal,
			MEDIAL_RENDER_RESOLUTION,
			vertexOffset,
			indexOffset);

		vertexOffset += medialSlabSurfaceVertices;
		indexOffset += medialSlabSurfaceTriangles;

		cuExecute(medialSlabSphereVertices,
			SetupVerticesForMedialSlabSphereInstances,
			vertices,
			sphereVertices,
			medialSlabInGlobal,
			vertexOffset);

		cuExecute(medialSlabSphereTriangles,
			SetupIndicesForMedialSlabSphereInstances,
			indices,
			sphereIndices,
			medialSlabInGlobal,
			sphereVertices.size(),
			vertexOffset,
			indexOffset);

		vertexOffset += medialSlabSphereVertices;
		indexOffset += medialSlabSphereTriangles;

		cuExecute(medialSlabCapVertices,
			SetupVerticesForMedialSlabCaps,
			vertices,
			medialSlabInGlobal,
			vertexOffset);

		cuExecute(medialSlabCapTriangles,
			SetupIndicesForMedialSlabCaps,
			indices,
			medialSlabInGlobal,
			vertexOffset,
			indexOffset);

		vertexOffset += medialSlabCapVertices;
		indexOffset += medialSlabCapTriangles;

		this->outTriangleSet()->getDataPtr()->update();

		return true;
	}

	DEFINE_CLASS(DiscreteElementsToTriangleSet);
}
