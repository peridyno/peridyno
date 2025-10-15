#include "DiscreteElementsToTriangleSet.h"

namespace dyno
{
	typedef typename ::dyno::TOrientedBox3D<Real> Box3D;

	template<typename TDataType>
	DiscreteElementsToTriangleSet<TDataType>::DiscreteElementsToTriangleSet()
		: TopologyMapping()
	{
		mStandardSphere.loadObjFile(getAssetPath() + "standard/standard_icosahedron.obj");
		mStandardCapsule.loadObjFile(getAssetPath() + "standard/standard_capsule.obj");
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

	__global__ void SetupVerticesForMedialConeInstances(
		DArray<Vec3f> vertices,
		DArray<Vec3f> sphereVertices,
		DArray<MedialCone3D> coneInstances,
		uint pointOffset,
		uint coneOffset
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= coneInstances.size() * sphereVertices.size() * 2) return; 

		uint instanceId = tId / (sphereVertices.size() * 2); 
		uint remaining = tId % (sphereVertices.size() * 2);
		uint sphereId = remaining / sphereVertices.size(); 
		uint vertexId = remaining % sphereVertices.size();

		if (instanceId >= coneInstances.size()) return;

		MedialCone3D cone = coneInstances[instanceId];
		Vec3f v = sphereVertices[vertexId];

		vertices[pointOffset + tId] = cone.v[sphereId] + cone.radius[sphereId] * v;
	}

	template<typename Triangle>
	__global__ void SetupIndicesForMedialConeInstances(
		DArray<Triangle> indices,
		DArray<Triangle> sphereIndices,
		DArray<MedialCone3D> coneInstances,
		uint vertexSize,        
		uint vertexDataStartOffset, 
		uint indexOutputStartOffset 
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= coneInstances.size() * sphereIndices.size() * 2) return;

		uint instanceId = tId / (sphereIndices.size() * 2);
		uint remaining = tId % (sphereIndices.size() * 2);
		uint sphereId = remaining / sphereIndices.size();
		uint indexId = remaining % sphereIndices.size();

		if (instanceId >= coneInstances.size()) return;


		uint baseVertexOffsetForInstance = vertexDataStartOffset + instanceId * vertexSize * 2;
		uint vertexOffsetForSphere = baseVertexOffsetForInstance + sphereId * vertexSize;

		Triangle tIndex = sphereIndices[indexId];
		indices[indexOutputStartOffset + tId] = Triangle( 
			tIndex[0] + vertexOffsetForSphere,
			tIndex[1] + vertexOffsetForSphere,
			tIndex[2] + vertexOffsetForSphere
		);
	}

	__global__ void SetupVerticesForMedialSlabInstances(
		DArray<Vec3f> vertices,
		DArray<Vec3f> sphereVertices,
		DArray<MedialSlab3D> slabInstances,
		uint pointOffset,
		uint slabOffset
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= slabInstances.size() * sphereVertices.size() * 3) return; 

		uint instanceId = tId / (sphereVertices.size() * 3);
		uint remaining = tId % (sphereVertices.size() * 3);
		uint sphereId = remaining / sphereVertices.size(); 
		uint vertexId = remaining % sphereVertices.size();

		if (instanceId >= slabInstances.size()) return;

		MedialSlab3D slab = slabInstances[instanceId];
		Vec3f v = sphereVertices[vertexId];

		vertices[pointOffset + tId] = slab.v[sphereId] + slab.radius[sphereId] * v;
	}


	template<typename Triangle>
	__global__ void SetupIndicesForMedialSlabInstances(
		DArray<Triangle> indices,
		DArray<Triangle> sphereIndices,
		DArray<MedialSlab3D> slabInstances,
		uint vertexSize,       
		uint vertexDataStartOffset, 
		uint indexOutputStartOffset
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= slabInstances.size() * sphereIndices.size() * 3) return;

		uint instanceId = tId / (sphereIndices.size() * 3);
		uint remaining = tId % (sphereIndices.size() * 3);
		uint sphereId = remaining / sphereIndices.size();
		uint indexId = remaining % sphereIndices.size();

		if (instanceId >= slabInstances.size()) return;

		
		uint baseVertexOffsetForInstance = vertexDataStartOffset + instanceId * vertexSize * 3;
		uint vertexOffsetForSphere = baseVertexOffsetForInstance + sphereId * vertexSize;

		Triangle tIndex = sphereIndices[indexId];
		indices[indexOutputStartOffset + tId] = Triangle( 
			tIndex[0] + vertexOffsetForSphere,
			tIndex[1] + vertexOffsetForSphere,
			tIndex[2] + vertexOffsetForSphere
		);
	}
	template<typename Triangle>
	__global__ void SetupIndicesForSphereInstances(
		DArray<Triangle> indices,
		DArray<Triangle> sphereIndices,
		DArray<Sphere3D> sphereInstances,
		uint vertexSize,						
		uint vertexDataStartOffset,         
		uint indexOutputStartOffset         
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= sphereInstances.size() * sphereIndices.size()) return;

		uint instanceId = tId / sphereIndices.size();
		uint indexId = tId % sphereIndices.size();

		int calculatedVertexOffsetForInstance = vertexDataStartOffset + instanceId * vertexSize; // 使用正确的顶点数据起始偏移量

		Triangle tIndex = sphereIndices[indexId];
		indices[indexOutputStartOffset + tId] = Triangle(tIndex[0] + calculatedVertexOffsetForInstance, tIndex[1] + calculatedVertexOffsetForInstance, tIndex[2] + calculatedVertexOffsetForInstance);
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

	template<typename TDataType>
	bool DiscreteElementsToTriangleSet<TDataType>::apply()
	{
		if (this->outTriangleSet()->isEmpty())
		{
			this->outTriangleSet()->allocate();
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
		
		int numOfVertices = 8 * numOfBoxes + 4 * numOfTets + sphereVertices.size() * numOfSpheres + capsuleVertices.size() * numofCaps + 2 * sphereVertices.size() * numOfMedialCones + 3 * sphereVertices.size() * numOfMedialSlabs;
		int numOfTriangles = 12 * numOfBoxes + 4 * numOfTets + sphereIndices.size() * numOfSpheres + capsuleIndices.size() * numofCaps + 2 * sphereIndices.size() * numOfMedialCones + 3 * sphereIndices.size() * numOfMedialSlabs;

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

		cuExecute(numOfMedialCones * sphereVertices.size() * 2,
			SetupVerticesForMedialConeInstances,
			vertices,
			sphereVertices,
			medialConeInGlobal,
			vertexOffset,
			elementOffset.medialConeIndex());

		cuExecute(numOfMedialCones* sphereIndices.size()* 2	,
			SetupIndicesForMedialConeInstances,
			indices,
			sphereIndices,
			medialConeInGlobal,
			sphereVertices.size(),
			vertexOffset,
			indexOffset);

		vertexOffset += numOfMedialCones * sphereVertices.size() * 2;
		indexOffset += numOfMedialCones * sphereIndices.size() * 2;

		cuExecute(numOfMedialSlabs* sphereVertices.size() * 3,
			SetupVerticesForMedialSlabInstances,
			vertices,
			sphereVertices,
			medialSlabInGlobal,
			vertexOffset,
			elementOffset.medialSlabIndex());

		cuExecute(numOfMedialSlabs* sphereIndices.size() * 3,
			SetupIndicesForMedialSlabInstances,
			indices,
			sphereIndices,
			medialSlabInGlobal,
			sphereVertices.size(),
			vertexOffset,
			indexOffset);

		vertexOffset += numOfMedialSlabs * sphereVertices.size() * 3;
		indexOffset += numOfMedialSlabs * sphereIndices.size() * 3;

		this->outTriangleSet()->getDataPtr()->update();

		return true;
	}

	DEFINE_CLASS(DiscreteElementsToTriangleSet);
}