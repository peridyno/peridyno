#include "NormalForce.h"

namespace dyno
{

	template<typename TDataType>
	NormalForce<TDataType>::NormalForce()
		:ConstraintModule()
	{
		this->outNormalForce()->allocate();
	}


	template<typename TDataType>
	NormalForce<TDataType>::~NormalForce()
	{
		mNormalForceFlag.clear();
	}



	template<typename Real, typename Coord>
	__global__ void NormalForce_UpdateVelocity(
		DArray<Coord> forces,
		DArray<bool> flags,
		DArray<Coord> Velocities,
		DArray<Coord> positions,
		DArray<Coord> normals,
		Real dt,
		Real scale
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= forces.size()) return;
		
		if (!flags[pId])
		{
			forces[pId] = Coord(0.0f);
			Velocities[pId] = Coord(0.0f);
			return;

		}

		forces[pId] = normals[pId] * scale;

		Velocities[pId] += dt * (forces[pId]);
	}


	template<typename Coord>
	Real __device__ PointToEdgeDistance(Coord O, Coord A, Coord B)
	{
		Coord AB = B - A;
		Coord AO = O - A;
		Coord AP = AB.dot(AO) * AB;
		Coord P = A + AP;
		return (O - P).norm();
	}


	template<typename Coord>
	Real __device__ PointToMeshDistance(Coord O, Coord A, Coord B, Coord C, Coord normal)
	{
		Coord AB = B - A;
		Coord AC = C - A;
		Coord AO = O - A;
		return AO.dot(normal) / normal.norm();

	}


	template<typename Coord>
	__global__ void NormalForce_EmptyEdge(
		DArray<bool> NormalForceFlag,
		DArray<Coord> positions,
		DArrayList<int> triangle_neighbors,
		DArray<int> particleMeshID,
		DArray<Coord> vertices,
		DArray<TopologyModule::Triangle> triangles,
		DArray<Coord> noramls,
		DArray<TopologyModule::Edg2Tri> Edg2Tri,
		DArray<TopologyModule::Tri2Edg> Tri2Edg
	) {
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= positions.size()) return;

		NormalForceFlag[pId] = true;

		List<int>& nbrTriIds_i = triangle_neighbors[pId];
		int nbrSize = nbrTriIds_i.size();

		Real min_d = 0.0f;
		int min_triID = -1;
		
		int triangleID = particleMeshID[pId];
		int& A_id = triangles[triangleID][0];
		int& B_id = triangles[triangleID][1];
		int& C_id = triangles[triangleID][2];

		//if (pId % 1000 == 0)
		//	printf("%d - %d, %d, %d\r\n", pId, A_id, B_id, C_id);

		Real d_ab = PointToEdgeDistance(positions[pId], vertices[A_id], vertices[B_id]);
		Real d_bc = PointToEdgeDistance(positions[pId], vertices[B_id], vertices[C_id]);
		Real d_ac = PointToEdgeDistance(positions[pId], vertices[A_id], vertices[C_id]);


		Real distance = PointToMeshDistance(positions[pId],
			vertices[triangles[triangleID][0]],
			vertices[triangles[triangleID][1]],
			vertices[triangles[triangleID][2]],
			noramls[pId]
		);

		if (distance > 0) {
			NormalForceFlag[pId] = false;
		}
		

	}




	template<typename TDataType>
	void NormalForce<TDataType>::constrain()
	{
		//std::cout << "Normal Force!" << std::endl;
		int num = this->inPosition()->size();

		if (this->outNormalForce()->isEmpty())
			this->outNormalForce()->allocate();

		if (this->outNormalForce()->size() != num)
		{
			this->outNormalForce()->resize(num);
		}

		if (mNormalForceFlag.size() != num)
		{
			mNormalForceFlag.resize(num);
		}

		auto ts = this->inTriangleSet()->constDataPtr();
		//ts->updateTriangle2Edge();
		auto& vertices = ts->getPoints();
		auto& triangles = ts->getTriangles();
		auto& edge2tri = ts->getEdge2Triangle();
		auto& tri2edge = ts->getTriangle2Edge();

		//std::cout << edge2tri.size() << ", " << tri2edge.size() << std::endl;

		cuExecute(num, NormalForce_EmptyEdge,
			mNormalForceFlag,
			this->inPosition()->getData(),
			this->inTriangleNeighborIds()->getData(),
			this->inParticleMeshID()->getData(),
			vertices,
			triangles,
			this->inParticleNormal()->getData(),
			edge2tri,
			tri2edge
		);

		cuExecute(num, NormalForce_UpdateVelocity,
			this->outNormalForce()->getData(),
			mNormalForceFlag,
			this->inVelocity()->getData(),
			this->inPosition()->getData(),
			this->inParticleNormal()->getData(),
			this->inTimeStep()->getValue(),
			this->varStrength()->getValue()
		);



	}



	DEFINE_CLASS(NormalForce);
}