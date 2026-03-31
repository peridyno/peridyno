#include "TetrahedronSet.h"
#include <fstream>
#include <iostream>
#include <sstream>

#include <thrust/sort.h>
#include <STL/Set.h>

#include <Primitive/Primitive3D.h>

namespace dyno
{
	const int maxNum = 100;	// upper limitation
	const int threadNum = 32;
	const int sharedThreadSkip = maxNum;
	const int sharedBlockSkip = maxNum * threadNum;
	const int sharedBlockSize = sharedBlockSkip * (int)(sizeof(int));
	template<typename TDataType>
	TetrahedronSet<TDataType>::TetrahedronSet()
		: TriangleSet<TDataType>()
	{

	}

	template<typename TDataType>
	TetrahedronSet<TDataType>::~TetrahedronSet()
	{
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::setTetrahedrons(std::vector<Tetrahedron>& tetrahedrons)
	{
		// std::vector<Triangle> triangles;

		mTethedrons.resize(tetrahedrons.size());
		mTethedrons.assign(tetrahedrons);

	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::setTetrahedrons(DArray<Tetrahedron>& tetrahedrons)
	{
		if (tetrahedrons.size() != mTethedrons.size())
		{
			mTethedrons.resize(tetrahedrons.size());
		}

		mTethedrons.assign(tetrahedrons);
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::loadTetFile(std::string filename)
	{
		std::string filename_node = filename;	filename_node.append(".node");
		std::string filename_ele = filename;	filename_ele.append(".ele");

		std::ifstream infile_node(filename_node);
		std::ifstream infile_ele(filename_ele);
		if (!infile_node || !infile_ele) {
			std::cerr << "Failed to open the tetrahedron file. Terminating.\n";
			exit(-1);
		}

		std::string line;
		std::getline(infile_node, line);
		std::stringstream ss_node(line);

		int node_num;
		ss_node >> node_num;
		std::vector<Coord> nodes;
		for (int i = 0; i < node_num; i++)
		{
			std::getline(infile_node, line);
			std::stringstream data(line);
			int id;
			Coord v;
			data >> id >> v[0] >> v[1] >> v[2];
			nodes.push_back(v);
		}


		std::getline(infile_ele, line);
		std::stringstream ss_ele(line);

		int ele_num;
		ss_ele >> ele_num;
		std::vector<Triangle> tris;
		std::vector<Tetrahedron> tets;
		for (int i = 0; i < ele_num; i++)
		{
			std::getline(infile_ele, line);
			std::stringstream data(line);
			int id;
			Tetrahedron tet;
			data >> id >> tet[0] >> tet[1] >> tet[2] >> tet[3];
			tet[0] -= 0;
			tet[1] -= 0;
			tet[2] -= 0;
			tet[3] -= 0;
			tets.push_back(tet);

			tris.push_back(Triangle(tet[0], tet[1], tet[2]));
			tris.push_back(Triangle(tet[0], tet[3], tet[1]));
			tris.push_back(Triangle(tet[1], tet[3], tet[2]));
			tris.push_back(Triangle(tet[0], tet[2], tet[3]));
		}

		this->setPoints(nodes);

		this->setTriangles(tris);
		this->setTetrahedrons(tets);
		this->update();
	}

	template<typename Tetrahedron>
	__global__ void TetSet_CountTets(
		DArray<uint> counter,
		DArray<Tetrahedron> tets)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tets.size()) return;

		Tetrahedron t = tets[tId];

		atomicAdd(&counter[t[0]], 1);
		atomicAdd(&counter[t[1]], 1);
		atomicAdd(&counter[t[2]], 1);
		atomicAdd(&counter[t[3]], 1);
	}

	template<typename Tetrahedron>
	__global__ void TetSet_SetupTetIds(
		DArrayList<int> tetIds,
		DArray<Tetrahedron> tets)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tets.size()) return;

		Tetrahedron t = tets[tId];

		tetIds[t[0]].atomicInsert(tId);
		tetIds[t[1]].atomicInsert(tId);
		tetIds[t[2]].atomicInsert(tId);
		tetIds[t[3]].atomicInsert(tId);
	}


	template<typename Real, typename Coord, typename Tetrahedron>
	__global__ void TetSet_CalculateVolume(
		DArray<Real> volume,
		DArray<Coord> vertices,
		DArray<Tetrahedron> indices)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= volume.size()) return;

		Real vol_i = Real(0);

		Tetrahedron tetIndex = indices[tId];

		TTet3D<Real> tet(vertices[tetIndex[0]], vertices[tetIndex[1]], vertices[tetIndex[2]], vertices[tetIndex[3]]);

		volume[tId] = abs(tet.volume());
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::calculateVolume(DArray<Real>& volume)
	{
		auto& tetIndices = this->tetrahedronIndices();
		uint tetSize = tetIndices.size();

		if (volume.size() != tetSize)
			volume.resize(tetSize);

		cuExecute(tetSize,
			TetSet_CalculateVolume,
			volume,
			this->getPoints(),
			tetIndices);
	}

	struct Info
	{
		DYN_FUNC Info() {
			tetId = EMPTY;
			tIndex = Topology::Triangle(EMPTY, EMPTY, EMPTY);
		}

		DYN_FUNC Info(int id, Topology::Triangle index) {
			tetId = id;
			tIndex = index;
		};

		int tetId;
		Topology::Triangle tIndex;
	};

	template<typename TKey, typename Tetrahedron>
	__global__ void TetSet_SetupKeys(
		DArray<TKey> keys,
		DArray<Info> ids,
		DArray<Tetrahedron> tets)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tets.size()) return;

		Tetrahedron tet = tets[tId];
		keys[4 * tId] = TKey(tet[0], tet[1], tet[2]);
		keys[4 * tId + 1] = TKey(tet[1], tet[2], tet[3]);
		keys[4 * tId + 2] = TKey(tet[2], tet[3], tet[0]);
		keys[4 * tId + 3] = TKey(tet[3], tet[0], tet[1]);

		ids[4 * tId] = Info(tId, Topology::Triangle(tet[0], tet[1], tet[2]));
		ids[4 * tId + 1] = Info(tId, Topology::Triangle(tet[1], tet[2], tet[3]));
		ids[4 * tId + 2] = Info(tId, Topology::Triangle(tet[2], tet[3], tet[0]));
		ids[4 * tId + 3] = Info(tId, Topology::Triangle(tet[3], tet[0], tet[1]));
	}

	template<typename TKey>
	__global__ void TetSet_CountTriangleNumber(
		DArray<int> counter,
		DArray<TKey> keys)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= keys.size()) return;

		if (tId == 0 || keys[tId] != keys[tId - 1])
			counter[tId] = 1;	// The first key different triangle
		else
			counter[tId] = 0;
	}

	__global__ void TetSet_CheckSurTri(
		DArray<int> surfaceTri)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= surfaceTri.size()) return;

		printf("sur[%d] %d ", tId, surfaceTri[tId]);

	}


	template<typename Triangle, typename Tri2Tet, typename Tet2Tri, typename TKey>
	__global__ void TetSet_SetupTriangles(
		DArray<Triangle> triangles,
		DArray<Tri2Tet> tri2Tet,
		DArray<Tet2Tri> tet2Tri,
		DArray<int> numTet2Tri,
		DArray<TKey> keys,
		DArray<int> counter,
		DArray<Info> tetIds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= keys.size()) return;

		int shift = counter[tId]; // exclusive scan
		if (tId == 0 || keys[tId] != keys[tId - 1])
		{
			TKey key = keys[tId];
			triangles[shift] = tetIds[tId].tIndex;

			Tri2Tet t2T(EMPTY, EMPTY);
			t2T[0] = tetIds[tId].tetId;

			if (tId + 1 < keys.size() && keys[tId + 1] == key)
			{
				t2T[1] = tetIds[tId + 1].tetId;

			}

			tri2Tet[shift] = t2T;
		} else shift -= 1; // exclusive scan

		// Tet2Tri
		int index = atomicAdd(&numTet2Tri[tetIds[tId].tetId], 1);
		tet2Tri[tetIds[tId].tetId][index] = shift;
	}


	template<typename Triangle, typename Tri2Tet, typename Tet2Tri, typename TKey>
	__global__ void TetSet_CheckTriangles(
		DArray<Triangle> triangles,
		DArray<Tri2Tet> tri2Tet,
		DArray<Tet2Tri> tet2Tri,
		DArray<int> surfaceTri,
		DArray<int> numTet2Tri,
		DArray<TKey> keys,
		DArray<int> counter,
		DArray<Info> tetIds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= keys.size()) return;

		int tetId = tetIds[tId].tetId;
		if(numTet2Tri[tetId] != 4)
		{
			printf("[Error] there is a tetrahedron with %d(!=4) triangles\n", numTet2Tri[tetIds[tId].tetId]);
		}

		for(int i = 0; i < numTet2Tri[tetId]; ++i)
		{
			int triId = tet2Tri[tetId][i];
			if(tri2Tet[triId][0] != tetId && tri2Tet[triId][1] != tetId)
			{
				printf("[Error] tetrahedron %d is not in triangle %d,"
							    "but belong to tet %d %d \n"
								, tetId, triId, tri2Tet[triId][0], tri2Tet[triId][1]);
			}
		}
		if(tId < surfaceTri.size() && surfaceTri[tId] != EMPTY)
		{
			if (tId != 0 && surfaceTri[tId] < surfaceTri[tId - 1])
			{
				printf("[Error] surface triangle %d is not sorted\n", tId);
			}
			int triId = surfaceTri[tId];
			if(tri2Tet[triId][0] != EMPTY && tri2Tet[triId][1] != EMPTY)
			{
				printf("[Error] surface triangle %d is not belong to one tetrahedron\n", triId);
			}
		}

	}	

	template<typename TKey>
	void printTKey(DArray<TKey> keys, int maxLength) {
		CArray<TKey> h_keys;
		h_keys.resize(keys.size());
		h_keys.assign(keys);

		int psize = min((int)h_keys.size(), maxLength);
		for (int i = 0; i < psize; i++)
		{
			printf("%d: %d %d %d \n", i, h_keys[i][0], h_keys[i][1], h_keys[i][2]);
		}

		h_keys.clear();
	}

	void printCount(DArray<int> keys, int maxLength) {
		CArray<int> h_keys;
		h_keys.resize(keys.size());
		h_keys.assign(keys);

		int psize = minimum((int)h_keys.size(), maxLength);
		for (int i = 0; i < psize; i++)
		{
			printf("%d: %d \n", i, h_keys[i]);
		}

		h_keys.clear();
	}

	DYN_FUNC bool oppVertex(int tId, int& oppVerId0, int& oppVerId1,
		Topology::Tri2Tet& t2t,
		Topology::Tetrahedron& tetA,
		Topology::Tetrahedron& tetB)
	{
		if(t2t[1] == EMPTY) return false;

		int b = 0;
		for(int i = 0; i < 4; i++)
		{
			b ^= tetA[i];
			b ^= tetB[i];
		}
		// b = oppVerId0 xor oppVerId1 

		// find lowbit of oppVerId0 xor oppVerId1
		b = b & (-b);
		
		oppVerId0 = oppVerId1 = 0;

		for (int i = 0; i < 4; i++)
		{
			if (tetA[i] & b) oppVerId0 ^= tetA[i];
			else			 oppVerId1 ^= tetA[i];
			if (tetB[i] & b) oppVerId0 ^= tetB[i];
			else			 oppVerId1 ^= tetB[i];
		}
		// oppVerId0/1 is the opposite vertex of tri tId in tetA/B
		// but oppVerId0 is not always belong to tetA, may belong to tetB.
		return true;
	}

	__global__ void TetSet_NeigCountEdgeNumber(
		DArray<uint> num,
		DArray<Topology::Edge> edges)
	{
		int eId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (eId >= edges.size()) return;

		Topology::Edge edge = edges[eId];

		atomicAdd(&(num[edge[0]]), 1);
		atomicAdd(&(num[edge[1]]), 1);
	}

	__global__ void TetSet_NeigCountTriNumber(
		DArray<uint> num,
		DArray<Topology::Tetrahedron> tets,
		DArray<Topology::Tri2Tet> tri2Tet)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tri2Tet.size()) return;

		int v0 = 0; int v1 = 0;
		if(tri2Tet[tId][1] != EMPTY && 
			oppVertex(tId, v0, v1, tri2Tet[tId], tets[tri2Tet[tId][0]], tets[tri2Tet[tId][1]])) 
		{
			atomicAdd(&(num[v0]), 1);
			atomicAdd(&(num[v1]), 1);
		}
	}

	__global__ void TetSet_NeigCount2HopNumber(
		DArray<uint> num,
		DArrayList<int> list_1hop)
	{
		int bId = blockIdx.x;
		int tId = threadIdx.x;
		int pId = tId + bId * blockDim.x;
		if (pId >= num.size()) return;

        //extern __shared__ int sBuf[];	// dynamic
		__shared__ int sBuf[sharedBlockSkip];	// static

        int* st_addr = sBuf + tId * (int)sharedThreadSkip;

        Set<int> set;
        int* setBuf = (int*)(st_addr);
        set.reserve(setBuf, sharedThreadSkip);

		int i = pId;
		int size_i = list_1hop[i].size();
		int c = 0;
		for (int i0 = 0; i0 < size_i; i0++)
		{
			int j = list_1hop[i][i0];
			if (i == j) continue;
			set.insert(j);	// 1-hop
			int size_j = list_1hop[j].size();
			for (int j0 = 0; j0 < size_j; j0++)
			{
				int k = list_1hop[j][j0];
				if (i == k) continue;
				set.insert(k); // 2-hop
			}
		}
		num[pId] = set.size();
	}

	__global__ void TetSet_NeigStore2HopIds(
		DArrayList<int> list_2hop,
		DArrayList<int> list_1hop,
		DArray<uint> num)
	{
		int bId = blockIdx.x;
		int tId = threadIdx.x;
		int pId = tId + bId * blockDim.x;
		if (pId >= num.size()) return;

		//extern __shared__ int sBuf[];	// dynamic
		__shared__ int sBuf[sharedBlockSkip];	// static

		int* st_addr = sBuf + tId * (int)sharedThreadSkip;

		Set<int> set;
		int* setBuf = (int*)(st_addr);
		set.reserve(setBuf, sharedThreadSkip);

		int i = pId;
		int size_i = list_1hop[i].size();
		for (int i0 = 0; i0 < size_i; i0++)
		{
			int j = list_1hop[i][i0];
			if (i == j) continue;
			set.insert(j);	// 1-hop
			int size_j = list_1hop[j].size();
			for (int j0 = 0; j0 < size_j; j0++)
			{
				int k = list_1hop[j][j0];
				if (i == k) continue;
				set.insert(k); // 2-hop
			}
		}
		
		for (auto it = set.begin(); it != set.end(); it++)
		{
			list_2hop[i].atomicInsert(*it);
		}
	}

	__global__ void TetSet_NeigStoreEdgeIds(
		DArrayList<int> ids,
		DArray<Topology::Edge> edges)
	{
		int eId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (eId >= edges.size()) return;

		Topology::Edge edge = edges[eId];
		int v0 = edge[0];
		int v1 = edge[1];

		ids[v0].atomicInsert(v1);
		ids[v1].atomicInsert(v0);
	}

	__global__ void TetSet_NeigStoreTriIds(
		DArrayList<int> ids,
		DArray<Topology::Tetrahedron> tets,
		DArray<Topology::Tri2Tet> tri2Tet)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tri2Tet.size()) return;

		int v0 = 0; int v1 = 0;
		if(tri2Tet[tId][1] != EMPTY && 
			oppVertex(tId, v0, v1, tri2Tet[tId], tets[tri2Tet[tId][0]], tets[tri2Tet[tId][1]])) 		
		{
			ids[v0].atomicInsert(v1);
			ids[v1].atomicInsert(v0);			
		}
	}	

	template<typename Tetrahedron>
	__global__ void TetSet_UpdateIndexAndShapeIds(
		DArray<Tetrahedron> indices,
		uint indexSize,
		uint vertexOffset,
		uint indexOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= indexSize) return;

		Tetrahedron t = indices[indexOffset + tId];
		t[0] += vertexOffset;
		t[1] += vertexOffset;
		t[2] += vertexOffset;
		t[3] += vertexOffset;

		indices[indexOffset + tId] = t;
	}

	template<typename TDataType>
	std::shared_ptr<dyno::TetrahedronSet<TDataType>> 
		TetrahedronSet<TDataType>::merge(std::vector<std::shared_ptr<TetrahedronSet<TDataType>>>& tsArray)
	{
		auto ret = std::make_shared<TetrahedronSet<TDataType>>();

		for (auto ts : tsArray)
			assert(ts != nullptr);

		uint vNum = 0;
		uint tNum = 0;
		for (auto ts : tsArray)
		{
			vNum += ts->getPoints().size();
			tNum += ts->tetrahedronIndices().size();
		}

		auto& vertices = ret->getPoints();
		auto& indices = ret->tetrahedronIndices();

		vertices.resize(vNum);
		indices.resize(tNum);

		uint vOffset = 0;
		uint tOffset = 0;
		for (auto ts : tsArray)
		{
			auto& vSrc = ts->getPoints();
			auto& tSrc = ts->tetrahedronIndices();

			vertices.assign(vSrc, vSrc.size(), vOffset, 0);
			indices.assign(tSrc, tSrc.size(), tOffset, 0);

			uint num = tSrc.size();
			cuExecute(num, TetSet_UpdateIndexAndShapeIds,
				indices,
				num,
				vOffset,
				tOffset);

			vOffset += vSrc.size();
			tOffset += tSrc.size();
		}

		ret->update();

		return ret;
	}



	template<typename TDataType>
	void TetrahedronSet<TDataType>::requestPointNeighbors(DArrayList<int>& lists)
	{
		if (this->mCoords.isEmpty())
			return;
		
		int verNum = this->mCoords.size();
		auto edge = this->edgeIndices();
		int edgeNum = edge.size();

		DArray<uint> counts;
		counts.resize(verNum);
		counts.reset();
		
		DArrayList<int> list_1hop;

		// 1 - hop
		cuExecute(edgeNum,
			TetSet_NeigCountEdgeNumber,
			counts,
			edge);
		list_1hop.resize(counts);

		cuExecute(edgeNum,
			TetSet_NeigStoreEdgeIds,
			list_1hop,
			this->mEdges);

		// 2 - hop
		uint pDims = cudaGridSize(verNum, threadNum);

		TetSet_NeigCount2HopNumber << <pDims, threadNum >> >(
			counts,
			list_1hop);
		cuSynchronize();

		lists.resize(counts);
		cuSynchronize();
		
		Reduction<uint> reduce;
		uint maxSize = reduce.maximum(counts.begin(), counts.size());
		printf("max size: %d for 2-hop\n", maxSize);
		
		TetSet_NeigStore2HopIds << <pDims, threadNum >> >(
			lists,
			list_1hop,
			counts);
		cuSynchronize();
		

		// opposite point
		// auto tri2tet = this->triangle2Tetrahedron();
		// auto tet = this->tetrahedronIndices();
		// cuExecute(tri2tet.size(),
		// 	TetSet_NeigCountTriNumber,
		// 	counts,
		// 	tet,
		// 	tri2tet);

		// cuExecute(tri2tet.size(),
		// 	TetSet_NeigStoreTriIds,
		// 	lists,
		// 	tet,
		// 	tri2tet);

		list_1hop.clear();
		counts.clear();
	}

	__global__ void TetSet_CountBoundaryTriangles(
		DArray<int> indicator,
		DArray<Topology::Tri2Tet> tri2Tet)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tri2Tet.size()) return;

		if (tri2Tet[tId][0] == EMPTY || tri2Tet[tId][1] == EMPTY)
			indicator[tId] = 1;
		else
			indicator[tId] = 0;
	}

	__global__ void TetSet_SetupBoundaryTriangles(
		DArray<Topology::Triangle> boundaryIndices,
		DArray<Topology::Triangle> allIndices,
		DArray<int> radix)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= allIndices.size()) return;

		if (radix[tId] != radix[tId + 1])
			boundaryIndices[radix[tId]] = allIndices[tId];
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::requestBoundaryTriangleIndices(DArray<Topology::Triangle>& indices)
	{
		auto& triIndices = this->triangleIndices();

		DArray<int> counter(triIndices.size() + 1);
		counter.reset();

		cuExecute(triIndices.size(), TetSet_CountBoundaryTriangles,
			counter,
			mTri2Tet);

		int boundaryTriNum = thrust::reduce(thrust::device, counter.begin(), counter.begin() + counter.size());
		thrust::exclusive_scan(thrust::device, counter.begin(), counter.begin() + counter.size(), counter.begin());

		indices.resize(boundaryTriNum);
		cuExecute(triIndices.size(), TetSet_SetupBoundaryTriangles,
			indices,
			triIndices,
			counter);

		counter.clear();
	}

	__global__ void TetSet_CountBoundaryVertices(
		DArray<int> counter,
		DArray<Topology::Tri2Tet> tri2Tet)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tri2Tet.size()) return;

		if (tri2Tet[tId][0] == EMPTY || tri2Tet[tId][1] == EMPTY)
			counter[tId] = 3;
		else
			counter[tId] = 0;
	}

	__global__ void TetSet_SetupBoundaryVertexIndices(
		DArray<int> edgeIndices,
		DArray<Topology::Triangle> triangles,
		DArray<int> radix)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triangles.size()) return;

		if (radix[tId] != radix[tId + 1])
		{
			Topology::Triangle t = triangles[tId];
			edgeIndices[radix[tId]] = t[0];
			edgeIndices[radix[tId] + 1] = t[1];
			edgeIndices[radix[tId] + 2] = t[2];
		}
	}

	__global__ void TetSet_CountUniqueEdgeIndices(
		DArray<int> counter,
		DArray<int> ids)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= ids.size()) return;

		if (tId == 0 || ids[tId] != ids[tId - 1])
			counter[tId] = 1;
		else
			counter[tId] = 0;
	}

	__global__ void TetSet_SetupUniqueVertexIndices(
		DArray<int> vertexIds,
		DArray<int> ids,
		DArray<int> radix)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= ids.size()) return;

		if (tId == 0 || ids[tId] != ids[tId - 1])
			vertexIds[radix[tId]] = ids[tId];
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::requestBoundaryVertexIndices(DArray<int>& indices)
	{
		auto& triIndices = this->triangleIndices();

		DArray<int> counter(triIndices.size() + 1);
		counter.reset();
		//Check whether a triangle is located on the boundary
		cuExecute(triIndices.size(), TetSet_CountBoundaryVertices,
			counter,
			mTri2Tet);

		int boundaryVertNum = thrust::reduce(thrust::device, counter.begin(), counter.begin() + counter.size());
		thrust::exclusive_scan(thrust::device, counter.begin(), counter.begin() + counter.size(), counter.begin());

		DArray<int> vertexIndexBuffer(boundaryVertNum);
		cuExecute(triIndices.size(), TetSet_SetupBoundaryVertexIndices,
			vertexIndexBuffer,
			triIndices,
			counter);

		//Remove duplicated vertex ids
		thrust::sort_by_key(thrust::device, vertexIndexBuffer.begin(), vertexIndexBuffer.begin() + vertexIndexBuffer.size(), vertexIndexBuffer.begin());

		counter.resize(boundaryVertNum);

		cuExecute(vertexIndexBuffer.size(), TetSet_CountUniqueEdgeIndices,
			counter,
			vertexIndexBuffer);

		int uniqueVertexNum = thrust::reduce(thrust::device, counter.begin(), counter.begin() + counter.size());
		thrust::exclusive_scan(thrust::device, counter.begin(), counter.begin() + counter.size(), counter.begin());

		indices.resize(uniqueVertexNum);
		cuExecute(vertexIndexBuffer.size(), TetSet_SetupUniqueVertexIndices,
			indices,
			vertexIndexBuffer,
			counter);

		counter.clear();
		vertexIndexBuffer.clear();
	}

	__global__ void TetSet_CountBoundaryEdges(
		DArray<int> counter,
		DArray<Topology::Tri2Tet> tri2Tet)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tri2Tet.size()) return;

		if (tri2Tet[tId][0] == EMPTY || tri2Tet[tId][1] == EMPTY)
			counter[tId] = 3;
		else
			counter[tId] = 0;
	}

	__global__ void TetSet_SetupBoundaryEdgeIndices(
		DArray<int> edgeIndices,
		DArray<Topology::Tri2Edg> tri2edge,
		DArray<int> radix)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tri2edge.size()) return;

		if (radix[tId] != radix[tId + 1])
		{
			Topology::Tri2Edg t2e = tri2edge[tId];
			edgeIndices[radix[tId]] = t2e[0];
			edgeIndices[radix[tId] + 1] = t2e[1];
			edgeIndices[radix[tId] + 2] = t2e[2];
		}
	}

	__global__ void TetSet_SetupUniqueEdgeIndices(
		DArray<Topology::Edge> boundaryEdges,
		DArray<Topology::Edge> allEdges,
		DArray<int> edgeIndices,
		DArray<int> radix)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= edgeIndices.size()) return;

		if (tId == 0 || edgeIndices[tId] != edgeIndices[tId - 1])
			boundaryEdges[radix[tId]] = allEdges[tId];
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::requestBoundaryEdgeIndices(DArray<Topology::Edge>& indices)
	{
		auto& t2e = this->triangle2Edge();
		auto& triIndices = this->triangleIndices();
		auto& edgeIndices = this->edgeIndices();

		DArray<int> counter(triIndices.size() + 1);
		counter.reset();
		
		//Check whether a triangle is located on the boundary
		cuExecute(triIndices.size(), TetSet_CountBoundaryEdges,
			counter,
			mTri2Tet);

		int boundaryEdgeNum = thrust::reduce(thrust::device, counter.begin(), counter.begin() + counter.size());
		thrust::exclusive_scan(thrust::device, counter.begin(), counter.begin() + counter.size(), counter.begin());

		DArray<int> edgeIndexBuffer(boundaryEdgeNum);
		cuExecute(triIndices.size(), TetSet_SetupBoundaryEdgeIndices,
			edgeIndexBuffer,
			t2e,
			counter);

		//Remove duplicated edge ids
		thrust::sort_by_key(thrust::device, edgeIndexBuffer.begin(), edgeIndexBuffer.begin() + edgeIndexBuffer.size(), edgeIndexBuffer.begin());

		counter.resize(boundaryEdgeNum);

		cuExecute(edgeIndexBuffer.size(), TetSet_CountUniqueEdgeIndices,
			counter,
			edgeIndexBuffer);

		int uniqueEdgeNum = thrust::reduce(thrust::device, counter.begin(), counter.begin() + counter.size());
		thrust::exclusive_scan(thrust::device, counter.begin(), counter.begin() + counter.size(), counter.begin());

		indices.resize(uniqueEdgeNum);
		cuExecute(edgeIndexBuffer.size(), TetSet_SetupUniqueEdgeIndices,
			indices,
			edgeIndices,
			edgeIndexBuffer,
			counter);

		counter.clear();
		edgeIndexBuffer.clear();
	}

	template<typename Coord>
	__global__ void TetSet_ExtractBoundaryVertices(
		DArray<Coord> triVertices,	//output
		DArray<int> tet2triMapper,	//output
		DArray<Coord> tetVertices,
		DArray<int> vertexIdsOnBoundary)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vertexIdsOnBoundary.size()) return;

		triVertices[tId] = tetVertices[vertexIdsOnBoundary[tId]];
		tet2triMapper[vertexIdsOnBoundary[tId]] = tId;
	}

	__global__ void TetSet_UpdateBoundaryTriangleIndices(
		DArray<Topology::Triangle> indices,
		DArray<int> tet2triMapper)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= indices.size()) return;

		Topology::Triangle t = indices[tId];

		indices[tId] = Topology::Triangle(tet2triMapper[t[0]], tet2triMapper[t[1]], tet2triMapper[t[2]]);
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::extractSurfaceMesh(TriangleSet<TDataType>& ts, DArray<int>& indices)
	{
		auto& tetVertices = this->getPoints();
		auto& triVertices = ts.getPoints();
		auto& triIndices = ts.triangleIndices();

		DArray<int> tet2triVertexMapper;
		this->requestBoundaryVertexIndices(indices);

		this->requestBoundaryTriangleIndices(triIndices);

		triVertices.resize(indices.size());
		tet2triVertexMapper.resize(tetVertices.size());
		cuExecute(indices.size(), TetSet_ExtractBoundaryVertices,
			triVertices,
			tet2triVertexMapper,
			tetVertices,
			indices);

		cuExecute(triIndices.size(), TetSet_UpdateBoundaryTriangleIndices,
			triIndices,
			tet2triVertexMapper);

		ts.update();

		tet2triVertexMapper.clear();
	}

	template<typename Coord>
	__global__ void TetSet_UpdateSurfaceMeshVertices(
		DArray<Coord> vertsOnBoundary,
		DArray<Coord> vertsOnTet,
		DArray<int> idMapper)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= idMapper.size()) return;

		vertsOnBoundary[tId] = vertsOnTet[idMapper[tId]];
	}


	template<typename TDataType>
	void TetrahedronSet<TDataType>::updateSurfaceMesh(TriangleSet<TDataType>& ts, DArray<int>& indices)
	{
		auto& tet_verts = this->getPoints();
		auto& tri_verts = ts.getPoints();

		assert(tri_verts.size() == indices.size());

		cuExecute(indices.size(), TetSet_UpdateSurfaceMeshVertices,
			tri_verts,
			tet_verts,
			indices);
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::updateTopology()
	{
		this->updateTetrahedrons();

		this->updateVertex2Tetrahedron();
		
		this->TriangleSet<TDataType>::updateTopology();

	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::updateVertex2Tetrahedron()
	{
		int verNum = this->mCoords.size();
		int tetNum = mTethedrons.size();

		// Ver2Tet
		DArray<uint> counter;
		counter.resize(verNum);
		counter.reset();

		cuExecute(tetNum, TetSet_CountTets,
			counter,
			mTethedrons);

		mVer2Tet.resize(counter);

		cuExecute(tetNum, TetSet_SetupTetIds,
			mVer2Tet,
			mTethedrons);

		counter.clear();
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::updateTriangles()
	{
		// Update Triangle from Tetrahedron Mesh
		// Light Map {Tri2Tet, Tet2Tri}
		uint tetSize = mTethedrons.size();

		if (tetSize == 0) return;	

		DArray<TKey> keys;
		DArray<Info> tetIds;
		
		keys.resize(4 * tetSize);
		tetIds.resize(4 * tetSize);

		cuExecute(tetSize,
			TetSet_SetupKeys,
			keys,
			tetIds,
			mTethedrons);

		thrust::sort_by_key(thrust::device, keys.begin(), keys.begin() + keys.size(), tetIds.begin());

		DArray<int> counter;
		counter.resize(4 * tetSize);

		cuExecute(keys.size(),
			TetSet_CountTriangleNumber,
			counter,
			keys);

		int triNum = thrust::reduce(thrust::device, counter.begin(), counter.begin() + counter.size());
		thrust::exclusive_scan(thrust::device, counter.begin(), counter.begin() + counter.size(), counter.begin());

		mTri2Tet.resize(triNum);

		auto& tris = this->triangleIndices();
		tris.resize(triNum);

		DArray<int> countTet2Tri;

		countTet2Tri.resize(tetSize); 
		countTet2Tri.reset();

		mTet2Tri.resize(tetSize);

		// Tri <-> Tet
		cuExecute(keys.size(), 
			TetSet_SetupTriangles,
			tris, 
			mTri2Tet,
			mTet2Tri, 
			countTet2Tri, 
			keys, 
			counter, 
			tetIds);

		counter.clear();
		tetIds.clear();
		keys.clear();
		countTet2Tri.clear();
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::copyFrom(TetrahedronSet<TDataType>& tetSet)
	{
		mTethedrons.resize(tetSet.mTethedrons.size());
		mTethedrons.assign(tetSet.mTethedrons);

		mTri2Tet.resize(tetSet.mTri2Tet.size());
		mTri2Tet.assign(tetSet.mTri2Tet);

		mTet2Tri.resize(tetSet.mTet2Tri.size());
		mTet2Tri.assign(tetSet.mTet2Tri);

		mVer2Tet.assign(tetSet.mVer2Tet);

		TriangleSet<TDataType>::copyFrom(tetSet);
	}
	template<typename TDataType>
	std::shared_ptr<TetrahedronSet<TDataType>> TetrahedronSet<TDataType>::merge(TetrahedronSet<TDataType>& tetSet)
	{
		auto ret = std::make_shared<TetrahedronSet<TDataType>>();

		auto& vertices = ret->getPoints();
		auto& indices = ret->tetrahedronIndices();

		uint vNum0 = PointSet<TDataType>::mCoords.size();
		uint vNum1 = tetSet.getPoints().size();

		uint tNum0 = mTethedrons.size();
		uint tNum1 = tetSet.tetrahedronIndices().size();

		vertices.resize(vNum0 + vNum1);
		indices.resize(tNum0 + tNum1);

		vertices.assign(PointSet<TDataType>::mCoords, vNum0, 0, 0);
		vertices.assign(tetSet.getPoints(), vNum1, vNum0, 0);

		indices.assign(mTethedrons, tNum0, 0, 0);
		indices.assign(tetSet.tetrahedronIndices(), tNum1, tNum0, 0);

		cuExecute(tNum1,
			TetSet_UpdateIndex,
			indices,
			vNum0,
			tNum0);

		return ret;
	}


	template<typename Tetrahedron>
	__global__ void TetSet_UpdateIndex(
		DArray<Tetrahedron> indices,
		uint vertexOffset,
		uint indexOffset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= indices.size() - indexOffset) return;

		Tetrahedron t = indices[indexOffset + tId];
		t[0] += vertexOffset;
		t[1] += vertexOffset;
		t[2] += vertexOffset;
		t[3] += vertexOffset;

		indices[indexOffset + tId] = t;
	}

	template<typename TDataType>
	bool TetrahedronSet<TDataType>::isEmpty()
	{
		return mTethedrons.size() && TriangleSet<TDataType>::isEmpty();
	}

	DEFINE_CLASS(TetrahedronSet);
}