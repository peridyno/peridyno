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
			tIndex = TopologyModule::Triangle(EMPTY, EMPTY, EMPTY);
		}

		DYN_FUNC Info(int id, TopologyModule::Triangle index) {
			tetId = id;
			tIndex = index;
		};

		int tetId;
		TopologyModule::Triangle tIndex;
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

		ids[4 * tId] = Info(tId, TopologyModule::Triangle(tet[0], tet[1], tet[2]));
		ids[4 * tId + 1] = Info(tId, TopologyModule::Triangle(tet[1], tet[2], tet[3]));
		ids[4 * tId + 2] = Info(tId, TopologyModule::Triangle(tet[2], tet[3], tet[0]));
		ids[4 * tId + 3] = Info(tId, TopologyModule::Triangle(tet[3], tet[0], tet[1]));
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


	template<typename Tri2Tet>
	__global__ void TetSet_CountSurfaceTri(
		DArray<int> surfaceTri,
		DArray<int> sfCounter,
		DArray<Tri2Tet> tri2Tet)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tri2Tet.size()) return;

		if (tri2Tet[tId][0] != EMPTY && tri2Tet[tId][1] != EMPTY)
		{
			surfaceTri[tId] = 0x7fffffff; // Invalid index
			sfCounter[tId] = 0;
		}
		else
		{
			surfaceTri[tId] = tId;
			sfCounter[tId] = 1;
		}
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

	template<typename Tri2Tri, typename EKey>
	__global__ void TetSet_SetupSufraceTri2Tri(
		DArray<Tri2Tri> surface_tri2Tri,
		DArray<int> surface_top,
		DArray<EKey> keys,
		DArray<int> triIds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= keys.size()) return;

		if (tId == 0 || keys[tId] != keys[tId - 1])
		{
			EKey key = keys[tId];
			int surId = triIds[tId];
			if (tId + 1 < keys.size() && keys[tId + 1] == key)
			{
				int nextSurId = triIds[tId + 1];
				int sur_top = atomicAdd(&surface_top[surId], 1);
				int next_top = atomicAdd(&surface_top[nextSurId], 1);
				surface_tri2Tri[surId][sur_top] = nextSurId;
				surface_tri2Tri[nextSurId][next_top] = surId;
			}
		}
	}

	__global__ void TetSet_SetupSurfaceTri(
		DArray<int> true_surface_id,
		DArray<int> surface_id)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= true_surface_id.size()) return;

		true_surface_id[pId] = surface_id[pId];
	}

	template<typename Coord, typename Triangle, typename Tetrahedron, typename Tri2Tet>
	__global__ void TetSet_SetupSurfaceNormal(
		DArray<int> triangle_normal,
		DArray<int> surface_id,
		DArray<Coord> vertex_pos,
		DArray<Triangle> triangles,
		DArray<Tetrahedron> tetrahedrons,
		DArray<Tri2Tet> tri2Tet)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= surface_id.size()) return;

		int tId = surface_id[pId];
		Triangle tri = triangles[tId];
		Tetrahedron tet = tetrahedrons[tri2Tet[tId][0]];
		int inside_id = -1;
		for (int i = 0; i < 4; i++)
		if(tet[i] != tri[0] && tet[i] != tri[1] && tet[i] != tri[2])
		{
			inside_id = tet[i];
			break;
		}

		if(inside_id == -1)
		{
			printf("[Error] surface triangle %d is not belong to one tetrahedron\n", tId);
		}
		
		Coord a[3] = { vertex_pos[tri[0]], vertex_pos[tri[1]], vertex_pos[tri[2]] };
		Coord b = vertex_pos[inside_id];
		int noraml_tri = true;
		Coord normal = (a[1] - a[0]).cross(a[2] - a[0]);
		normal.normalize();
		if(normal.dot(b - a[0]) > 0.f)
		{
			normal = -normal;
			noraml_tri = false;
		}
		triangle_normal[tId] = noraml_tri;// false means reverse
	}

	
	template<typename Coord>
	__global__ void TetSet_Debug(
		DArray<Coord> vertex_pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vertex_pos.size()) return;

		Coord b = vertex_pos[pId];

		if (b.y < 0.5)
		{
			printf("Error point %d is %f %f %f\n", pId, b[0], b[1], b[2]);
		}
	}

	template<typename EKey, typename Triangle, typename Tri2Tri>
	__global__ void TetSet_SetupEKeys(
		DArray<EKey> keys,
		DArray<int> ids,
		DArray<Tri2Tri> surface_tri2Tri,
		DArray<int> surface_id,
		DArray<Triangle> triangles)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= surface_id.size()) return;

		surface_tri2Tri[pId][0] = EMPTY;
		surface_tri2Tri[pId][1] = EMPTY;
		surface_tri2Tri[pId][2] = EMPTY;

		int tId = surface_id[pId];
		Triangle tri = triangles[tId];
		keys[3 * pId    ] = EKey(tri[0], tri[1]);
		keys[3 * pId + 1] = EKey(tri[1], tri[2]);
		keys[3 * pId + 2] = EKey(tri[2], tri[0]);

		ids[3 * pId    ] = pId;
		ids[3 * pId + 1] = pId;
		ids[3 * pId + 2] = pId;
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
		TopologyModule::Tri2Tet& t2t,
		TopologyModule::Tetrahedron& tetA,
		TopologyModule::Tetrahedron& tetB)
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
		DArray<TopologyModule::Edge> edges)
	{
		int eId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (eId >= edges.size()) return;

		TopologyModule::Edge edge = edges[eId];

		atomicAdd(&(num[edge[0]]), 1);
		atomicAdd(&(num[edge[1]]), 1);
	}

	__global__ void TetSet_NeigCountTriNumber(
		DArray<uint> num,
		DArray<TopologyModule::Tetrahedron> tets,
		DArray<TopologyModule::Tri2Tet> tri2Tet)
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
		DArray<TopologyModule::Edge> edges)
	{
		int eId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (eId >= edges.size()) return;

		TopologyModule::Edge edge = edges[eId];
		int v0 = edge[0];
		int v1 = edge[1];

		ids[v0].atomicInsert(v1);
		ids[v1].atomicInsert(v0);
	}

	__global__ void TetSet_NeigStoreTriIds(
		DArrayList<int> ids,
		DArray<TopologyModule::Tetrahedron> tets,
		DArray<TopologyModule::Tri2Tet> tri2Tet)
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
			mEdges);

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

	template<typename TDataType>
	void TetrahedronSet<TDataType>::requestSurfaceMeshIds(DArray<int>& surfaceIds, DArray<int>& towardOutside, DArray<::dyno::TopologyModule::Tri2Tri>& t2t)
	{
		// Update {SurfaceTir, TriNormalSig, SurfaceTri2Tri}
		auto triIndices = this->triangleIndices();
		int triNum = triIndices.size();

		DArray<int> pSurfaceTri;
		DArray<int> sfCounter;
		DArray<int> counter;
		pSurfaceTri.resize(triNum);
		sfCounter.resize(triNum);
		sfCounter.reset();

		// count surface Tri
		cuExecute(triNum, TetSet_CountSurfaceTri,
			pSurfaceTri, sfCounter,
			mTri2Tet);

		int sfTriNum = thrust::reduce(thrust::device, sfCounter.begin(), sfCounter.begin() + sfCounter.size());
		// Sort [surface id... | 0x7fffffff... ]
		thrust::sort(thrust::device, pSurfaceTri.begin(), pSurfaceTri.begin() + triNum);

		surfaceIds.resize(sfTriNum);
		cuExecute(sfTriNum, TetSet_SetupSurfaceTri,
			surfaceIds,
			pSurfaceTri);
		// mSurfaceTri.assign(pSurfaceTri, sfTriNum);

		t2t.resize(sfTriNum);
		towardOutside.resize(triNum);
		printf("surface triangle number: %d\n", sfTriNum);

		// Surface Tri Normal (Outside)
		cuExecute(sfTriNum, TetSet_SetupSurfaceNormal,
			towardOutside, surfaceIds,
			this->mCoords, triIndices, mTethedrons, mTri2Tet
		);

		// Surface Tri 2 Tri
		// 1. Sort Surface Edge 

		DArray<EKey> ekeys;
		DArray<int> triIds;

		ekeys.resize(3 * sfTriNum);
		triIds.resize(3 * sfTriNum);

		cuExecute(sfTriNum, TetSet_SetupEKeys,
			ekeys, triIds, t2t,
			surfaceIds, triIndices);

		thrust::sort_by_key(thrust::device, ekeys.begin(), ekeys.begin() + ekeys.size(), triIds.begin());

		// 2. Surface Tri 2 Tri
		counter.resize(sfTriNum);
		counter.reset();
		cuExecute(ekeys.size(), TetSet_SetupSufraceTri2Tri,
			t2t, counter,
			ekeys, triIds);

		counter.clear();
		pSurfaceTri.clear();
		sfCounter.clear();
		ekeys.clear();
		triIds.clear();
	}

	template<typename TDataType>
	void TetrahedronSet<TDataType>::extractSurfaceMesh(TriangleSet<TDataType>& ts)
	{

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
	bool TetrahedronSet<TDataType>::isEmpty()
	{
		return mTethedrons.size() && TriangleSet<TDataType>::isEmpty();
	}

	DEFINE_CLASS(TetrahedronSet);
}