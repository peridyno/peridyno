#include "HexahedronSet.h"
#include "QuadSet.h"
#include <fstream>
#include <iostream>
#include <sstream>

#include <thrust/sort.h>

namespace dyno
{
	template<typename TDataType>
	HexahedronSet<TDataType>::HexahedronSet()
		: QuadSet<TDataType>()
	{

	}

	template<typename TDataType>
	HexahedronSet<TDataType>::~HexahedronSet()
	{
	}

	template<typename TDataType>
	void HexahedronSet<TDataType>::setHexahedrons(std::vector<Hexahedron>& hexahedrons)
	{
		std::vector<Quad> quads;

		m_hexahedrons.resize(hexahedrons.size());
		m_hexahedrons.assign(hexahedrons);

		this->updateQuads();
	}

	template<typename TDataType>
	void HexahedronSet<TDataType>::setHexahedrons(DArray<Hexahedron>& hexahedrons)
	{
		if (hexahedrons.size() != m_hexahedrons.size())
		{
			m_hexahedrons.resize(hexahedrons.size());
		}

		m_hexahedrons.assign(hexahedrons);

		this->updateQuads();
	}

	template<typename Hexahedron>
	__global__ void HS_CountHexs(
		DArray<uint> counter,
		DArray<Hexahedron> hexs)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= hexs.size()) return;

		Hexahedron t = hexs[tId];

		atomicAdd(&counter[t[0]], 1);
		atomicAdd(&counter[t[1]], 1);
		atomicAdd(&counter[t[2]], 1);
		atomicAdd(&counter[t[3]], 1);
		atomicAdd(&counter[t[4]], 1);
		atomicAdd(&counter[t[5]], 1);
		atomicAdd(&counter[t[6]], 1);
		atomicAdd(&counter[t[7]], 1);
	}

	template<typename Hexahedron>
	__global__ void HS_SetupHexIds(
		DArrayList<int> hexIds,
		DArray<Hexahedron> hexs)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= hexs.size()) return;

		Hexahedron t = hexs[tId];

		hexIds[t[0]].atomicInsert(tId);
		hexIds[t[1]].atomicInsert(tId);
		hexIds[t[2]].atomicInsert(tId);
		hexIds[t[3]].atomicInsert(tId);
		hexIds[t[4]].atomicInsert(tId);
		hexIds[t[5]].atomicInsert(tId);
		hexIds[t[6]].atomicInsert(tId);
		hexIds[t[7]].atomicInsert(tId);
	}

	template<typename TDataType>
	DArrayList<int>& HexahedronSet<TDataType>::getVer2Hex()
	{
		DArray<uint> counter;
		counter.resize(this->mCoords.size());
		counter.reset();

		cuExecute(m_hexahedrons.size(),
			HS_CountHexs,
			counter,
			m_hexahedrons);

		m_ver2Hex.resize(counter);

		counter.reset();
		cuExecute(m_hexahedrons.size(),
			HS_SetupHexIds,
			m_ver2Hex,
			m_hexahedrons);

		counter.clear();

		return m_ver2Hex;
	}

	template<typename TDataType>
	void HexahedronSet<TDataType>::getVolume(DArray<Real>& volume)
	{

	}

	template<typename QKey, typename Hexahedron>
	__global__ void HS_SetupKeys(
		DArray<QKey> keys,
		DArray<int> ids,
		DArray<Hexahedron> hexs)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= hexs.size()) return;

		Hexahedron hex = hexs[tId];
		keys[6 * tId] = QKey(hex[0], hex[1], hex[2], hex[3]);
		keys[6 * tId + 1] = QKey(hex[4], hex[5], hex[6], hex[7]);
		keys[6 * tId + 2] = QKey(hex[0], hex[1], hex[5], hex[4]);
		keys[6 * tId + 3] = QKey(hex[1], hex[2], hex[6], hex[5]);
		keys[6 * tId + 4] = QKey(hex[3], hex[2], hex[6], hex[7]);
		keys[6 * tId + 5] = QKey(hex[0], hex[3], hex[7], hex[4]);

		ids[6 * tId] = tId;
		ids[6 * tId + 1] = tId;
		ids[6 * tId + 2] = tId;
		ids[6 * tId + 3] = tId;
		ids[6 * tId + 4] = tId;
		ids[6 * tId + 5] = tId;
	}

	template<typename QKey>
	__global__ void HS_CountQuadNumber(
		DArray<int> counter,
		DArray<QKey> keys)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= keys.size()) return;

		if (tId == 0 || keys[tId] != keys[tId - 1])
			counter[tId] = 1;
		else
			counter[tId] = 0;
	}

	template<typename Quad, typename Quad2Hex, typename QKey>
	__global__ void HS_SetupQuads(
		DArray<Quad> quads,
		DArray<Quad2Hex> quad2Hex,
		DArray<QKey> keys,
		DArray<int> counter,
		DArray<int> hexIds)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= keys.size()) return;

		int shift = counter[tId];
		if (tId == 0 || keys[tId] != keys[tId - 1])
		{
			QKey key = keys[tId];
			quads[shift] = Quad(key[0], key[1], key[2], key[3]);

			Quad2Hex q2H(EMPTY, EMPTY);
			q2H[0] = hexIds[tId];

			if (tId + 1 < keys.size() && keys[tId + 1] == key)
				q2H[1] = hexIds[tId + 1];

			quad2Hex[shift] = q2H;

		}
	}

	template<typename QKey>
	void printTKey(DArray<QKey> keys, int maxLength) {
		CArray<QKey> h_keys;
		h_keys.resize(keys.size());
		h_keys.assign(keys);

		int psize = min((int)h_keys.size(), maxLength);
		for (int i = 0; i < psize; i++)
		{
			printf("%d: %d %d %d %d \n", i, h_keys[i][0], h_keys[i][1], h_keys[i][2], h_keys[i][3]);
		}

		h_keys.clear();
	}

	/*void printCount(DArray<int> keys, int maxLength) {
		CArray<int> h_keys;
		h_keys.resize(keys.size());
		h_keys.assign(keys);

		int psize = minimum((int)h_keys.size(), maxLength);
		for (int i = 0; i < psize; i++)
		{
			printf("%d: %d \n", i, h_keys[i]);
		}

		h_keys.clear();
	}*/

	template<typename TDataType>
	void HexahedronSet<TDataType>::updateQuads()
	{
		uint hexSize = m_hexahedrons.size();

		DArray<QKey> keys;
		DArray<int> hexIds;

		keys.resize(6 * hexSize);
		hexIds.resize(6 * hexSize);

		cuExecute(hexSize,
			HS_SetupKeys,
			keys,
			hexIds,
			m_hexahedrons);

		thrust::sort_by_key(thrust::device, keys.begin(), keys.begin() + keys.size(), hexIds.begin());

		DArray<int> counter;
		counter.resize(6 * hexSize);

		cuExecute(keys.size(),
			HS_CountQuadNumber,
			counter,
			keys);

		int quadNum = thrust::reduce(thrust::device, counter.begin(), counter.begin() + counter.size());
		thrust::exclusive_scan(thrust::device, counter.begin(), counter.begin() + counter.size(), counter.begin());

		quad2Hex.resize(quadNum);

		auto& pQuad = this->getQuads();
		pQuad.resize(quadNum);
		cuExecute(keys.size(),
			HS_SetupQuads,
			pQuad,
			quad2Hex,
			keys,
			counter,
			hexIds);

		counter.clear();
		hexIds.clear();
		keys.clear();

		
// 		this->updateTriangles();
// 		this->updateEdges();
	}


	template<typename TDataType>
	void HexahedronSet<TDataType>::copyFrom(HexahedronSet<TDataType> hexSet)
	{
		m_hexahedrons.resize(hexSet.m_hexahedrons.size());
		m_hexahedrons.assign(hexSet.m_hexahedrons);

		quad2Hex.resize(hexSet.quad2Hex.size());
		quad2Hex.assign(hexSet.quad2Hex);

		m_ver2Hex.assign(hexSet.m_ver2Hex);

		QuadSet<TDataType>::copyFrom(hexSet);
	}

	DEFINE_CLASS(HexahedronSet);
}