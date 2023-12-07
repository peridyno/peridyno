#include "NeighborPointQuery.h"
#include "Topology/GridHash.h"

#include "SceneGraph.h"

namespace dyno
{
	__constant__ int offset_nq[27][3] = { 
		0, 0, 0,
		0, 0, 1,
		0, 1, 0,
		1, 0, 0,
		0, 0, -1,
		0, -1, 0,
		-1, 0, 0,
		0, 1, 1,
		0, 1, -1,
		0, -1, 1,
		0, -1, -1,
		1, 0, 1,
		1, 0, -1,
		-1, 0, 1,
		-1, 0, -1,
		1, 1, 0,
		1, -1, 0,
		-1, 1, 0,
		-1, -1, 0,
		1, 1, 1,
		1, 1, -1,
		1, -1, 1,
		-1, 1, 1,
		1, -1, -1,
		-1, 1, -1,
		-1, -1, 1,
		-1, -1, -1
	};

	IMPLEMENT_TCLASS(NeighborPointQuery, TDataType)

	template<typename TDataType>
	NeighborPointQuery<TDataType>::NeighborPointQuery()
		: ComputeModule()
	{
		this->inOther()->tagOptional(true);

		this->varSizeLimit()->setRange(0, 100);
	}

	template<typename TDataType>
	NeighborPointQuery<TDataType>::~NeighborPointQuery()
	{
	}

	template<typename TDataType>
	void NeighborPointQuery<TDataType>::compute()
	{
		if (this->varSizeLimit()->getData() <= 0) {
			requestDynamicNeighborIds();
		}
		else {
			requestFixedSizeNeighborIds();
		}
	}

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_CalNeighborSize(
		DArray<uint> count,
		DArray<Coord> position_new,
		DArray<Coord> position, 
		GridHash<TDataType> hash, 
		Real h)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position_new.size()) return;

		Coord pos_ijk = position_new[pId];
		int3 gId3 = hash.getIndex3(pos_ijk);

		int counter = 0;
		for (int c = 0; c < 27; c++)
		{
			int cId = hash.getIndex(gId3.x + offset_nq[c][0], gId3.y + offset_nq[c][1], gId3.z + offset_nq[c][2]);
			if (cId >= 0) {
				int totalNum = hash.getCounter(cId);
				for (int i = 0; i < totalNum; i++) {
					int nbId = hash.getParticleId(cId, i);
					Real d_ij = (pos_ijk - position[nbId]).norm();
					if (d_ij < h)
					{
						counter++;
					}
				}
			}
		}

		count[pId] = counter;
	}
	

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_GetNeighborElements(
		DArrayList<int> nbrIds,
		DArray<Coord> position_new,
		DArray<Coord> position, 
		GridHash<TDataType> hash, 
		Real h)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position_new.size()) return;

		Coord pos_ijk = position_new[pId];
		int3 gId3 = hash.getIndex3(pos_ijk);

		List<int>& list_i = nbrIds[pId];

		int j = 0;
		for (int c = 0; c < 27; c++)
		{
			int cId = hash.getIndex(gId3.x + offset_nq[c][0], gId3.y + offset_nq[c][1], gId3.z + offset_nq[c][2]);
			if (cId >= 0) {
				int totalNum = hash.getCounter(cId);
				for (int i = 0; i < totalNum; i++) {
					int nbId = hash.getParticleId(cId, i);
					Real d_ij = (pos_ijk - position[nbId]).norm();
					if (d_ij < h)
					{
						list_i.insert(nbId);
						j++;
					}
				}
			}
		}
	}

	template<typename TDataType>
	void NeighborPointQuery<TDataType>::requestDynamicNeighborIds()
	{
		// Prepare inputs
		auto& points	= this->inPosition()->getData();
		auto& other		= this->inOther()->isEmpty() ? this->inPosition()->getData() : this->inOther()->getData();
		auto h			= this->inRadius()->getData();

		// Prepare outputs
		if (this->outNeighborIds()->isEmpty())
			this->outNeighborIds()->allocate();

		auto& nbrIds = this->outNeighborIds()->getData();

		// Construct hash grid
		Reduction<Coord> reduce;
		Coord hiBound = reduce.maximum(points.begin(), points.size());
		Coord loBound = reduce.minimum(points.begin(), points.size());

		// To avoid particles running out of the simulation domain
		auto scn = this->getSceneGraph();
		if (scn != NULL)
		{
			auto loLimit = scn->getLowerBound();
			auto hiLimit = scn->getUpperBound();

			hiBound = hiBound.minimum(hiLimit);
			loBound = loBound.maximum(loLimit);
		}

		GridHash<TDataType> hashGrid;
		hashGrid.setSpace(h, loBound - Coord(h), hiBound + Coord(h));
		hashGrid.clear();
		hashGrid.construct(points);

		DArray<uint> counter(other.size());
		cuExecute(other.size(),
			K_CalNeighborSize,
			counter,
			other,
			points, 
			hashGrid, 
			h);

		nbrIds.resize(counter);

		cuExecute(other.size(),
			K_GetNeighborElements,
			nbrIds, 
			other,
			points, 
			hashGrid,
			h);

		counter.clear();
		hashGrid.release();
	}
	

	template <typename T> __device__ void inline swap_on_device(T& a, T& b) {
		T c(a); a = b; b = c;
	}

	template <typename T>
	__device__ void heapify_up(int* keys, T* vals, int child)
	{
		int parent = (child - 1) / 2;
		while (child > 0)
		{
			if (vals[child] > vals[parent])
			{
				swap_on_device(vals[child], vals[parent]);
				swap_on_device(keys[child], keys[parent]);

				child = parent;
				parent = (child - 1) / 2;
			}
			else
			{
				break;
			}
		}
	}

	template <typename T>
	__device__ void heapify_down(int* keys, T* vals, int node, int size) {
		int j = node;
		while (true) {
			int left = 2 * j + 1;
			int right = 2 * j + 2;
			int largest = j;
			if (left<size && vals[left]>vals[largest]) {
				largest = left;
			}
			if (right<size && vals[right]>vals[largest]) {
				largest = right;
			}
			if (largest == j) return;
			swap_on_device(vals[j], vals[largest]);
			swap_on_device(keys[j], keys[largest]);
			j = largest;
		}
	}

	template <typename T>
	__device__ void heap_sort(int* keys, T* vals, int size) {
		while (size) {
			swap_on_device(vals[0], vals[size - 1]);
			swap_on_device(keys[0], keys[size - 1]);
			heapify_down(keys, vals, 0, --size);
		}
	}

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_ComputeNeighborFixed(
		DArrayList<int> nbrIds, 
		DArray<Coord> position_new,
		DArray<Coord> position, 
		GridHash<TDataType> hash, 
		Real h,
		int sizeLimit,
		DArray<int> heapIDs,
		DArray<Real> heapDistance)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position_new.size()) return;

		//TODO: used shared memory for speedup
		int* ids(heapIDs.begin() + pId * sizeLimit);// = new int[nbrLimit];
		Real* distance(heapDistance.begin() + pId * sizeLimit);// = new Real[nbrLimit];

		for (int i = 0; i < sizeLimit; i++) {
			ids[i] = INT_MAX;
			distance[i] = REAL_MAX;
		}

		Coord pos_ijk = position_new[pId];
		int3 gId3 = hash.getIndex3(pos_ijk);

		int counter = 0;
		for (int c = 0; c < 27; c++)
		{
			int cId = hash.getIndex(gId3.x + offset_nq[c][0], gId3.y + offset_nq[c][1], gId3.z + offset_nq[c][2]);
			if (cId >= 0) {
				int totalNum = hash.getCounter(cId);// min(hash.getCounter(cId), hash.npMax);
				for (int i = 0; i < totalNum; i++) {
					int nbId = hash.getParticleId(cId, i);
					float d_ij = (pos_ijk - position[nbId]).norm();
					if (d_ij < h)
					{
						if (counter < sizeLimit)
						{
							ids[counter] = nbId;
							distance[counter] = d_ij;

							heapify_up(ids, distance, counter);
							counter++;
						}
						else
						{
							if (d_ij < distance[0])
							{
								ids[0] = nbId;
								distance[0] = d_ij;

								heapify_down(ids, distance, 0, counter);
							}
						}
						
					}
				}
			}
		}

		List<int>& list_i = nbrIds[pId];

		heap_sort(ids, distance, counter);
		for (int bId = 0; bId < counter; bId++)
		{
			list_i.insert(ids[bId]);
		}
	}

	template<typename TDataType>
	void NeighborPointQuery<TDataType>::requestFixedSizeNeighborIds()
	{
		// Prepare inputs
		auto& points	= this->inPosition()->getData();
		auto& other		= this->inOther()->isEmpty() ? this->inPosition()->getData() : this->inOther()->getData();
		auto h			= this->inRadius()->getData();

		// Prepare outputs
		if (this->outNeighborIds()->isEmpty())
			this->outNeighborIds()->allocate();

		auto& nbrIds = this->outNeighborIds()->getData();

		uint numPt  = this->inPosition()->getDataPtr()->size();
		uint sizeLimit = this->varSizeLimit()->getData();
		
		nbrIds.resize(numPt, sizeLimit);

		// Construct hash grid
		Reduction<Coord> reduce;
		Coord hiBound = reduce.maximum(points.begin(), points.size());
		Coord loBound = reduce.minimum(points.begin(), points.size());

		// To avoid particles running out of the simulation domain
		auto scn = this->getSceneGraph();
		if (scn != NULL)
		{
			auto loLimit = scn->getLowerBound();
			auto hiLimit = scn->getUpperBound();

			hiBound = hiBound.minimum(hiLimit);
			loBound = loBound.maximum(loLimit);
		}


		GridHash<TDataType> hashGrid;
		hashGrid.setSpace(h, loBound - Coord(h), hiBound + Coord(h));
		hashGrid.clear();
		hashGrid.construct(points);

		DArray<int> ids(numPt * sizeLimit);
		DArray<Real> distance(numPt * sizeLimit);
		cuExecute(numPt,
			K_ComputeNeighborFixed,
			nbrIds,
			other,
			points,
			hashGrid,
			h,
			sizeLimit,
			ids,
			distance);

		ids.clear();
		distance.clear();
		//hashGrid.clear();
		hashGrid.release();
	}

	DEFINE_CLASS(NeighborPointQuery);
}