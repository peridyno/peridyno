#include "NeighborPointQueryAdaptive.h"
#include <thrust/sort.h>
#include <ctime>

namespace dyno
{
	IMPLEMENT_TCLASS(NeighborPointQueryAdaptive, TDataType)

	template<typename TDataType>
	NeighborPointQueryAdaptive<TDataType>::NeighborPointQueryAdaptive()
		: ComputeModule()
	{
		mAGrid = std::make_shared<AdaptiveGridSet<TDataType>>();

		mAGridGen = std::make_shared<MSTsGenerator<TDataType>>();

		this->varSizeLimit()->setRange(0, 100);
	}

	template<typename TDataType>
	NeighborPointQueryAdaptive<TDataType>::~NeighborPointQueryAdaptive()
	{
	}


	template<typename TDataType>
	void NeighborPointQueryAdaptive<TDataType>::initParameter()
	{
		auto& points = this->inPosition()->getData();
		auto m_dx = this->inRadius()->getData();

		Reduction<Coord> reduce;
		Coord m_max = reduce.maximum(points.begin(), points.size());
		Coord m_min = reduce.minimum(points.begin(), points.size());
		Coord center = (m_max + m_min) / 2;

		int rs = std::floor(std::max(std::max(m_max[0] - m_min[0], m_max[1] - m_min[1]), m_max[2] - m_min[2]) / m_dx);
		rs += 10;

		Level m_level = log2(float(rs));
		m_level += 1;

		rs = (1 << m_level);
		Coord unit(1, 1, 1);
		m_min = center - (m_dx*rs / 2)*unit;
		m_max = center + (m_dx*rs / 2)*unit;
		std::printf("NeighborPointQueryAdaptive: %f  %f  %f,  %f,  %d \n", m_min[0], m_min[1], m_min[2], m_dx, m_level);
	
		//m_origin = m_min;

		mAGrid->setOrigin(m_min);
		mAGrid->setDx(m_dx);
		mAGrid->setLevelMax(m_level);
		//if (m_level > 5) m_level = m_level - 4;
		//mAGrid->setLevelNum(m_level);
	}

	template <typename Real, typename Coord>
	__global__ void NPQA_ComputeBuf(
		DArray<OcKey> buf,
		DArray<Coord> pos,
		Coord origin,
		Real dx)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= pos.size()) return;

		Coord fp = (pos[tId] - origin) / dx;

		OcIndex nx = std::floor(fp[0]);
		OcIndex ny = std::floor(fp[1]);
		OcIndex nz = std::floor(fp[2]);
		buf[tId] = CalculateMortonCode(nx, ny, nz);
	}

	__global__ void NPQA_CountNode(
		DArray<uint> count,
		DArray<OcKey> buf,
		int num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= count.size()) return;

		if (tId == 0 || (buf[tId] != buf[tId - 1]))
		{
			int n = 1, i = 1;
			while ((tId + i) < count.size())
			{
				if (buf[tId + i] == buf[tId])
					n++;
				else
					break;

				if (n >= num)
					break;

				i++;
			}

			if (n >= num)
				count[tId] = 1;
		}
	}

	__global__ void NPQA_ComputeNode(
		DArray<OcKey> node,
		DArray<OcKey> buf,
		DArray<uint> count)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= count.size()) return;

		if (tId == (count.size() - 1) && count[tId] < node.size())
		{
			node[count[tId]] = buf[tId];
			return;
		}
		if (count[tId] != count[tId + 1])
		{
			node[count[tId]] = buf[tId];
			return;
		}
	}

	template<typename TDataType>
	void NeighborPointQueryAdaptive<TDataType>::initialNode()
	{
		initParameter();
		auto& points = this->inPosition()->getData();
		Real dx = this->inRadius()->getData();
		Coord m_origin = mAGrid->adaptiveGridOrigin();

		DArray<OcKey> node_buf(points.size());
		cuExecute(points.size(),
			NPQA_ComputeBuf,
			node_buf,
			points,
			m_origin,
			dx);
		thrust::sort(thrust::device, node_buf.begin(), node_buf.begin() + node_buf.size());

		DArray<uint> data_count(points.size());
		data_count.reset();
		cuExecute(data_count.size(),
			NPQA_CountNode,
			data_count,
			node_buf,
			this->varSizeMin()->getData());
		Reduction<uint> reduce;
		int node_num = reduce.accumulate(data_count.begin(), data_count.size());
		Scan<uint> scan;
		scan.exclusive(data_count.begin(), data_count.size());

		m_morton.resize(node_num);
		m_morton.reset();
		cuExecute(data_count.size(),
			NPQA_ComputeNode,
			m_morton,
			node_buf,
			data_count);

		node_buf.clear();
		data_count.clear();
	}

	__global__ void ESS_CalculateParticleNumber(
		DArray<int> index,
		DArray<int> node_index)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= node_index.size()) return;

		if (node_index[tId] == EMPTY) return;

		atomicAdd(&(index[node_index[tId]]), 1);
	}

	__global__ void ESS_CalculateParticleIdx(
		DArray<int> ids,
		DArray<int> counter,
		DArray<int> index,
		DArray<int> node_index)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= node_index.size()) return;

		if (node_index[tId] == EMPTY) return;

		int ind = atomicAdd(&(counter[node_index[tId]]), 1);

		ids[index[node_index[tId]] + ind] = tId;
	}

	template<typename TDataType>
	void NeighborPointQueryAdaptive<TDataType>::construct()
	{
		//printf("Neighbor Point Query Adaptive \n");

		initialNode();

		auto& particles = this->inPosition()->getData();
		Real m_dx = this->inRadius()->getData();

		mAGridGen->inpMorton()->assign(m_morton);
		mAGridGen->inAGridSet()->setDataPtr(mAGrid);
		mAGridGen->varOctreeType()->setCurrentKey(AdaptiveGridGenerator<DataType3f>::FACE_BALANCED);
		mAGridGen->varNeighMode()->setCurrentKey(AdaptiveGridGenerator<DataType3f>::TEWNTY_SEVEN_NEIGHBOR);
		Level lnum = mAGrid->adaptiveGridLevelMax();
		mAGridGen->varLevelNum()->setValue(lnum);
		mAGridGen->compute();

		//auto volumeSet = mVolumeOctree->stateSDFTopology()->getDataPtr();
		//std::clock_t Time1 = clock();

		mAGrid->extractLeafs27(m_node, m_neighbor);
		mAGrid->accessRandom(m_pIndex, particles);
		//std::clock_t Time2 = clock();
		//printf("Neighbor Point Query Adaptive: access time  %d  %d \n", int(Time2 - Time1), particles.size());

		m_index.resize(m_node.size());
		m_index.reset();
		cuExecute(particles.size(),
			ESS_CalculateParticleNumber,
			m_index,
			m_pIndex);
		Scan<int> scan;
		scan.exclusive(m_index.begin(), m_index.size());

		m_counter.resize(m_node.size());
		m_counter.reset();
		m_ids.resize(particles.size());
		m_ids.reset();
		cuExecute(particles.size(),
			ESS_CalculateParticleIdx,
			m_ids,
			m_counter,
			m_index,
			m_pIndex);

		if (this->outAdaptiveGrids()->isEmpty())
		{
			this->outAdaptiveGrids()->allocate();
		}
		this->outAdaptiveGrids()->setDataPtr(mAGrid);
	}

	template<typename TDataType>
	void NeighborPointQueryAdaptive<TDataType>::compute()
	{
		//std::clock_t Time1 = clock();

		construct();

		//if (this->varSizeLimit()->getData() <= 0)
		//{
			requestDynamicNeighborIds();
		//}
		//std::clock_t Time2 = clock();
		//printf("Neighbor Point Query Adaptive time  %d  %d \n", int(Time2 - Time1), this->inPosition()->getDataPtr()->size());
	}

	template<typename Real, typename Coord>
	__global__ void NPQA_CountNeighborSize(
		DArray<uint> count,
		DArray<Coord> position,
		DArray<int> pIndex,
		DArray<int> index,
		DArray<int> counter,
		DArray<int> ids,
		DArrayList<int> neighbors,
		Real h)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= position.size()) return;

		Coord pos = position[tId];
		int gind = pIndex[tId];

		int cnum = 0;
		int totalNum = counter[gind];
		for (int i = 0; i < totalNum; i++)
		{
			int nbId = ids[index[gind] + i];
			Real d_ij = (pos - position[nbId]).norm();
			if (d_ij < h)
				cnum++;
		}

		for (int c = 0; c < neighbors[gind].size(); c++)
		{
			int cId = neighbors[gind][c];
			totalNum = counter[cId];
			for (int i = 0; i < totalNum; i++)
			{
				int nbId = ids[index[cId] + i];
				Real d_ij = (pos - position[nbId]).norm();
				if (d_ij < h)
				{
					cnum++;
				}
			}
		}

		count[tId] = cnum;
	}
	

	template<typename Real, typename Coord>
	__global__ void NPQA_GetNeighborElements(
		DArrayList<int> nbrIds,
		DArray<Coord> position,
		DArray<int> pIndex,
		DArray<int> index,
		DArray<int> counter,
		DArray<int> ids,
		DArrayList<int> neighbors,
		Real h)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= position.size()) return;

		Coord pos = position[tId];
		int gind = pIndex[tId];

		List<int>& list_i = nbrIds[tId];

		int cnum = 0;
		int totalNum = counter[gind];
		for (int i = 0; i < totalNum; i++)
		{
			int nbId = ids[index[gind] + i];
			Real d_ij = (pos - position[nbId]).norm();
			if (d_ij < h)
			{
				list_i.insert(nbId);
				cnum++;
			}
		}

		for (int c = 0; c < neighbors[gind].size(); c++)
		{
			int cId = neighbors[gind][c];
			totalNum = counter[cId];
			for (int i = 0; i < totalNum; i++)
			{
				int nbId = ids[index[cId] + i];
				Real d_ij = (pos - position[nbId]).norm();
				if (d_ij < h)
				{
					list_i.insert(nbId);
					cnum++;
				}
			}
		}
	}

	template<typename TDataType>
	void NeighborPointQueryAdaptive<TDataType>::requestDynamicNeighborIds()
	{
		// Prepare inputs
		auto& points = this->inPosition()->getData();

		// Prepare outputs
		if (this->outNeighborIds()->isEmpty())
			this->outNeighborIds()->allocate();

		auto& nbrIds = this->outNeighborIds()->getData();

		DArray<uint> count(points.size());
		cuExecute(points.size(),
			NPQA_CountNeighborSize,
			count,
			points,
			m_pIndex,
			m_index,
			m_counter,
			m_ids,
			m_neighbor,
			this->inRadius()->getData());

		nbrIds.resize(count);
		cuExecute(points.size(),
			NPQA_GetNeighborElements,
			nbrIds,
			points,
			m_pIndex,
			m_index,
			m_counter,
			m_ids,
			m_neighbor,
			this->inRadius()->getData());

		count.clear();
	}






	//template<typename Real, typename Coord, typename TDataType>
	//__global__ void K_ComputeNeighborFixed(
	//	DArrayList<int> nbrIds, 
	//	DArray<Coord> position_new,
	//	DArray<Coord> position, 
	//	GridHash<TDataType> hash, 
	//	Real h,
	//	int sizeLimit,
	//	DArray<int> heapIDs,
	//	DArray<Real> heapDistance)
	//{
	//	int pId = threadIdx.x + (blockIdx.x * blockDim.x);
	//	if (pId >= position_new.size()) return;

	//	//TODO: used shared memory for speedup
	//	int* ids(heapIDs.begin() + pId * sizeLimit);// = new int[nbrLimit];
	//	Real* distance(heapDistance.begin() + pId * sizeLimit);// = new Real[nbrLimit];

	//	for (int i = 0; i < sizeLimit; i++) {
	//		ids[i] = INT_MAX;
	//		distance[i] = REAL_MAX;
	//	}

	//	Coord pos_ijk = position_new[pId];
	//	int3 gId3 = hash.getIndex3(pos_ijk);

	//	int counter = 0;
	//	for (int c = 0; c < 27; c++)
	//	{
	//		int cId = hash.getIndex(gId3.x + offset_nq[c][0], gId3.y + offset_nq[c][1], gId3.z + offset_nq[c][2]);
	//		if (cId >= 0) {
	//			int totalNum = hash.getCounter(cId);// min(hash.getCounter(cId), hash.npMax);
	//			for (int i = 0; i < totalNum; i++) {
	//				int nbId = hash.getParticleId(cId, i);
	//				float d_ij = (pos_ijk - position[nbId]).norm();
	//				if (d_ij < h)
	//				{
	//					if (counter < sizeLimit)
	//					{
	//						ids[counter] = nbId;
	//						distance[counter] = d_ij;

	//						heapify_up(ids, distance, counter);
	//						counter++;
	//					}
	//					else
	//					{
	//						if (d_ij < distance[0])
	//						{
	//							ids[0] = nbId;
	//							distance[0] = d_ij;

	//							heapify_down(ids, distance, 0, counter);
	//						}
	//					}
	//					
	//				}
	//			}
	//		}
	//	}

	//	List<int>& list_i = nbrIds[pId];

	//	heap_sort(ids, distance, counter);
	//	for (int bId = 0; bId < counter; bId++)
	//	{
	//		list_i.insert(ids[bId]);
	//	}
	//}

	//template<typename TDataType>
	//void NeighborPointQueryAdaptive<TDataType>::requestFixedSizeNeighborIds()
	//{
	//	// Prepare inputs
	//	auto& points	= this->inPosition()->getData();
	//	auto& other		= this->inOther()->isEmpty() ? this->inPosition()->getData() : this->inOther()->getData();
	//	auto h			= this->inRadius()->getData();

	//	// Prepare outputs
	//	if (this->outNeighborIds()->isEmpty())
	//		this->outNeighborIds()->allocate();

	//	auto& nbrIds = this->outNeighborIds()->getData();

	//	uint numPt  = this->inPosition()->getDataPtr()->size();
	//	uint sizeLimit = this->varSizeLimit()->getData();
	//	
	//	nbrIds.resize(numPt, sizeLimit);

	//	// Construct hash grid
	//	Reduction<Coord> reduce;
	//	Coord hiBound = reduce.maximum(points.begin(), points.size());
	//	Coord loBound = reduce.minimum(points.begin(), points.size());

	//	GridHash<TDataType> hashGrid;
	//	hashGrid.setSpace(h, loBound - Coord(h), hiBound + Coord(h));
	//	hashGrid.clear();
	//	hashGrid.construct(points);

	//	DArray<int> ids(numPt * sizeLimit);
	//	DArray<Real> distance(numPt * sizeLimit);
	//	cuExecute(numPt,
	//		K_ComputeNeighborFixed,
	//		nbrIds,
	//		other,
	//		points,
	//		hashGrid,
	//		h,
	//		sizeLimit,
	//		ids,
	//		distance);

	//	ids.clear();
	//	distance.clear();
	//	//hashGrid.clear();
	//	hashGrid.release();
	//}

	DEFINE_CLASS(NeighborPointQueryAdaptive);
}