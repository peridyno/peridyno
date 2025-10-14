#include "VirtualSpatiallyAdaptiveStrategy.h"
#include "Node.h"
#include "ParticleSystem/Module/SummationDensity.h"
#include "Topology/GridHash.h"

#include <thrust/sort.h>


namespace dyno
{
	

	IMPLEMENT_TCLASS(VirtualSpatiallyAdaptiveStrategy, TDataType)

	template<typename TDataType>
	VirtualSpatiallyAdaptiveStrategy<TDataType>::VirtualSpatiallyAdaptiveStrategy()
		: VirtualParticleGenerator<TDataType>()
	{

		this->varSamplingDistance()->setValue(Real(0.005));
		this->varRestDensity()->setValue(Real(1000));
		gridSize = this->varSamplingDistance()->getData();
		
		
	}

	template<typename TDataType>
	VirtualSpatiallyAdaptiveStrategy<TDataType>::~VirtualSpatiallyAdaptiveStrategy()
	{

	}

	template<typename Real, typename Coord>
	__global__ void AFV_VirtualPositionCompute(
		DArray<Coord> pos,
		DArray<Coord> vpos,
		Coord loPoint,
		int nx,
		int ny,
		int nz,
		Real ds
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vpos.size()) return;
		if (nx == 0) return;

		vpos[pId][1] = loPoint[1] + Real(pId / (nx * nz)) * ds;
		vpos[pId][0] = loPoint[0] + Real(pId % nx) * ds;
		vpos[pId][2] = loPoint[2] + Real(pId % (nx * nz) / nx) * ds;
	}

	/*
	*@brief	Virtual particles' candinate points
	*		Every particle has 8 neighbors.
	*/
	template<typename Real, typename Coord>
	__global__ void AFV_AnchorNeighbor_8
		(
			DArray<Coord> anchorPoint,
			DArray<Coord> pos,
			Coord origin,
			Real dh
		)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= pos.size()) return;



		Coord pos_ref = pos[id] - origin + Coord (dh/4.0);
		//Coord pos_ref = pos[id] - origin;
		Coord a(0);

		a[0] = (Real)((int)(floor(pos_ref[0] / dh))) * dh;
		a[1] = (Real)((int)(floor(pos_ref[1] / dh))) * dh;
		a[2] = (Real)((int)(floor(pos_ref[2] / dh))) * dh;

		anchorPoint[id * 8] = Coord(a[0], a[1], a[2]);
		anchorPoint[id * 8 + 1] = Coord(a[0] + dh, a[1], a[2]);
		anchorPoint[id * 8 + 2] = Coord(a[0] + dh, a[1] + dh, a[2]);
		anchorPoint[id * 8 + 3] = Coord(a[0] + dh, a[1] + dh, a[2] + dh);
		anchorPoint[id * 8 + 4] = Coord(a[0], a[1] + dh, a[2]);
		anchorPoint[id * 8 + 5] = Coord(a[0], a[1] + dh, a[2] + dh);
		anchorPoint[id * 8 + 6] = Coord(a[0], a[1], a[2] + dh);
		anchorPoint[id * 8 + 7] = Coord(a[0] + dh, a[1], a[2] + dh);
		
	}


	/*
	*@brief	Virtual particles' candinate points
	*		Every particle has 27 neighbors.
	*/
	template<typename Real, typename Coord>
	__global__ void AFV_AnchorNeighbor_27
	(
		DArray<Coord> anchorPoint,
		DArray<Coord> pos,
		Coord origin,
		Real dh
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= pos.size()) return;

		Coord pos_ref = pos[id] - origin + dh/2;
		Coord a(0);

		a[0] = (Real)((int)(floor(pos_ref[0] / dh))) * dh;
		a[1] = (Real)((int)(floor(pos_ref[1] / dh))) * dh;
		a[2] = (Real)((int)(floor(pos_ref[2] / dh))) * dh;

	
		for (int i = 0; i < 27; i++)
		{
			anchorPoint[id * 27 + i] = Coord(	a[0] + dh * Real(diff_v[i][0]),
												a[1] + dh * Real(diff_v[i][1]),
												a[2] + dh * Real(diff_v[i][2])
			);
		}

	}

	/*
*@brief	Virtual particles' candinate points
*		Every particle has 33 neighbors.
*/
	template<typename Real, typename Coord>
	__global__ void AFV_AnchorNeighbor_33
	(
		DArray<Coord> anchorPoint,
		DArray<Coord> pos,
		Coord origin,
		Real dh
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= pos.size()) return;

		//Coord pos_ref = pos[id] - origin + dh / 2;
		Coord pos_ref = pos[id] - origin + dh / 2;
		Coord a(0);

		a[0] = (Real)((int)(floor(pos_ref[0] / dh))) * dh;
		a[1] = (Real)((int)(floor(pos_ref[1] / dh))) * dh;
		a[2] = (Real)((int)(floor(pos_ref[2] / dh))) * dh;


		for (int i = 0; i < 33; i++)
		{
			anchorPoint[id * 33 + i] = Coord(a[0] + dh * Real(diff_v_33[i][0]),
				a[1] + dh * Real(diff_v_33[i][1]),
				a[2] + dh * Real(diff_v_33[i][2])
			);
		}

	}

	/*
*@brief	Virtual particles' candinate points
*		Every particle has 125 neighbors.
*/
	template<typename Real, typename Coord>
	__global__ void AFV_AnchorNeighbor_125
	(
		DArray<Coord> anchorPoint,
		DArray<Coord> pos,
		Coord origin,
		Real dh
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= pos.size()) return;

		Coord pos_ref = pos[id] - origin + dh / 2;
		Coord a(0);

		a[0] = (Real)((int)(floor(pos_ref[0] / dh))) * dh;
		a[1] = (Real)((int)(floor(pos_ref[1] / dh))) * dh;
		a[2] = (Real)((int)(floor(pos_ref[2] / dh))) * dh;


		for (int i = 0; i < 125; i++)
		{
			anchorPoint[id * 125 + i] = Coord(a[0] + dh * Real(diff_v_125[i][0]),
				a[1] + dh * Real(diff_v_125[i][1]),
				a[2] + dh * Real(diff_v_125[i][2])
			);
		}

	}


	template< typename Coord>
	__global__ void AFV_PositionToMortonCode_TEST
	(
		DArray<uint32_t>  mortonCode,
		DArray<Coord> gridPoint,
		Real dh
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= gridPoint.size()) return;

		uint32_t  x = (uint32_t)(gridPoint[id][0] / dh);
		uint32_t  y = (uint32_t)(gridPoint[id][1] / dh);
		uint32_t  z = (uint32_t)(gridPoint[id][2] / dh);

		uint32_t  xi = x;
		uint32_t  yi = y;
		uint32_t  zi = z;

		uint32_t  key = 0;


		xi = (xi | (xi << 16)) & 0x030000FF;
		xi = (xi | (xi << 8)) & 0x0300F00F;
		xi = (xi | (xi << 4)) & 0x030C30C3;
		xi = (xi | (xi << 2)) & 0x09249249;

		yi = (yi | (yi << 16)) & 0x030000FF;
		yi = (yi | (yi << 8)) & 0x0300F00F;
		yi = (yi | (yi << 4)) & 0x030C30C3;
		yi = (yi | (yi << 2)) & 0x09249249;

		zi = (zi | (zi << 16)) & 0x030000FF;
		zi = (zi | (zi << 8)) & 0x0300F00F;
		zi = (zi | (zi << 4)) & 0x030C30C3;
		zi = (zi | (zi << 2)) & 0x09249249;

		key = xi | (yi << 1) | (zi << 2);

		uint32_t  xv = key & 0x09249249;
		uint32_t  yv = (key >> 1) & 0x09249249;
		uint32_t  zv = (key >> 2) & 0x09249249;

		xv = ((xv >> 2) | xv) & 0x030C30C3;
		xv = ((xv >> 4) | xv) & 0x0300F00F;
		xv = ((xv >> 8) | xv) & 0x030000FF;
		xv = ((xv >> 16) | xv) & 0x000003FF;

		yv = ((yv >> 2) | yv) & 0x030C30C3;
		yv = ((yv >> 4) | yv) & 0x0300F00F;
		yv = ((yv >> 8) | yv) & 0x030000FF;
		yv = ((yv >> 16) | yv) & 0x000003FF;

		zv = ((zv >> 2) | zv) & 0x030C30C3;
		zv = ((zv >> 4) | zv) & 0x0300F00F;
		zv = ((zv >> 8) | zv) & 0x030000FF;
		zv = ((zv >> 16) | zv) & 0x000003FF;
	
		if(x!=xv || y!=yv || z!=zv)
			printf("gridPoint: %d, %d, %d || key: %d || invert: %d, %d, %d \r\n", x, y, z, key, xv, yv, zv);

	}

	/*
	*@brief	Positions(3-dimension) convert to Morton Codes
	*/
	template< typename Coord>
	__global__ void AFV_PositionToMortonCode
	(
		DArray<uint32_t>  mortonCode,
		DArray<Coord> gridPoint,
		Real dh
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= gridPoint.size()) return;

		uint32_t  x = (uint32_t)(gridPoint[id][0] / dh +  100 * EPSILON);
		uint32_t  y = (uint32_t)(gridPoint[id][1] / dh +  100 * EPSILON);
		uint32_t  z = (uint32_t)(gridPoint[id][2] / dh +  100 * EPSILON);

		uint32_t  xi = x;
		uint32_t  yi = y;
		uint32_t  zi = z;

		uint32_t  key = 0;


		xi = (xi | (xi << 16)) & 0x030000FF;
		xi = (xi | (xi << 8)) & 0x0300F00F;
		xi = (xi | (xi << 4)) & 0x030C30C3;
		xi = (xi | (xi << 2)) & 0x09249249;

		yi = (yi | (yi << 16)) & 0x030000FF;
		yi = (yi | (yi << 8)) & 0x0300F00F;
		yi = (yi | (yi << 4)) & 0x030C30C3;
		yi = (yi | (yi << 2)) & 0x09249249;

		zi = (zi | (zi << 16)) & 0x030000FF;
		zi = (zi | (zi << 8)) & 0x0300F00F;
		zi = (zi | (zi << 4)) & 0x030C30C3;
		zi = (zi | (zi << 2)) & 0x09249249;

		key = xi | (yi << 1) | (zi << 2);

		mortonCode[id] = key;
	}


	/*
	*@brief	Morton Codes convert to Positions(3-dimension) 
	*/
	template< typename Coord>
	__global__ void AFV_MortonCodeToPosition
	(
		DArray<Coord> gridPoint,
		DArray<uint32_t>  mortonCode,
		Real dh
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= mortonCode.size()) return;


		uint32_t key = mortonCode[id];

		uint32_t  xv = key & 0x09249249;
		uint32_t  yv = (key >> 1) & 0x09249249;
		uint32_t  zv = (key >> 2) & 0x09249249;

		xv = ((xv >> 2) | xv) & 0x030C30C3;
		xv = ((xv >> 4) | xv) & 0x0300F00F;
		xv = ((xv >> 8) | xv) & 0x030000FF;
		xv = ((xv >> 16) | xv) & 0x000003FF;

		yv = ((yv >> 2) | yv) & 0x030C30C3;
		yv = ((yv >> 4) | yv) & 0x0300F00F;
		yv = ((yv >> 8) | yv) & 0x030000FF;
		yv = ((yv >> 16) | yv) & 0x000003FF;

		zv = ((zv >> 2) | zv) & 0x030C30C3;
		zv = ((zv >> 4) | zv) & 0x0300F00F;
		zv = ((zv >> 8) | zv) & 0x030000FF;
		zv = ((zv >> 16) | zv) & 0x000003FF;

		Real x = (float)(xv)* dh;
		Real y = (float)(yv)* dh;
		Real z = (float)(zv)* dh;

		gridPoint[id] = Coord(x, y, z);
	}


	/*
	*@brief	Search for adjacent elements with the same value
	*/
	__global__ void AFV_RepeatedElementSearch
	(
		DArray<uint32_t>  morton,
		DArray<uint32_t>  counter  
		)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= morton.size()) return;


		if (id == 0 || morton[id] != morton[id - 1])
		{
			counter[id] = 1;
		}
		else
			counter[id] = 0;
	}

	/*
	*@brief	Accumulate the number of non-repeating elements
	*/
	__global__ void AFV_nonRepeatedElementsCal
	(
		DArray<uint32_t>  non_repeated_elements,
		DArray<uint32_t>  post_elements,
		DArray<uint32_t> counter
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= post_elements.size()) return;

		if (id == 0 || post_elements[id] != post_elements[id - 1])
		{
			non_repeated_elements[counter[id]] = post_elements[id];
		}
	}




	template< typename Coord>
	__global__ void AFV_CopyToVpos(
		DArray<Coord> vpos,
		Coord origin,
		DArray<Coord>  elements

	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= elements.size()) return;
		vpos[id] = elements[id] + origin;
	}


	__global__ void AFV_ElementsPrint(
		DArray<uint32_t>  elements
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= elements.size()) return;

		if (id == 100)
		{
			printf("size: %d \r\n", elements.size());
			for(int i = 0; i < elements.size(); i++)
				printf("%d\t",  elements[i]);
		}
	}


	template< typename Real>
	__global__ void AFV_SortMortonCodePrint
	(
		DArray<uint32_t>  mortonCode,
		DArray<uint32_t>  counter,
		Real dh
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= mortonCode.size()) return;

		if (id == 0)
		{
			for (int i = 0; i < mortonCode.size(); i++)
			{
				printf("%d , %d \t", counter[i], mortonCode[i]);
			}
		}
	}



	template<typename TDataType>
	void VirtualSpatiallyAdaptiveStrategy<TDataType>::constrain()
	{
		cudaDeviceSynchronize();
		gridSize = this->varSamplingDistance()->getData();
		int num = this->inRPosition()->size();
		
		if (num == 0) return;

		int node_num = num * (int)(this->varCandidatePointCount()->getDataPtr()->currentKey());

		if (m_anchorPoint.size() != node_num)
		{
			m_anchorPoint.resize(node_num);
		}

		if (m_anchorPointCodes.size() != node_num)
		{
			m_anchorPointCodes.resize(node_num);
		}

		if (m_nonRepeatedCount.size() != node_num)
		{
			m_nonRepeatedCount.resize(node_num);
		}

		Reduction<Coord> reduce;
		Coord hiBound = reduce.maximum(this->inRPosition()->getData().begin(), this->inRPosition()->getData().size());
		Coord loBound = reduce.minimum(this->inRPosition()->getData().begin(), this->inRPosition()->getData().size());

		int padding = 2;
		hiBound += Coord((padding + 1)* gridSize);
		loBound -= Coord(padding * gridSize);

		loBound[0] = (Real)((int)(floor(loBound[0] / gridSize)))*gridSize;
		loBound[1] = (Real)((int)(floor(loBound[1] / gridSize)))*gridSize;
		loBound[2] = (Real)((int)(floor(loBound[2] / gridSize)))*gridSize;
	
		origin = Coord(loBound[0], loBound[1], loBound[2]);

		if (this->varCandidatePointCount()->getValue() == CandidatePointCount::neighbors_8) {
			cuExecute(this->inRPosition()->getData().size(),
				AFV_AnchorNeighbor_8,
				m_anchorPoint,
				this->inRPosition()->getData(),
				origin,
				gridSize
			);
		
		}
		else if (this->varCandidatePointCount()->getValue() == CandidatePointCount::neighbors_27) {
			cuExecute(this->inRPosition()->getData().size(),
				AFV_AnchorNeighbor_27,
				m_anchorPoint,
				this->inRPosition()->getData(),
				origin,
				gridSize
			);
		}
		else if (this->varCandidatePointCount()->getValue() == CandidatePointCount::neighbors_33) {
			cuExecute(this->inRPosition()->getData().size(),
				AFV_AnchorNeighbor_33,
				m_anchorPoint,
				this->inRPosition()->getData(),
				origin,
				gridSize
			);
		}
		else if (this->varCandidatePointCount()->getValue() == CandidatePointCount::neighbors_125) {
			cuExecute(this->inRPosition()->getData().size(),
				AFV_AnchorNeighbor_125,
				m_anchorPoint,
				this->inRPosition()->getData(),
				origin,
				gridSize
			);
		}
		else
		{
			std::cout << "*VIRTUAL PARTICLE:: AdaptiveVirtualPosition ERROR!!!!!" << std::endl;
		}

		cuExecute(m_anchorPointCodes.size(),
			AFV_PositionToMortonCode,
			m_anchorPointCodes,
			m_anchorPoint,
			gridSize
		);

		thrust::sort(thrust::device, m_anchorPointCodes.begin(), m_anchorPointCodes.begin() + m_anchorPointCodes.size());

		cuExecute(m_anchorPointCodes.size(),
			AFV_RepeatedElementSearch,
			m_anchorPointCodes,
			m_nonRepeatedCount
		);


		int candidatePoint_num = thrust::reduce(thrust::device, m_nonRepeatedCount.begin(), m_nonRepeatedCount.begin() + m_nonRepeatedCount.size(), (int)0, thrust::plus<uint32_t>());

		thrust::exclusive_scan(thrust::device, m_nonRepeatedCount.begin(), m_nonRepeatedCount.begin() + m_nonRepeatedCount.size(), m_nonRepeatedCount.begin());

		m_candidateCodes.resize(candidatePoint_num);


		cuExecute(m_anchorPointCodes.size(),
			AFV_nonRepeatedElementsCal,
			m_candidateCodes,
			m_anchorPointCodes,
			m_nonRepeatedCount
		);

		cuSynchronize();

		m_virtual_position.resize(m_candidateCodes.size());

		cuExecute(m_candidateCodes.size(),
			AFV_MortonCodeToPosition,
			m_virtual_position,
			m_candidateCodes,
			gridSize
		);


		if (this->outVirtualParticles()->isEmpty())
		{
			this->outVirtualParticles()->allocate();
		}
		
		this->outVirtualParticles()->resize(m_virtual_position.size());
		
		cuExecute(m_virtual_position.size(),
			AFV_CopyToVpos,
			this->outVirtualParticles()->getData(),
			origin,
			m_virtual_position
		);

		std::cout << "*DUAL-ISPH::SpatiallyAdaptiveStrategy(S.C.)::Model_"<< this->varCandidatePointCount()->getDataPtr()->currentKey() 
			<<"::RealPoints:" << this->inRPosition()->size() << "||VirtualPoints:" << this->outVirtualParticles()->size() <<std::endl;
	}



	DEFINE_CLASS(VirtualSpatiallyAdaptiveStrategy);

}