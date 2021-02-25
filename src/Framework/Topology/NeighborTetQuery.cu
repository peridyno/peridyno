#include "NeighborTetQuery.h"
#include "Framework/FieldArray.h"
#include "Collision/CollisionDetectionBroadPhase.h"
#include "Topology/Primitive3D.h"
#include "NeighborConstraints.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(NeighborTetQuery, TDataType)
	

	template<typename TDataType>
	NeighborTetQuery<TDataType>::NeighborTetQuery()
		: ComputeModule()
	{
		this->inRadius()->setValue(Real(0.011));

		m_broadPhaseCD = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();
	}

	template<typename TDataType>
	NeighborTetQuery<TDataType>::~NeighborTetQuery()
	{
	}

	template<typename Real, typename Coord>
	__global__ void NTQ_SetupAABB(
		DeviceArray<AABB> boundingBox,
		DeviceArray<Coord> position,
		Real radius)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		AABB box;
		Coord p = position[pId];
		box.v0 = p - radius;
		box.v1 = p + radius;

		boundingBox[pId] = box;
	}

	template<typename Coord>
	__global__ void NTQ_SetupAABB(
		DeviceArray<AABB> boundingBox,
		DeviceArray<TopologyModule::Tetrahedron> tet,
		DeviceArray<Coord> position)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= tet.size()) return;
		AABB box;
		Tet3D tet_i = Tet3D(position[tet[pId][0]], position[tet[pId][1]], position[tet[pId][2]], position[tet[pId][3]]);
		box = tet_i.aabb();
		//Coord p = (position[tet[pId][0]] + position[tet[pId][1]] + position[tet[pId][2]] + position[tet[pId][3]]) / 4.0f;
		//box.v0 = p - 0.000025;//radius;
		//box.v1 = p + 0.000025;// radius;
		boundingBox[pId] = box;
	}
	
	template<typename Coord>
	__global__ void NTQ_Narrow_Count(
		DeviceArray<Coord> pos,
		NeighborList<int> nbr,
		DeviceArray<int> tag,
		DeviceArray<Coord> pos_tet,
		DeviceArray<TopologyModule::Tetrahedron> tet,
		DeviceArray<int> count,
		Real radius)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tet.size()) return;

		int cnt = 0;
		Tet3D tet_i = Tet3D(pos_tet[tet[tId][0]], pos_tet[tet[tId][1]], pos_tet[tet[tId][2]], pos_tet[tet[tId][3]]);

		int nbSize = nbr.getNeighborSize(tId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbr.getElement(tId, ne);
			Tet3D tet_j = Tet3D(pos_tet[tet[j][0]], pos_tet[tet[j][1]], pos_tet[tet[j][2]], pos_tet[tet[j][3]]);
			bool connect = false;
			
			for (int ii = 0; ii < 4; ii++)
			{
				for (int jj = 0; jj < 4; jj++)
					if (tet[tId][ii] == tet[j][jj])
					{
						connect = true;
						break;
					}
				if (connect) break;
			}
			
			if (connect) continue;
			
			Coord3D inter_norm, p1, p2;
			Real inter_dist;
			AABB interBox;
			if (tet_i.aabb().intersect(tet_j.aabb(), interBox))//
				if(tet_i.intersect(tet_j, inter_norm, inter_dist, p1, p2, 0) || tet_j.intersect(tet_i, inter_norm, inter_dist, p1, p2, 0))
			{
					cnt++;
					tag[nbr.getElementIndex(tId,ne)] = 1;
			}
		}
// 		if (cnt != 0)
// 			printf("from count: %d\n", tId, cnt);
		count[tId] = cnt;
	}

	

	template<typename Coord>
	__global__ void NTQ_Narrow_Set(
		DeviceArray<Coord> pos,
		NeighborList<int> nbr,
		DeviceArray<int> tag,
		NeighborList<int> nbr_out,
		DeviceArray<Coord> pos_tet,
		DeviceArray<TopologyModule::Tetrahedron> tet,
		Real radius)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tet.size()) return;
		
		int cnt = 0;
		Tet3D tet_i = Tet3D(pos_tet[tet[tId][0]], pos_tet[tet[tId][1]], pos_tet[tet[tId][2]], pos_tet[tet[tId][3]]);

		int nbSize = nbr.getNeighborSize(tId);
		//printf("%d\n", nbSize);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbr.getElement(tId, ne);
			Tet3D tet_j = Tet3D(pos_tet[tet[j][0]], pos_tet[tet[j][1]], pos_tet[tet[j][2]], pos_tet[tet[j][3]]);

			if(tag[nbr.getElementIndex(tId, ne)])
			{
				nbr_out.setElement(tId, cnt, j);
				cnt++;
			}
			
		}
		
		
	}


	template<typename TDataType>
	bool NeighborTetQuery<TDataType>::initializeImpl()
	{
		compute();
		return true;
	}
	 
	template<typename TDataType>
	void NeighborTetQuery<TDataType>::compute()
	{
		{
			GTimer t1;
			GTimer t2;
			//int p_num = this->inPosition()->getElementCount();
			
			t1.start();
			int t_num = tetSet->getTetrahedrons().size();
			if (m_queriedAABB.size() != t_num)
			{
				m_queriedAABB.resize(t_num);
			}
			if (m_queryAABB.size() != t_num)
			{
				m_queryAABB.resize(t_num);
			}
			/*
			cuExecute(p_num,
				NTQ_SetupAABB,
				m_queryAABB,
				this->inPosition()->getValue(),
				this->inRadius()->getValue());
				*/

			cuExecute(t_num,
				NTQ_SetupAABB,
				m_queriedAABB,
				tetSet->getTetrahedrons(),
				this->inPosition()->getValue()
				);

			Function1Pt::copy(m_queryAABB, m_queriedAABB);
			this->inRadius()->setValue(0.017);
			Real radius = this->inRadius()->getValue();
			

			m_broadPhaseCD->varGridSizeLimit()->setValue(2 * radius);
			m_broadPhaseCD->setSelfCollision(true);


			if (this->outNeighborhood()->getElementCount() != t_num)
			{
				this->outNeighborhood()->setElementCount(t_num);
			}

			t2.start();
			m_broadPhaseCD->inSource()->setValue(m_queryAABB);
			m_broadPhaseCD->inTarget()->setValue(m_queriedAABB);
			// 
			m_broadPhaseCD->update();

			t2.stop();

			
			printf("broad phase time: %f\n", t2.getEclipsedTime());
			//broad phase end

			DeviceArray<int>& nbrNum = this->outNeighborhood()->getValue().getIndex();
			
			DeviceArray<int> tag;
			tag.resize(m_broadPhaseCD->outContactList()->getValue().getElementSize());
			tag.reset();
			
			cuExecute(t_num, 
				NTQ_Narrow_Count,
				this->inPosition()->getValue(),
				m_broadPhaseCD->outContactList()->getValue(),
				tag,
				this->inPosition()->getValue(),
				tetSet->getTetrahedrons(),
				nbrNum,
				this->inRadius()->getValue());

			//queryNeighborSize(nbrNum, pos, h);
			
			int sum = m_reduce.accumulate(nbrNum.begin(), nbrNum.size());

			
			m_scan.exclusive(nbrNum, true);
			cuSynchronize();
			printf("Neighbor Tet Sum: %d %d\n", sum, m_broadPhaseCD->outContactList()->getValue().getElementSize());


			DeviceArray<int>& elements = this->outNeighborhood()->getValue().getElements();
			elements.resize(sum);

			if (sum > 0)
			{
				
				//nbr_cons.setElementCount(sum);
				
				
				
				Real zero = 0;
				cuExecute(t_num,
					NTQ_Narrow_Set,
					this->inPosition()->getValue(),
					m_broadPhaseCD->outContactList()->getValue(),
					tag,
					outNeighborhood()->getValue(),
					this->inPosition()->getValue(),
					tetSet->getTetrahedrons(),
					this->inRadius()->getValue());
					
				cuSynchronize();
			}

			tag.release();
			t1.stop();
			printf("find_all time: %f\n", t1.getEclipsedTime());
			
		}
		
	}

}