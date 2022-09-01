#include "NeighborTriQueryOctree.h"
#include "Primitive3D.h"
#include "Topology/EdgeSet.h"
#include "SceneGraph.h"
#include "Collision/CollisionDetectionBroadPhase.h"


namespace dyno
{
	
	IMPLEMENT_TCLASS(NeighborTriQueryOctree, TDataType)


	template<typename TDataType>
	NeighborTriQueryOctree<TDataType>::NeighborTriQueryOctree()
		: ComputeModule()
	{
		this->inRadius()->setValue(Real(0.011));
		m_broadPhaseCD = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();
	}

	template<typename TDataType>
	NeighborTriQueryOctree<TDataType>::~NeighborTriQueryOctree()
	{
	}

	template<typename TDataType>
	void NeighborTriQueryOctree<TDataType>::compute()
	{
		//printf("NBR_COMPUTE: Tri\n");
		if (!this->inPosition()->isEmpty() && !this->inTriPosition()->isEmpty() && !this->inTriangles()->isEmpty())
		{

			int p_num = this->inPosition()->size();
			//printf("p_num = %d\n", p_num);
			if (p_num == 0) return;
			if (m_queryAABB.size() != p_num)
			{
				m_queryAABB.resize(p_num);
			}
			
			int t_num = this->inTriangles()->size();
			if (t_num == 0) return;
			if (m_queriedAABB.size() != t_num)
			{
				m_queriedAABB.resize(t_num);
			}

		//	printf("NBR_COMPUTE  triSize: %d\n", inTriPosition()->getData().size());
			
			cuExecute(p_num,
				NTQ_SetupAABB,
				m_queryAABB,
				this->inPosition()->getData(),
				this->inRadius()->getData()*0.9);

			cuExecute(t_num,
				NTQ_SetupAABB,
				m_queriedAABB,
				this->inTriPosition()->getData(),
				this->inTriangles()->getData());

			Real radius = this->inRadius()->getData();

			m_broadPhaseCD->varGridSizeLimit()->setValue(2 * radius);
			m_broadPhaseCD->inSource()->assign(m_queryAABB);
			m_broadPhaseCD->inTarget()->assign(m_queriedAABB);

			m_broadPhaseCD->update();

			auto& nbr = m_broadPhaseCD->outContactList()->getData();

			if (this->outNeighborIds()->size() != p_num)
			{
				this->outNeighborIds()->allocate();
				DArray<int> nbrNum;
				nbrNum.resize(p_num);
				nbrNum.reset();
				auto& nbrIds = this->outNeighborIds()->getData();
				nbrIds.resize(nbrNum);
				nbrNum.clear();

				//this->outNeighborIds()->getData().resize(p_num);
			}
			auto& nbrIds = this->outNeighborIds()->getData();


		//	m_broadPhaseCD->outContactList()->connect(this->outNeighborIds());
			//printf("NBTQ Octree: outContactList Size: %d\n", nbr.elementSize());
			
			//new
			
			DArray<int> nbrNum;
			nbrNum.resize(p_num);
			nbrNum.reset();
			//printf("NBR_COMPUTE outContactList(): %d \n", m_broadPhaseCD->outContactList()->getData().size());
			int sum1 = nbr.elementSize();
			//printf("NBTQ Octree: outContactList Size(sum1): %d\n", sum1);
			if (sum1 > 0)
			{
				//printf("one %d %d\n", nbrNum.size(), p_num);
				cuExecute(p_num,
					NTQ_Narrow_Count,
					nbr,
					this->inPosition()->getData(),
					this->inTriPosition()->getData(),
					this->inTriangles()->getData(),
					nbrNum,
					this->inRadius()->getData()*0.9);

				nbrIds.resize(nbrNum);
				
				int sum = m_reduce.accumulate(nbrNum.begin(), nbrNum.size());
			//	printf("NBTQ Octree: sum: %d\n", sum);
				cuSynchronize();
				if (sum > 0)
				{
					cuExecute(p_num,
						NTQ_Narrow_Set,
						nbr,
						nbrIds,
						this->inPosition()->getData(),
						this->inTriPosition()->getData(),
						this->inTriangles()->getData(),
						this->inRadius()->getData()*0.9);

					nbrNum.clear();
					cuSynchronize();
				}

				//printf("outnbrSize = %d\n", this->outNeighborIds()->size());
			}
			
		}
		//printf("NBR_COMPUTE end!\n");
	}

	template<typename Real, typename Coord>
	__global__ void NTQ_SetupAABB(
		DArray<AABB> boundingBox,
		DArray<Coord> position,
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
		DArray<AABB> boundingBox,
		DArray<Coord> vertex,
		DArray<TopologyModule::Triangle> tIndex)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;

		AABB box;
		TopologyModule::Triangle index = tIndex[tId];

		Coord v0 = vertex[index[0]];
		Coord v1 = vertex[index[1]];
		Coord v2 = vertex[index[2]];

		box.v0 = minimum(v0, minimum(v1, v2));
		box.v1 = maximum(v0, maximum(v1, v2));

		boundingBox[tId] = box;
	}

	template<typename Coord>
	__global__ void NTQ_Narrow_Count(
		DArrayList<int> nbr,
		DArray<Coord> position,
		DArray<Coord> vertex,
		DArray<TopologyModule::Triangle> triangles,
		DArray<int> count,
		Real radius)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= position.size()) return;
		int cnt = 0;
		
		List<int>& nbrIds_i = nbr[tId];
		int nbSize = nbrIds_i.size();  
		//printf("CNT_OKKKK nbSize: %d\n", nbSize);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbrIds_i[ne];
			Point3D point(position[tId]);
			Triangle3D t3d_n(vertex[triangles[j][0]], vertex[triangles[j][1]], vertex[triangles[j][2]]);
			Real p_dis_t = abs(point.distance(t3d_n));
			if (p_dis_t< radius && p_dis_t > EPSILON
				//&& (((point.project(t3d_n).origin - point.origin) / (point.project(t3d_n).origin - point.origin).norm()).cross(t3d_n.normal() / t3d_n.normal().norm())).norm() < 0.001
				)
			{
				//printf("CNT_OKKKK\n");
				cnt++;
			}
		}
		count[tId] = cnt;
	}

	template<typename Coord>
	__global__ void NTQ_Narrow_Set(
		DArrayList<int> nbr,
		DArrayList<int> nbr_out,
		DArray<Coord> position,
		DArray<Coord> vertex,
		DArray<TopologyModule::Triangle> triangles,
		Real radius)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= position.size()) return;
		int cnt = 0;

		List<int>& nbrIds_i = nbr[tId];
		int nbSize = nbrIds_i.size();  //int nbSize = nbr.getNeighborSize(tId); 
		List<int>& nbrOutIds_i = nbr_out[tId];
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbrIds_i[ne];

			Point3D point(position[tId]);
			Triangle3D t3d_n(vertex[triangles[j][0]], vertex[triangles[j][1]], vertex[triangles[j][2]]);
			Real proj_dist = abs(point.distance(t3d_n));

			if (proj_dist < radius && proj_dist > EPSILON
			//	&& (((point.project(t3d_n).origin - point.origin) / (point.project(t3d_n).origin - point.origin).norm()).cross(t3d_n.normal() / t3d_n.normal().norm())).norm() < 0.001
				)
			{
				nbrOutIds_i.insert(j);
				//cnt++;
			}
		}

	}

	DEFINE_CLASS(NeighborTriQueryOctree);
}