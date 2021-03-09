#include "NeighborElementQuery.h"
#include "Framework/FieldArray.h"
#include "Collision/CollisionDetectionBroadPhase.h"
#include "Topology/Primitive3D.h"
#include "NeighborConstraints.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(NeighborElementQuery, TDataType)
	typedef typename TOrientedBox3D<Real> Box3D;
	

	template<typename TDataType>
	NeighborElementQuery<TDataType>::NeighborElementQuery()
		: ComputeModule()
	{
		this->inRadius()->setValue(Real(0.011));

		m_broadPhaseCD = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();
	}

	template<typename TDataType>
	NeighborElementQuery<TDataType>::~NeighborElementQuery()
	{
	}

	template<typename Real, typename Coord>
	__global__ void NEQ_SetupAABB(
		GArray<AABB> boundingBox,
		GArray<Coord> position,
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

	template<typename Box3D>
	__global__ void NEQ_SetupAABB(
		GArray<AABB> boundingBox,
		GArray<Box3D> boxes,
		GArray<Sphere3D> spheres,
		int start_sphere,
		int start_box)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;

		printf("spheres size: %d box size: %d\n", spheres.size(), boxes.size());
		AABB box;

		if (tId >= start_box)
		{
			box = boxes[tId - start_box].aabb();
		}
		else
		{
			box = spheres[tId - start_sphere].aabb();
		}
		
		boundingBox[tId] = box;
	}
	template<typename Coord, typename Box3D>
	__global__ void NEQ_Narrow_Count(
		GArray<Coord> pos,
		NeighborList<int> nbr,
		GArray<Box3D> boxes,
		GArray<Sphere3D> spheres,
		GArray<int> count,
		int start_sphere,
		int start_box,
		Real radius)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= pos.size()) return;

		int cnt = 0;
		Point3D p(pos[tId]);

		int nbSize = nbr.getNeighborSize(tId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbr.getElement(tId, ne);
			if (j >= start_box)
			{
				Bool inside;
				Point3D p_project = p.project(boxes[j - start_box], inside);
				if (inside == true || p_project.distance(p) < radius)
				{
					cnt++;
				}
			}
			else
			{

			}
		}

		count[tId] = cnt;
	}

	template<typename Box3D>
	__global__ void NEQ_Narrow_Count(
		NeighborList<int> nbr,
		GArray<Box3D> boxes,
		GArray<Sphere3D> spheres,
		GArray<int> count,
		int start_sphere,
		int start_box)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nbr.size()) return;
		int cnt = 0;
		if (tId >= start_box)
		{
			int nbSize = nbr.getNeighborSize(tId);
			for (int ne = 0; ne < nbSize; ne++)
			{
				int j = nbr.getElement(tId, ne);
				if (j >= start_box)
				{
					Coord3D inter_norm, p1, p2;
					Real inter_dist;
					if(boxes[tId - start_box].point_intersect(boxes[j - start_box], inter_norm, inter_dist, p1,p2))
					{ 
						cnt++;
					}
					else if (boxes[j - start_box].point_intersect(boxes[tId - start_box], inter_norm, inter_dist, p1, p2))
					{
						cnt++;
					}
				}
				else
				{

				}
			}
		}
		else
		{
			//box = spheres[tId - start_sphere].aabb();
		}
		count[tId] = cnt;
	}

	template<typename Box3D, typename NeighborConstraints>
	__global__ void NEQ_Narrow_Set(
		NeighborList<int> nbr,
		GArray<Box3D> boxes,
		GArray<Sphere3D> spheres,
		NeighborList<int> nbr_out,
		GArray<NeighborConstraints> nbr_cons,
		int start_sphere,
		int start_box)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nbr.size()) return;
		int cnt = 0;
		if (tId >= start_box) //box
		{
			int nbSize = nbr.getNeighborSize(tId);
			for (int ne = 0; ne < nbSize; ne++)
			{
				int j = nbr.getElement(tId, ne);
				if (j >= start_box)
				{
					Coord3D inter_norm1, p11, p21;
					Coord3D inter_norm2, p12, p22;
					Real inter_dist1;
					Real inter_dist2;

				
					int type = 0;
					bool insert_one = boxes[tId - start_box].point_intersect(boxes[j - start_box], inter_norm1, inter_dist1, p11, p21);
					bool insert_two = boxes[j - start_box].point_intersect(boxes[tId - start_box], inter_norm2, inter_dist2, p12, p22);

					if (insert_one && insert_two)
					{
						if (inter_dist1 < inter_dist2) type = 1;
						else type = 2;
					}
					else if (insert_one) type = 1;
					else if (insert_two) type = 2;

					if (type == 1)
					{
						//printf("FROM NEQ: id1: %d id2: %d inter_dist: %.5lf\ninter norm: %.3lf %.3lf %.3lf\np1: %.3lf %.3lf %.3lf\np2: %.3lf %.3lf %.3lf\n", 
						//	tId,j,inter_dist1,
						//	inter_norm1[0], inter_norm1[1], inter_norm1[2],
						//	p11[0], p11[1], p11[2],
						//	p21[0], p21[1], p21[2]
						//	);

						nbr_out.setElement(tId,cnt,j);

						/*set up constraints*/
						int idx_con = nbr_out.getElementIndex(tId, cnt);
						nbr_cons[idx_con] = NeighborConstraints(tId, j, 1, p11, p21, inter_norm1, Coord3D(0, 0, 0));
						nbr_cons[idx_con].s4 = -inter_dist1;
						cnt++;
					}
					else if(type == 2)
					{
						//printf("FROM NEQ: id1: %d id2: %d inter_dist: %.5lf\ninter norm: %.3lf %.3lf %.3lf\np1: %.3lf %.3lf %.3lf\np2: %.3lf %.3lf %.3lf\n",
						//	j, tId, inter_dist2,
						//	inter_norm2[0], inter_norm2[1], inter_norm2[2],
						//	p12[0], p12[1], p12[2],
						//	p22[0], p22[1], p22[2]
					//	);
						nbr_out.setElement(tId, cnt, j);

						/*set up constraints*/
						int idx_con = nbr_out.getElementIndex(tId, cnt);
						nbr_cons[idx_con] = NeighborConstraints(j,tId,1,p12,p22,inter_norm2,Coord3D(0, 0, 0));
						nbr_cons[idx_con].s4 = -inter_dist2;
						
						cnt++;
					}
				}
				else
				{

				}
			}
		}
		else //else
		{
			
		}
	}

	template<typename Coord, typename Box3D, typename NeighborConstraints>
	__global__ void NEQ_Narrow_Set(
		GArray<Coord> pos,
		NeighborList<int> nbr,
		GArray<Box3D> boxes,
		GArray<Sphere3D> spheres,
		NeighborList<int> nbr_out,
		GArray<NeighborConstraints> nbr_cons,
		int start_sphere,
		int start_box,
		Real radius)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= pos.size()) return;
		
		int cnt = 0;
		Point3D p(pos[tId]);

		int nbSize = nbr.getNeighborSize(tId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbr.getElement(tId, ne);
			if (j >= start_box)
			{
				Bool inside;
				Point3D p_project = p.project(boxes[j - start_box], inside);
				if (inside == true || p_project.distance(p) < radius)
				{
					nbr_out.setElement(tId, cnt, j);

					/*set up constraints*/
					int idx_con = nbr_out.getElementIndex(tId, cnt);
					nbr_cons[idx_con] = NeighborConstraints(tId, j, 2, p.origin, p_project.origin, Coord(0, 0, 0), Coord(0, 0, 0));
					nbr_cons[idx_con].s1 = (inside == true) ? (-p_project.distance(p)) : p_project.distance(p);
					nbr_cons[idx_con].s2 = radius;

					cnt++;
				}
			}
			else
			{

			}
		}
	}


	template<typename TDataType>
	bool NeighborElementQuery<TDataType>::initializeImpl()
	{
		compute();
		return true;
	}
	 
	template<typename TDataType>
	void NeighborElementQuery<TDataType>::compute()
	{
		
		if (!this->inPosition()->isEmpty() && this->discreteSet != NULL)
			// NOT SELF COLLISION
		{
			
			int p_num = this->inPosition()->getElementCount();
			if (m_queryAABB.size() != p_num)
			{
				m_queryAABB.resize(p_num);
			}

			int t_num = discreteSet->getBoxes().size() + discreteSet->getSpheres().size();
			if (m_queriedAABB.size() != t_num)
			{
				m_queriedAABB.resize(t_num);
			}

			cuExecute(p_num,
				NEQ_SetupAABB,
				m_queryAABB,
				this->inPosition()->getValue(),
				this->inRadius()->getValue());

			cuExecute(t_num,
				NEQ_SetupAABB,
				m_queriedAABB,
				discreteSet->getBoxes(),
				discreteSet->getSpheres(),
				0,
				discreteSet->getSpheres().size()
				);
			this->inRadius()->setValue(0.0075);
			Real radius = this->inRadius()->getValue();
			//printf("RADIUS!~ : %.10lf\n", this->inRadius()->getValue());
			//this->inRadius()->setValue(0.0075);

			m_broadPhaseCD->varGridSizeLimit()->setValue(2 * radius);



			if (this->outNeighborhood()->getElementCount() != p_num)
			{
				this->outNeighborhood()->setElementCount(p_num);
			}

			m_broadPhaseCD->inSource()->setValue(m_queryAABB);
			m_broadPhaseCD->inTarget()->setValue(m_queriedAABB);
			// 
			m_broadPhaseCD->update();

			//broad phase end

			GArray<int>& nbrNum = this->outNeighborhood()->getValue().getIndex();
			
			
			cuExecute(p_num, 
				NEQ_Narrow_Count,
				this->inPosition()->getValue(),
				m_broadPhaseCD->outContactList()->getValue(),
				discreteSet->getBoxes(),
				discreteSet->getSpheres(),
				nbrNum,
				0,
				discreteSet->getSpheres().size(),
				this->inRadius()->getValue());

			//queryNeighborSize(nbrNum, pos, h);
			
			int sum = m_reduce.accumulate(nbrNum.begin(), nbrNum.size());

			
			m_scan.exclusive(nbrNum, true);
			cuSynchronize();


			if (sum > 0)
			{
				GArray<int>& elements = this->outNeighborhood()->getValue().getElements();
				elements.resize(sum);
				nbr_cons.setElementCount(sum);
				
				
				Real zero = 0;
				cuExecute(p_num,
					NEQ_Narrow_Set,
					this->inPosition()->getValue(),
					m_broadPhaseCD->outContactList()->getValue(),
					discreteSet->getBoxes(),
					discreteSet->getSpheres(),
					outNeighborhood()->getValue(),
					nbr_cons.getValue(),
					zero,
					discreteSet->getSpheres().size(),
					this->inRadius()->getValue());
					
				cuSynchronize();
			}
			
		}
		else if (this->discreteSet != NULL)
			// SELF COLLISION 
		{
			
			int t_num = discreteSet->getBoxes().size() + discreteSet->getSpheres().size();
			if (m_queriedAABB.size() != t_num)
			{
				m_queriedAABB.resize(t_num);
			}
			if(m_queryAABB.size() != t_num)
			{
				m_queryAABB.resize(t_num);
			}


			cuExecute(t_num,
				NEQ_SetupAABB,
				m_queriedAABB,
				discreteSet->getBoxes(),
				discreteSet->getSpheres(),
				0,
				discreteSet->getSpheres().size()
			);

			m_queryAABB.assign(m_queriedAABB);

			Real radius = this->inRadius()->getValue();

			m_broadPhaseCD->varGridSizeLimit()->setValue(2 * radius);
			m_broadPhaseCD->setSelfCollision(true);


			if (this->outNeighborhood()->getElementCount() != t_num)
			{
				this->outNeighborhood()->setElementCount(t_num);
			}

			m_broadPhaseCD->inSource()->setValue(m_queryAABB);
			m_broadPhaseCD->inTarget()->setValue(m_queriedAABB);
			// 
			m_broadPhaseCD->update();
	
			//broad phase end

			GArray<int>& nbrNum = this->outNeighborhood()->getValue().getIndex();
			
			Real zero = 0;
			
			cuExecute(t_num,
				NEQ_Narrow_Count,
				m_broadPhaseCD->outContactList()->getValue(),
				discreteSet->getBoxes(),
				discreteSet->getSpheres(),
				nbrNum,
				zero,
				discreteSet->getSpheres().size());
				
			//queryNeighborSize(nbrNum, pos, h);

			int sum = m_reduce.accumulate(nbrNum.begin(), nbrNum.size());
	

			m_scan.exclusive(nbrNum, true);
			cuSynchronize();

			printf("FROM NEQ: %d", sum);

			GArray<int>& elements = this->outNeighborhood()->getValue().getElements();
			elements.resize(sum);
			nbr_cons.setElementCount(sum);

			if (sum > 0)
			{
				
				

				
				cuExecute(t_num,
					NEQ_Narrow_Set,
					m_broadPhaseCD->outContactList()->getValue(),
					discreteSet->getBoxes(),
					discreteSet->getSpheres(),
					this->outNeighborhood()->getValue(),
					nbr_cons.getValue(),
					0,
					discreteSet->getSpheres().size());
					
				cuSynchronize();
			}
		}
		else
		{
			printf("NeighborElementQuery: Empty discreteSet! \n");
		}
	}

}