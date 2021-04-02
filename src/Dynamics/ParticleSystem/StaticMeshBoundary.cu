#include "StaticMeshBoundary.h"
#include "Framework/Log.h"
#include "Framework/Node.h"
#include "ParticleSystem/BoundaryConstraint.h"

#include "Topology/DistanceField3D.h"
#include "Topology/TriangleSet.h"
#include "Topology/NeighborTriangleQuery.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(StaticMeshBoundary, TDataType)



	template<typename Real, typename Coord>
	__global__ void K_CD_mesh(
		DArray<Coord> points,
		DArray<Coord> pointsTri,
		DArray<TopologyModule::Triangle> m_triangle_index,
		DArray<Coord> vels,
		NeighborList<int> neighborsTriangle,
		Real radius,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= points.size()) return;
		int nbSizeTri = neighborsTriangle.getNeighborSize(pId);
	//	if (pId == 0)
	//		printf("******************************************%d\n",points.size());

		Coord pos_i = points[pId];
		if (vels[pId].norm() > radius / dt)
			vels[pId] = vels[pId] / vels[pId].norm() * radius / dt;
		Coord vel_tmp = vels[pId];

		
		Coord old_pos = pos_i;
		Coord new_pos(0);
		Real weight(0);
		for (int ne = 0; ne < nbSizeTri; ne++)
		{
				int j = neighborsTriangle.getElement(pId, ne);
				//if(j == 4 || j == 5)
				//printf("j = %d sizeof triangle = %d\n", j,  m_triangle_index.size());
				//if (j >= 0) continue;
				//j *= -1;
				//j--;
				Triangle3D t3d(pointsTri[m_triangle_index[j][0]], pointsTri[m_triangle_index[j][1]], pointsTri[m_triangle_index[j][2]]);

				Point3D p3d(pos_i);
				Point3D nearest_point = p3d.project(t3d);

				Real r = (p3d.distance(t3d));
				r = abs(r);
				Coord n = t3d.normal();
				if (n.norm() > EPSILON)
				{
					n.normalize();
				}
				if (((r) < radius) && abs(r) > EPSILON)
				{
					Point3D pt_neartest = nearest_point;
					Coord3D pt_norm = -pt_neartest.origin + p3d.origin;
					pt_norm /= (r);
					new_pos += pt_neartest.origin + radius * pt_norm;
					weight += 1.0;
				}

		}
		if (weight > EPSILON)
		{
			pos_i = new_pos / weight;
			Coord dir = (pos_i - old_pos) / (pos_i - old_pos).norm();
			vels[pId] -= vels[pId].dot(dir) * dir;

			//printf("%.3lf %.3lf %.3lf *** %.3lf %.3lf %.3lf \n", pos_i[0], pos_i[1], pos_i[2], old_pos[0], old_pos[1], old_pos[2]);
		}
		//points[pId] = Coord(0);
		points[pId] = pos_i;

		
		//printf("%.3lf %.3lf %.3lf *** %.3lf %.3lf %.3lf \n", points[pId][0], points[pId][1], points[pId][2], vels[pId][0], vels[pId][1], vels[pId][2]);
	}

	template<typename Coord>
	__global__ void TEST_mesh(
		DArray<Coord> points,
		DArray<Coord> vels
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= points.size()) return;
		//printf("YES\n");
		if (points[pId].norm() > EPSILON|| vels[pId].norm()) printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@ERROR\n");
	}

	template<typename TDataType>
	StaticMeshBoundary<TDataType>::StaticMeshBoundary()
		: Node()
	{
		radius.setValue(0.005);
		m_nbrQuery = this->template addComputeModule<NeighborTriangleQuery<TDataType>>("neighborhood");
		radius.connect(m_nbrQuery->inRadius());
		this->currentParticlePosition()->connect(m_nbrQuery->inPosition());
		this->currentTriangleVertex()->connect(m_nbrQuery->inTriangleVertex());

		this->currentTriangleIndex()->connect(m_nbrQuery->inTriangleIndex());
	}

	template<typename TDataType>
	StaticMeshBoundary<TDataType>::~StaticMeshBoundary()
	{
	}

	template<typename TDataType>
	void StaticMeshBoundary<TDataType>::loadMesh(std::string filename)
	{
		printf("inside load\n");
		auto boundary = std::make_shared<TriangleSet<TDataType>>();
		boundary->loadObjFile(filename);
		m_obstacles.push_back(boundary);
		printf("outside load\n");
	}

	template<typename TDataType>
	bool StaticMeshBoundary<TDataType>::initialize()
	{
		return true;
	}

	template<typename TDataType>
	bool StaticMeshBoundary<TDataType>::resetStatus()
	{
		auto triangle_index		= this->currentTriangleIndex();
		auto triangle_vertex	= this->currentTriangleVertex();

		auto particle_position	= this->currentParticlePosition();
		auto particle_velocity	= this->currentParticleVelocity();

		int sum_tri_index = 0;
		int sum_tri_pos = 0;
		int sum_poi_pos = 0;

		for (size_t t = 0; t < m_obstacles.size(); t++)
		{
			sum_tri_index += m_obstacles[t]->getTriangles()->size();
			sum_tri_pos += m_obstacles[t]->getPoints().size();
		}
		triangle_index->setElementCount(sum_tri_index);
		triangle_vertex->setElementCount(sum_tri_pos);

		int start_pos = 0;
		int start_tri = 0;
		for (size_t t = 0; t < m_obstacles.size(); t++)
		{
			DArray<Coord> posTri = m_obstacles[t]->getPoints();
			DArray<Triangle>* idxTri = m_obstacles[t]->getTriangles();
			int num_p = posTri.size();
			int num_i = idxTri->size();
			if (num_p > 0)
			{
				cudaMemcpy(triangle_vertex->getReference()->begin() + start_pos, posTri.begin(), num_p * sizeof(Coord), cudaMemcpyDeviceToDevice);
				start_pos += num_p;
			}
			if(num_i > 0)
			{
				cudaMemcpy(triangle_index->getReference()->begin() + start_tri, idxTri->begin(), num_i * sizeof(Triangle), cudaMemcpyDeviceToDevice);
				start_tri += num_i;
			}
			
		}
		cuSynchronize();

		auto pSys = this->getParticleSystems();
		for (int i = 0; i < pSys.size(); i++)
		{
			DeviceArrayField<Coord>* posFd = pSys[i]->currentPosition();
			sum_poi_pos += posFd->getElementCount();
		}
		particle_position->setElementCount(sum_poi_pos);
		particle_velocity->setElementCount(sum_poi_pos);
		int start_point = 0;
		for (int i = 0; i < pSys.size(); i++)
		{
			DeviceArrayField<Coord>* posFd = pSys[i]->currentPosition();
			DeviceArrayField<Coord>* velFd = pSys[i]->currentVelocity();
			int num = posFd->getElementCount();
			if (num > 0)
			{
				cudaMemcpy(particle_position->getReference()->begin() + start_point, posFd->getValue().begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
				cudaMemcpy(particle_velocity->getReference()->begin() + start_point, velFd->getValue().begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
				start_point += num;
			}
		}

		
		m_nbrQuery->initialize();
		m_nbrQuery->compute();

		return Node::resetStatus();
	}

	template<typename TDataType>
	void StaticMeshBoundary<TDataType>::advance(Real dt)
	{
		auto triangle_index		= this->currentTriangleIndex();
		auto triangle_vertex	= this->currentTriangleVertex();

		auto particle_position	= this->currentParticlePosition();
		auto particle_velocity	= this->currentParticleVelocity();

		int total_num = 0;
		auto pSys = this->getParticleSystems();
		for (int i = 0; i < pSys.size(); i++)
		{
			total_num += pSys[i]->currentPosition()->getElementCount();
		}

		if (total_num > 0)
		{
			particle_position->setElementCount(total_num);
			particle_velocity->setElementCount(total_num);

			int start_point = 0;
			for (int i = 0; i < pSys.size(); i++)
			{
				DeviceArrayField<Coord>* posFd = pSys[i]->currentPosition();
				DeviceArrayField<Coord>* velFd = pSys[i]->currentVelocity();
				int num = posFd->getElementCount();
				if (num > 0)
				{
					cudaMemcpy(particle_position->getReference()->begin() + start_point, posFd->getValue().begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
					cudaMemcpy(particle_velocity->getReference()->begin() + start_point, velFd->getValue().begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
					start_point += num;
				}
			}

			m_nbrQuery->compute();

			DArray<Coord>& posRef = particle_position->getValue();
			DArray<Coord>& velRef = particle_velocity->getValue();

			cuExecute(total_num, K_CD_mesh,
				posRef,
				triangle_vertex->getValue(),
				triangle_index->getValue(),
				velRef,
				m_nbrQuery->outNeighborhood()->getValue(),
				radius.getValue(),
				dt);


			start_point = 0;
			for (int i = 0; i < pSys.size(); i++)
			{
				DeviceArrayField<Coord>* posFd = pSys[i]->currentPosition();
				DeviceArrayField<Coord>* velFd = pSys[i]->currentVelocity();

				int num = posFd->getElementCount();
				if (num > 0)
				{
					cudaMemcpy(posFd->getValue().begin(), posRef.begin() + start_point, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
					cudaMemcpy(velFd->getValue().begin(), velRef.begin() + start_point, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
					start_point += num;
				}
			}
		}
	}

	DEFINE_CLASS(StaticMeshBoundary);
}
