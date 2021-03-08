#include "TetCollision.h"
#include "Utility.h"
#include "Framework/Node.h"
#include "Framework/CollidableObject.h"
#include "Collision/CollidablePoints.h"
#include "Topology/NeighborQuery.h"
#include "Topology/Primitive3D.h"
#include "Topology/PrimitiveSweep3D.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(TetCollision, TDataType)

		template<typename TDataType>
	TetCollision<TDataType>::TetCollision()
		: CollisionModel()
	{
	}

	template<typename TDataType>
	TetCollision<TDataType>::~TetCollision()
	{
		m_collidableObjects.clear();
	}

	template<typename TDataType>
	bool TetCollision<TDataType>::isSupport(std::shared_ptr<CollidableObject> obj)
	{
		if (obj->getType() == CollidableObject::POINTSET_TYPE)
		{
			return true;
		}
		return false;
	}


	template<typename TDataType>
	void TetCollision<TDataType>::addCollidableObject(std::shared_ptr<CollidableObject> obj)
	{
		auto derived = std::dynamic_pointer_cast<CollidablePoints<TDataType>>(obj);
		if (obj->getType() == CollidableObject::POINTSET_TYPE)
		{
			m_collidableObjects.push_back(derived);
		}
	}








	template<typename Real, typename Coord>
	__global__ void K_SETUP_TET(
		GArray<Coord> tet_vertex,
		GArray<TopologyModule::Tetrahedron> tet_index,
		NeighborList<int> tet_neighbors,
		GArray<Coord> interNormal,
		GArray<Real> interDistance,
		GArray<int> x_array,
		GArray<int> y_array,
		GArray<int> sum_nbr,
		GArray<Real> volume,
		GArray<Real> volume2
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= tet_index.size()) return;


		int nbrSize = tet_neighbors.getNeighborSize(pId);

		Tet3D tet_i = Tet3D(tet_vertex[tet_index[pId][0]], tet_vertex[tet_index[pId][1]], tet_vertex[tet_index[pId][2]], tet_vertex[tet_index[pId][3]]);
		volume[pId] = tet_i.volume() * 1000.0f;
		//return;
		atomicAdd(&volume2[pId], tet_i.volume() * 1000.0f);



		for (int ne = 0; ne < nbrSize; ne++)
		{
			int j = tet_neighbors.getElement(pId, ne);

			x_array[tet_neighbors.getElementIndex(pId, ne)] = pId;
			y_array[tet_neighbors.getElementIndex(pId, ne)] = j;

			Tet3D tet_j = Tet3D(tet_vertex[tet_index[j][0]], tet_vertex[tet_index[j][1]], tet_vertex[tet_index[j][2]], tet_vertex[tet_index[j][3]]);

			atomicAdd(&sum_nbr[pId], 1);
			atomicAdd(&volume2[pId], tet_j.volume() * 1000.0f);
			atomicAdd(&sum_nbr[j], 1);
			atomicAdd(&volume2[j], tet_i.volume() * 1000.0f);

			Coord3D inter_norm, p1, p2;
			Real inter_dist;

			Real interdist1, interdist2;
			Coord inter_norm1, pp11, pp12;
			Coord inter_norm2, pp21, pp22;

			bool intersect_1 = tet_i.intersect(tet_j, inter_norm1, interdist1, pp11, pp12);
			bool intersect_2 = tet_j.intersect(tet_i, inter_norm2, interdist2, pp21, pp22);

			Coord norm_tmp = ((tet_i.v[0] + tet_i.v[1] + tet_i.v[2] + tet_i.v[3]) / 4.0f -
				(tet_j.v[0] + tet_j.v[1] + tet_j.v[2] + tet_j.v[3]) / 4.0f);
			norm_tmp /= norm_tmp.norm();

			if (intersect_1)/* &&
				((!intersect_2) ||
					interdist1 < interdist2))*/
			{
				interNormal[tet_neighbors.getElementIndex(pId, ne)] = -inter_norm1;
				interDistance[tet_neighbors.getElementIndex(pId, ne)] = max(abs(interdist1) - 0.0005, 0.0f);
			}

			else if (intersect_2)//(tet_j.intersect(tet_i, inter_norm, inter_dist, p1, p2))
			{
				interNormal[tet_neighbors.getElementIndex(pId, ne)] = inter_norm2;
				interDistance[tet_neighbors.getElementIndex(pId, ne)] = max(abs(interdist2) - 0.0005, 0.0f);
			}
		}
		return;
	}


	template<typename Coord>
	__global__ void K_DAMP_TET(
		GArray<Coord> velocity
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velocity.size()) return;
		velocity[pId] *= 1.0f;
	}



	template<typename Real, typename Coord>
	__global__ void K_ITER_TET(
		GArray<int> array_i,
		GArray<int> array_j,
		GArray<Real> volume,
		GArray<TopologyModule::Tetrahedron> tet_index,
		NeighborList<int> tet_neighbors,
		GArray<Coord> interNormal,
		GArray<Real> interDistance,
		GArray<Coord> velocity,
		GArray<Coord> velocity_init,
		GArray<Real> force_previous,
		GArray<Real> force,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= array_i.size()) return;

		int i = array_i[pId];
		int j = array_j[pId];

		int nbrSize_i = tet_neighbors.getNeighborSize(i);
		int nbrSize_j = tet_neighbors.getNeighborSize(j);

		Real ratio = (1.0f / volume[i] + 1.0f / volume[j]);

		Real force_ij =
			(min(abs(interDistance[pId]), 0.001f) / dt / 2.0f - 1.0f * (
				((velocity[tet_index[i][0]] + velocity[tet_index[i][1]] + velocity[tet_index[i][2]] + velocity[tet_index[i][3]]) / 4.0f -
					(velocity[tet_index[j][0]] + velocity[tet_index[j][1]] + velocity[tet_index[j][2]] + velocity[tet_index[j][3]]) / 4.0f).dot(interNormal[pId])))
			;


		Real force_new = force_previous[pId] + force_ij;

		force[pId] = force_new - force_previous[pId];

		force_previous[pId] = force_new;
	}

	template<typename Real, typename Coord>
	__global__ void K_UPDATE_TET(
		GArray<int> array_i,
		GArray<int> array_j,
		GArray<Real> volume,
		GArray<Real> volume2,
		GArray<Coord> interNorm,
		GArray<TopologyModule::Tetrahedron> tet_index,
		GArray<Coord> velocity,
		GArray<Real> force,
		GArray<int> sum_nbr,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= array_i.size()) return;

		int i = array_i[pId];
		int j = array_j[pId];

		Coord force_ij = force[pId] * interNorm[pId]; // (sum_nbr[i] + sum_nbr[j]) * 2.0f;
		Real ratio = 0.025f;//0.5f;//(volume2[j]) / (volume2[j] + volume2[i]);

		//force_ij *= min(ratio, 1.0f - ratio) / 0.5;

		for (int ii = 0; ii < 4; ii++)
			for (int jj = 0; jj < 3; jj++)
			{
				atomicAdd(&velocity[tet_index[i][ii]][jj], force_ij[jj] * ratio);
				atomicAdd(&velocity[tet_index[j][ii]][jj], -force_ij[jj] * ratio);

			}
	}

	template<typename TDataType>
	void TetCollision<TDataType>::doCollision()
	{
		Real radius = 0.005;


		if (m_position.getElementCount() == 0) return;


		printf("%d %d\n", m_position.getElementCount(), m_velocity.getElementCount());
		if (m_position_previous.isEmpty() || m_position_previous.size() != m_position.getElementCount())
		{
			//	m_flip.setElementCount(m_position.getElementCount());
				//printf("")
			posBuf.resize(m_position.getElementCount());
			weights.resize(m_position.getElementCount());
			init_pos.resize(m_position.getElementCount());

			m_position_previous.resize(m_position.getElementCount());
			Function1Pt::copy(m_position_previous, m_position.getValue());
			printf("resize finished %d %d\n", m_position.getElementCount(), m_velocity.getElementCount());

		}
		if (m_tet_vertex.getElementCount() != m_position.getElementCount())
			m_tet_vertex.setElementCount(m_position.getElementCount());


		Function1Pt::copy(init_pos, m_position.getValue());
		Function1Pt::copy(m_tet_vertex.getValue(), m_position.getValue());



		int total_num_particles = m_position.getValue().size();
		int total_num_tets = tetSet->getTetrahedrons().size();
		//printf("SIZE TET: %d\n", tetSet->getTetrahedrons().size());


		GArray<int> x_array;
		GArray<int> y_array;
		GArray<Coord> interNormal;
		GArray<Real> interDistance;

		GArray<Real> force;
		GArray<Real> force_buffer;
		GArray<Coord> velocity_buffer;
		GArray<int> collision_type;
		GArray<Real> volume;
		GArray<Real> volume2;



		GArray<int> sum_nbr;


		x_array.resize(m_neighborhood_tri.getValue().getElementSize());
		y_array.resize(m_neighborhood_tri.getValue().getElementSize());
		interNormal.resize(m_neighborhood_tri.getValue().getElementSize());
		interDistance.resize(m_neighborhood_tri.getValue().getElementSize());
		force.resize(m_neighborhood_tri.getValue().getElementSize());
		force_buffer.resize(m_neighborhood_tri.getValue().getElementSize());
		force.reset();
		force_buffer.reset();


		volume.resize(tetSet->getTetrahedrons().size());
		volume2.resize(tetSet->getTetrahedrons().size());
		volume2.reset();
		velocity_buffer.resize(m_velocity.getElementCount());

		collision_type.resize(m_neighborhood_tri.getValue().getElementSize());
		collision_type.reset();

		sum_nbr.resize(tetSet->getTetrahedrons().size());
		sum_nbr.reset();

		Function1Pt::copy(velocity_buffer, m_velocity.getValue());

		if (m_neighborhood_tri.getValue().getElementSize() > 0)
		{

			cuExecute(total_num_tets, K_SETUP_TET,
				m_tet_vertex.getValue(),
				tetSet->getTetrahedrons(),
				m_neighborhood_tri.getValue(),
				interNormal,
				interDistance,
				x_array,
				y_array,
				sum_nbr,
				volume,
				volume2
			);

			printf("INSEDE TET COLLISION! %d %.10lf\n", m_neighborhood_tri.getValue().getElementSize(), dt);
			for (int i = 0; i < 25; i++)
			{

				cuExecute(m_neighborhood_tri.getValue().getElementSize(), K_ITER_TET,
					x_array,
					y_array,
					volume,
					tetSet->getTetrahedrons(),
					m_neighborhood_tri.getValue(),
					interNormal,
					interDistance,
					m_velocity.getValue(),
					velocity_buffer,
					force_buffer,
					force,
					dt
				);

				//Function1Pt::copy(force_buffer, force);

				cuExecute(m_neighborhood_tri.getValue().getElementSize(), K_UPDATE_TET,
					x_array,
					y_array,
					volume,
					volume2,
					interNormal,
					tetSet->getTetrahedrons(),
					m_velocity.getValue(),
					force,
					sum_nbr,
					dt
				);
			}



		}


		cuExecute(m_velocity.getElementCount(), K_DAMP_TET,
			m_velocity.getValue()
		);


		x_array.clear();
		y_array.clear();

		interNormal.clear();
		interDistance.clear();
		collision_type.clear();
		force.clear();
		force_buffer.clear();
		volume.clear();
		volume2.clear();
		velocity_buffer.clear();
		sum_nbr.clear();
	}


	template<typename TDataType>
	bool TetCollision<TDataType>::initializeImpl()
	{
		//m_tet_vertex_previous.resize(m_tet_vertex.getElementCount());
		//Function1Pt::copy(m_tet_vertex_previous, m_tet_vertex.getValue());

		if (m_position.getElementCount() == 0)
			return true;
		if (m_flip.isEmpty())
		{
			m_flip.setElementCount(m_position.getElementCount());
		}
		posBuf.resize(m_position.getElementCount());
		weights.resize(m_position.getElementCount());
		init_pos.resize(m_position.getElementCount());

		m_position_previous.resize(m_position.getElementCount());
		Function1Pt::copy(m_position_previous, m_position.getValue());



		return true;
	}

}