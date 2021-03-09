#pragma once
#include <cuda_runtime.h>
#include "PointSetToPointSet.h"
#include "Topology/NeighborQuery.h"
#include "Matrix/MatrixFunc.h"

template <typename Real>
DYN_FUNC inline Real PP_Weight(const Real r, const Real h)
{
	const Real q = r / h;
	if (q > 1.0f) return 0.0f;
	else {
		return (1.0f - q * q);
	}
}

namespace dyno
{
	template<typename TDataType>
	PointSetToPointSet<TDataType>::PointSetToPointSet()
		: TopologyMapping()
	{

	}

	template<typename TDataType>
	PointSetToPointSet<TDataType>::PointSetToPointSet(std::shared_ptr<PointSet<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to)
	{
		m_from = from;
		m_to = to;
	}

	template<typename TDataType>
	PointSetToPointSet<TDataType>::~PointSetToPointSet()
	{

	}


	template<typename TDataType>
	bool PointSetToPointSet<TDataType>::initializeImpl()
	{
		match(m_from, m_to);
		return true;
	}

	template <typename Real, typename Coord>
	__global__ void K_ApplyTransform(
		GArray<Coord> to, //new position of surface mesh
		GArray<Coord> from,//inner particle's new position
		GArray<Coord> initTo,  //initial
		GArray<Coord> initFrom,
		NeighborList<int> neighbors,
		Real smoothingLength)  //radius
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= to.size()) return;


		Real totalWeight = 0;
		Coord to_i = to[pId];
		Coord initTo_i = initTo[pId];
		Coord accDisplacement_i = Coord(0);
		int nbSize = neighbors.getNeighborSize(pId);

		Real total_weight1 = 0.0f;
		Matrix3f mat_i = Matrix3f(0);

		Real total_weight2 = 0.0f;
		Matrix3f deform_i = Matrix3f(0.0f);

		for (int ne = 0; ne < nbSize; ne++)
		{

			int j = neighbors.getElement(pId, ne);

			//1
			Real r1 = (initTo_i - initFrom[j]).norm();//j->to

			//2
			Real r2 = (initFrom[j] - initTo_i).norm();//to->j

			/*if (pId == 0)
			{
				printf("initFrom**************************************");

				printf("\n initFrom[j0]: %f %f %f \n from[j0]: %f %f %f \n initFrom: \n %f %f %f \n from: %f %f %f \n initTo: %f %f %f \n\n\n",
					initFrom[j0][0], initFrom[j0][1], initFrom[j0][2],
					from[j][0], from[j][1], from[j][2],
					initFrom[j][0], initFrom[j][1], initFrom[j][2],
					from[j][0], from[j][1], from[j][2],
					initTo_i[0], initTo_i[1], initTo_i[2]);

			}*/

			//1
			if (r1 > EPSILON)
			{
				Real weight1 = PP_Weight(r1, smoothingLength);
				Coord q = (initFrom[j] - initTo_i) / smoothingLength * sqrt(weight1);

				mat_i(0, 0) += q[0] * q[0]; mat_i(0, 1) += q[0] * q[1]; mat_i(0, 2) += q[0] * q[2];
				mat_i(1, 0) += q[1] * q[0]; mat_i(1, 1) += q[1] * q[1]; mat_i(1, 2) += q[1] * q[2];
				mat_i(2, 0) += q[2] * q[0]; mat_i(2, 1) += q[2] * q[1]; mat_i(2, 2) += q[2] * q[2];

				total_weight1 += weight1;
			}
			//


			//2
			if (r2 > EPSILON)
			{
				Real weight2 = PP_Weight(r2, smoothingLength);

				Coord p = (from[j] - to[pId]) / smoothingLength;
				Coord q2 = (initFrom[j] - initTo_i) / smoothingLength * weight2;

				deform_i(0, 0) += p[0] * q2[0]; deform_i(0, 1) += p[0] * q2[1]; deform_i(0, 2) += p[0] * q2[2];
				deform_i(1, 0) += p[1] * q2[0]; deform_i(1, 1) += p[1] * q2[1]; deform_i(1, 2) += p[1] * q2[2];
				deform_i(2, 0) += p[2] * q2[0]; deform_i(2, 1) += p[2] * q2[1]; deform_i(2, 2) += p[2] * q2[2];
				total_weight2 += weight2;
			}
			//
		}

		//1
		if (total_weight1 > EPSILON)
		{
			mat_i *= (1.0f / total_weight1);
		}

		Matrix3f R(0), U(0), D(0), V(0);
		polarDecomposition(mat_i, R, U, D, V);

		Real threshold = 0.0001f*smoothingLength;
		D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
		D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
		D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;

		mat_i = V * D * U.transpose(); //inverse 
		//

		//2
		if (total_weight2 > EPSILON)
		{
			deform_i *= (1.0f / total_weight2);
			deform_i = deform_i * mat_i;//deformation gradient
		}
		else
		{
			total_weight2 = 1.0f;
		}

		//Check whether the reference shape is inverted, if yes, simply set K^{-1} to be an identity matrix
		//Note other solutions are possible.
		//if ((deform_i.determinant()) < -0.001f)
		//{
		//		deform_i = Matrix3f::identityMatrix();
		//	//	printf("**************************************");
		//}

		//get new position
		for (int ne = 0; ne < nbSize; ne++)
		{

			int j = neighbors.getElement(pId, ne);
			Real r = (initFrom[j] - initTo[pId]).norm();

			if (r > 0.01f * smoothingLength)
			{
				Real weight = PP_Weight(r, smoothingLength);

				Coord deformed_ij = deform_i * (initTo[pId] - initFrom[j]);//F*u(ji)
				deformed_ij = deformed_ij.norm() > EPSILON ? deformed_ij.normalize() : Coord(0, 0, 0);

				totalWeight += weight;
				accDisplacement_i += weight * (from[j] + r * deformed_ij);
			}
		}
		accDisplacement_i = totalWeight > EPSILON ? (accDisplacement_i / totalWeight) : accDisplacement_i;
		to[pId] = accDisplacement_i;


	}

	template<typename TDataType>
	bool PointSetToPointSet<TDataType>::apply()
	{
		uint pDim = cudaGridSize(m_to->getPoints().size(), BLOCK_SIZE);

		K_ApplyTransform << <pDim, BLOCK_SIZE >> > (
			m_to->getPoints(),
			m_from->getPoints(),
			m_initTo->getPoints(),
			m_initFrom->getPoints(),
			m_neighborhood,
			m_radius);

		return true;
	}

	template<typename TDataType>
	void PointSetToPointSet<TDataType>::match(std::shared_ptr<PointSet<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to)
	{
		m_initFrom = std::make_shared<PointSet<TDataType>>();
		m_initTo = std::make_shared<PointSet<TDataType>>();

		m_initFrom->copyFrom(*from);
		m_initTo->copyFrom(*to);

		NeighborQuery<TDataType>* nbQuery = new NeighborQuery<TDataType>(m_initFrom->getPoints());

		m_neighborhood.resize(m_initTo->getPoints().size());
		nbQuery->queryParticleNeighbors(m_neighborhood, m_initTo->getPoints(), m_radius);

		delete nbQuery;
	}
}