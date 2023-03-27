#include "TetrahedronSetToPointSet.h"

#include "Matrix/MatrixFunc.h"
#include "Topology/NeighborPointQuery.h"

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
	TetrahedronSetToPointSet<TDataType>::TetrahedronSetToPointSet()
		: TopologyMapping()
	{

	}

	template<typename TDataType>
	TetrahedronSetToPointSet<TDataType>::TetrahedronSetToPointSet(std::shared_ptr<TetrahedronSet<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to)
	{
		m_from = from;
		m_to = to;
	}

	template<typename TDataType>
	TetrahedronSetToPointSet<TDataType>::~TetrahedronSetToPointSet()
	{

	}


	template<typename TDataType>
	bool TetrahedronSetToPointSet<TDataType>::initializeImpl()
	{
		match(m_from, m_to);
		return true;
	}

	//TODO: fix the problem
	template <typename Real, typename Coord>
	__global__ void K_ApplyTransform(
		DArray<Coord> to, //new position of surface mesh
		DArray<Coord> from,//inner particle's new position
		DArray<Coord> initTo,  //initial
		DArray<Coord> initFrom,
		DArrayList<int> neighbors,
		Real smoothingLength)  //radius
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= to.size()) return;


		Real totalWeight = 0;
		Coord to_i = to[pId];
		Coord initTo_i = initTo[pId];
		Coord accDisplacement_i = Coord(0);

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();

		Real total_weight1 = 0.0f;
		Mat3f mat_i = Mat3f(0);

		Real total_weight2 = 0.0f;
		Mat3f deform_i = Mat3f(0.0f);

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];

			//1
			Real r1 = (initTo_i - initFrom[j]).norm();//j->to

			//2
			Real r2 = (initFrom[j] - initTo_i).norm();//to->j


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

		Mat3f R(0), U(0), D(0), V(0);
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

		//get new position
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
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
	bool TetrahedronSetToPointSet<TDataType>::apply()
	{
		uint pDim = cudaGridSize(m_to->getPoints().size(), BLOCK_SIZE);

		K_ApplyTransform << <pDim, BLOCK_SIZE >> > (
			m_to->getPoints(),
			m_from->getPoints(),
			m_initTo->getPoints(),
			m_initFrom->getPoints(),
			mNeighborIds,
			m_radius);

		return true;
	}

	template<typename TDataType>
	void TetrahedronSetToPointSet<TDataType>::match(std::shared_ptr<TetrahedronSet<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to)
	{
		m_initFrom = std::make_shared<TetrahedronSet<TDataType>>();
		m_initTo = std::make_shared<PointSet<TDataType>>();

		m_initFrom->copyFrom(*from);
		m_initTo->copyFrom(*to);

		auto nbQuery = std::make_shared<NeighborPointQuery<TDataType>>();

		nbQuery->inRadius()->setValue(m_radius);
		nbQuery->inPosition()->allocate()->assign(m_initFrom->getPoints());
		nbQuery->inOther()->allocate()->assign(m_initTo->getPoints());

		nbQuery->update();

		mNeighborIds.assign(nbQuery->outNeighborIds()->getData());
	}

	DEFINE_CLASS(TetrahedronSetToPointSet);
}