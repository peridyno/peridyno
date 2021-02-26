#include "HyperelasticFractureModule.h"

#include "ParticleSystem/SummationDensity.h"
#include "ParticleSystem/Kernel.h"

#include "Topology/TetrahedronSet.h"
#include "Framework/Node.h"

namespace dyno
{
	template<typename TDataType>
	HyperelasticFractureModule<TDataType>::HyperelasticFractureModule()
		: ConstraintModule()
	{
	}


	template <typename Real, typename Coord, typename NPair>
	__global__ void PM_ComputeInvariants(
		GArray<Real> bulk_stiffiness,
		GArray<Coord> position,
		NeighborList<NPair> restShape,
		Real horizon,
		Real A,
		Real B,
		Real mu,
		Real lambda)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= position.size()) return;

		CorrectedKernel<Real> kernSmooth;

		Real s_A = A;

		Coord rest_pos_i = restShape.getElement(i, 0).pos;
		Coord cur_pos_i = position[i];

		Real I1_i = 0.0f;
		Real J2_i = 0.0f;
		//compute the first and second invariants of the deformation state, i.e., I1 and J2
		int size_i = restShape.getNeighborSize(i);
		Real total_weight = Real(0);
		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = restShape.getElement(i, ne);
			Coord rest_pos_j = np_j.pos;
			int j = np_j.index;
			Real r = (rest_pos_i - rest_pos_j).norm();

			if (r > 0.01*horizon)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				Coord p = (position[j] - cur_pos_i);
				Real ratio_ij = p.norm() / r;

				I1_i += weight*ratio_ij;

				total_weight += weight;
			}
		}

		if (total_weight > EPSILON)
		{
			I1_i /= total_weight;
		}
		else
		{
			I1_i = 1.0f;
		}

		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = restShape.getElement(i, ne);
			int j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_pos_i - rest_pos_j).norm();

			if (r > 0.01*horizon)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				Vector3f p = (position[j] - cur_pos_i);
				Real ratio_ij = p.norm() / r;
				J2_i = (ratio_ij - I1_i)*(ratio_ij - I1_i)*weight;
			}
		}
		if (total_weight > EPSILON)
		{
			J2_i /= total_weight;
			J2_i = sqrt(J2_i);
		}
		else
		{
			J2_i = 0.0f;
		}

		Real D1 = 1 - I1_i;		//positive for compression and negative for stretching

		Real yield_I1_i = 0.0f;
		Real yield_J2_i = 0.0f;

		Real s_J2 = J2_i*mu*bulk_stiffiness[i];
		Real s_D1 = D1*lambda*bulk_stiffiness[i];

		//Drucker-Prager yield criterion
		if (s_J2 > s_A + B*s_D1)
		{
			bulk_stiffiness[i] = 0.0f;
		}
	}

	template<typename TDataType>
	bool HyperelasticFractureModule<TDataType>::constrain()
	{
		updateTopology();
		return true;
	}

	template<typename Real>
	__device__ bool Fracture1D(Real l, Real L, Real crictialStretch)
	{
		if (l / L > crictialStretch)
		{
			return true;
		}

		return false;
	}

	template<typename Coord, typename Tri2Tet, typename Tetrahedron>
	__global__ void HFM_TagFracture(
		GArray<bool> tags,
		GArray<Coord> positions,
		GArray<Coord> restPositions,
		GArray<Tri2Tet> tri2Tets,
		GArray<Tetrahedron> tets,
		Real crictialStretch)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tags.size()) return;

		Tri2Tet neiIds = tri2Tets[tId];

		int tetId0 = neiIds[0];
		int tetId1 = neiIds[1];

		if (tetId0 != EMPTY && tetId1 != EMPTY)
		{
			Tetrahedron tet0 = tets[tetId0];
			Tetrahedron tet1 = tets[tetId1];

			Coord v0 = (positions[tet0[0]] + positions[tet0[1]] + positions[tet0[2]] + positions[tet0[3]]) / 4;
			Coord v1 = (positions[tet1[0]] + positions[tet1[1]] + positions[tet1[2]] + positions[tet1[3]]) / 4;

			Coord v0_rest = (restPositions[tet0[0]] + restPositions[tet0[1]] + restPositions[tet0[2]] + restPositions[tet0[3]]) / 4;
			Coord v1_rest = (restPositions[tet1[0]] + restPositions[tet1[1]] + restPositions[tet1[2]] + restPositions[tet1[3]]) / 4;

			tags[tId] = Fracture1D((v0 - v1).norm(), (v0_rest - v1_rest).norm(), crictialStretch);
		}
	}

	template<typename TDataType>
	void HyperelasticFractureModule<TDataType>::updateTopology()
	{
		Node* p = this->getParent();
		if (p == nullptr) return;

		auto tetSet = TypeInfo::cast<TetrahedronSet<TDataType>>(p->getTopologyModule());
		if (tetSet == nullptr) return;

		auto& tri2Tet = tetSet->getTri2Tet();
		auto& tets = tetSet->getTetrahedrons();

		int tagSize = this->inFractureTag()->getElementCount();

		if (tri2Tet.size() != tagSize) return;

		cuExecute(tagSize,
			HFM_TagFracture,
			this->inFractureTag()->getValue(),
			this->inPosition()->getValue(),
			this->inRestPosition()->getValue(),
			tri2Tet,
			tets,
			this->varCriticalStretch()->getValue());
	}
}