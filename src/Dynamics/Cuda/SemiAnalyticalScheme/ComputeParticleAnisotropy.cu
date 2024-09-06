 #include "ComputeParticleAnisotropy.h"
#include "Matrix/MatrixFunc.h"

namespace dyno 
{
	template <typename Real>
	DYN_FUNC inline Real iso_Weight(const Real r, const Real h)
	{
		const Real q = r / h;
		if (q >= 1.0f) return 0.0f;
		else {
			return (1.0f - q*q*q);
		}
	}

	template <typename Real, typename Coord, typename Transform>
	__global__ void CalculateTransform(
		DArray<Transform> transform,
		DArray<Coord> pos,
		DArrayList<int> neighbors,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		Real totalWeight = 0;
		Coord pos_i = pos[pId];


		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		Real total_weight = 0.0f;
		Mat3f mat_i = Mat3f(0);
	
		Coord W_pos_i = Coord(0);

		Real total_weight1=0.0f;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - pos[j]).norm();
			Real h = (2.0f)*smoothingLength;
			Real weight = iso_Weight(r, h);
			W_pos_i += pos[j] * weight;// smoothingLength;
			total_weight1 += weight;
		}
		if (total_weight1> EPSILON)
		{
			W_pos_i *= (1.0f / total_weight1);
		}

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - pos[j]).norm();

			if (r > EPSILON)
			{
				Real h = (1.0f)*smoothingLength;
				Real weight = iso_Weight(r, h);
				Coord q = (pos[j] - pos_i)*weight / smoothingLength;

				mat_i(0, 0) += q[0] * q[0]; mat_i(0, 1) += q[0] * q[1]; mat_i(0, 2) += q[0] * q[2];
				mat_i(1, 0) += q[1] * q[0]; mat_i(1, 1) += q[1] * q[1]; mat_i(1, 2) += q[1] * q[2];
				mat_i(2, 0) += q[2] * q[0]; mat_i(2, 1) += q[2] * q[1]; mat_i(2, 2) += q[2] * q[2];

				total_weight += weight;
			}
		}

		if (total_weight > EPSILON)
		{
			mat_i *= (1.0f / total_weight);
		}

		Mat3f R(0), U(0), D(0), V(0);

		polarDecomposition(mat_i, R, U, D, V);

		Real e0 = D(0, 0);
		Real e1 = D(1, 1);
		Real e2 = D(2, 2);

		Real threshold = 0.0001f;
		Real minD = min(e0, min(e1, e2));
		Real maxD = max(e0, max(e1, e2));
		Real maxE = 0.824;
		if (maxD > maxE)
		{
			maxE = maxD;
		}
		if (maxD < threshold)
		{
			D(0, 0) = maxE;
			D(1, 1) = maxE;
			D(2, 2) = maxE;
		}
		else
		{
			D(0, 0) = e0 / maxE;
			D(1, 1) = e1 / maxE;
			D(2, 2) = e2 / maxE;

			if (e1 < threshold)
				D(1, 1) = threshold;
			if (e2 < threshold)
				D(2, 2) = threshold;
		}

		Transform3f tm;
		tm.translation() = pos[pId];
		tm.rotation() = U;
		tm.scale() = 0.02 * Vec3f(D(0, 0), D(1, 1), D(2, 2));
		transform[pId] = tm;
	}

	template<typename TDataType>
	ComputeParticleAnisotropy<TDataType>::ComputeParticleAnisotropy()
		: ComputeModule()
	{
		this->varSmoothingLength()->setValue(Real(0.0125));

	};

	template<typename TDataType>
	ComputeParticleAnisotropy<TDataType>::~ComputeParticleAnisotropy()
	{
	};

	template<typename TDataType>
	void ComputeParticleAnisotropy<TDataType>::compute()
	{
		int num = this->inPosition()->size();

		if (this->outTransform()->size() != num) {
			this->outTransform()->resize(num);
		}

		cuExecute(num,
			CalculateTransform,
			this->outTransform()->getData(),
			this->inPosition()->getData(),
			this->inNeighborIds()->getData(),
			this->varSmoothingLength()->getData());
	};

	DEFINE_CLASS(ComputeParticleAnisotropy);
}