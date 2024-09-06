#include <cuda_runtime.h>
#include "SimpleVelocityConstraint.h"
#include <string>
#include "Algorithm/Function2Pt.h"


namespace dyno
{
	//F-norms
	template<typename Real>
	__device__ Real VB_FNorm(const Real a, const Real b, const Real c)
	{
		Real p = a + b + c;
		return sqrt(p * p / 2);
	}

	//CrossModel Viscosity Coefficient
	template<typename Real>
	__device__ Real VB_Viscosity(const Real Viscosity_h, const Real Viscosity_l, const Real StrainRate, const Real CrossModel_K, const Real CrossModel_n)
	{
		Real p = CrossModel_K * StrainRate;
		p = pow(p, CrossModel_n);
		return Viscosity_h + (Viscosity_l - Viscosity_h) / (1 + p);
	}

	__device__ inline float kernWeight(const float r, const float h)
	{
		const float q = r / h;
		if (q > 1.0f) return 0.0f;
		else {
			const float d = 1.0f - q;
			const float hh = h * h;
			return (1.0 - pow(q, 4.0f));
		}
	}


	__device__ inline float kernWR(const float r, const float h)
	{
		float w = kernWeight(r, h);
		const float q = r / h;
		if (q < 0.4f)
		{
			return w / (0.4f * h);
		}
		return w / r;
	}

	__device__ inline float kernWRR(const float r, const float h)
	{
		float w = kernWeight(r, h);
		const float q = r / h;
		if (q < 0.4f)
		{
			return w / (0.16f * h * h);
		}
		return w / r / r;
	}


	template <typename Real, typename Coord>
	__global__ void SIMPLE_ComputeAlpha
	(
		DArray<Real> alpha,
		DArray<Coord> position,
		DArray<Attribute> attribute,
		DArrayList<int> neighbors,
		Real smoothingLength
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isFluid()) return;

		Coord pos_i = position[pId];
		Real alpha_i = 0.0f;

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - position[j]).norm();;

			if (r > EPSILON)
			{
				Real a_ij = kernWeight(r, smoothingLength);
				alpha_i += a_ij;
			}
		}

		alpha[pId] = alpha_i;
	}

	template <typename Real>
	__global__ void SIMPLE_CorrectAlpha
	(
		DArray<Real> alpha,
		Real maxAlpha)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= alpha.size()) return;

		Real alpha_i = alpha[pId];
		if (alpha_i < maxAlpha)
		{
			alpha_i = maxAlpha;
		}
		alpha[pId] = alpha_i;
	}

	template <typename Real, typename Coord>
	__global__ void SIMPLE_ComputeDiagonalElement
	(
		DArray<Real> diaA,
		DArray<Real> alpha,
		DArray<Coord> position,
		DArray<Attribute> attribute,
		DArrayList<int> neighbors,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isFluid()) return;

		Coord pos_i = position[pId];
		Real invAlpha_i = 1.0f / alpha[pId];
		Real A_i = 0.0f;

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - position[j]).norm();

			if (r > EPSILON && attribute[j].isFluid())
			{
				Real wrr_ij = invAlpha_i * kernWRR(r, smoothingLength);
				A_i += wrr_ij;
				atomicAdd(&diaA[j], wrr_ij);
			}
		}

		atomicAdd(&diaA[pId], A_i);
	}

	template <typename Real, typename Coord>
	__global__ void SIMPLE_ComputeDiagonalElement
	(
		DArray<Real> AiiFluid,
		DArray<Real> AiiTotal,
		DArray<Real> alpha,
		DArray<Coord> position,
		DArray<Attribute> attribute,
		DArrayList<int> neighbors,
		Real smoothingLength
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isFluid()) return;

		Real invAlpha = 1.0f / alpha[pId];


		Real diaA_total = 0.0f;
		Real diaA_fluid = 0.0f;
		Coord pos_i = position[pId];

		bool bNearWall = false;

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			//int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - position[j]).norm();

			Attribute att_j = attribute[j];


			if (r > EPSILON)
			{
				Real wrr_ij = invAlpha * kernWRR(r, smoothingLength);
				if (att_j.isFluid())
				{
					diaA_total += wrr_ij;
					diaA_fluid += wrr_ij;
					atomicAdd(&AiiFluid[j], wrr_ij);
					atomicAdd(&AiiTotal[j], wrr_ij);
				}
				else
				{
					diaA_total += 2.0f * wrr_ij;
				}

			}

		}
		atomicAdd(&AiiFluid[pId], diaA_fluid);
		atomicAdd(&AiiTotal[pId], diaA_total);


	}

	template <typename Real, typename Coord>
	__global__ void SIMPLE_DetectSurface
	(
		DArray<Real> Aii,
		DArray<bool> bSurface,
		DArray<Real> AiiFluid,
		DArray<Real> AiiTotal,
		DArray<Coord> position,
		DArray<Attribute> attribute,
		DArrayList<int> neighbors,
		Real smoothingLength,
		Real maxA
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isFluid()) return;

		Real total_weight = 0.0f;
		Coord div_i = Coord(0);

		SmoothKernel<Real> kernSmooth;

		Coord pos_i = position[pId];

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		bool bNearWall = false;
		for (int ne = 0; ne < nbSize; ne++)
		{

			int j = list_i[ne];
			Real r = (pos_i - position[j]).norm();

			if (r > EPSILON && attribute[j].isFluid())
			{
				float weight = -kernSmooth.Gradient(r, smoothingLength);
				total_weight += weight;
				div_i += (position[j] - pos_i) * (weight / r);
			}

			if (!attribute[j].isFluid())
			{
				bNearWall = true;
			}
		}

		total_weight = total_weight < EPSILON ? 1.0f : total_weight;
		Real absDiv = div_i.norm() / total_weight;

		bool bSurface_i = false;
		Real diagF_i = AiiFluid[pId];
		Real diagT_i = AiiTotal[pId];
		Real aii = diagF_i;
		Real eps = 0.001f;
		Real diagS_i = diagT_i - diagF_i;
		Real threshold = 0.0f;
		if (bNearWall && diagT_i < maxA * (1.0f - threshold))
		{
			bSurface_i = true;
			aii = maxA - (diagT_i - diagF_i);
		}

		if (!bNearWall && diagF_i < maxA * (1.0f - threshold))
		{
			bSurface_i = true;
			aii = maxA;
		}
		bSurface[pId] = bSurface_i;
		Aii[pId] = aii;
	}


	template <typename Real, typename Coord>
	__global__ void SIMPLE_ComputeDivergence
	(
		DArray<Real> divergence,
		DArray<Real> alpha,
		DArray<Coord> position,
		DArray<Coord> velocity,
		DArray<Coord> normals,
		DArray<Attribute> attribute,
		DArrayList<int> neighbors,
		Real separation,
		Real tangential,
		Real restDensity,
		Real smoothingLength,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isFluid()) return;
		Coord pos_i = position[pId];
		Coord vel_i = velocity[pId];

		Real div_vi = 0.0f;

		Real invAlpha_i = 1.0f / alpha[pId];

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];

			Real r = (pos_i - position[j]).norm();

			if (r > EPSILON && attribute[j].isFluid())
			{
				Real wr_ij = kernWR(r, smoothingLength);
				Coord g = -invAlpha_i * (pos_i - position[j]) * wr_ij * (1.0f / r);

				if (attribute[j].isFluid())
				{
					Real div_ij = 0.5f * (vel_i - velocity[j]).dot(g) * restDensity / dt;	//dv_ij = 1 / alpha_i * (v_i-v_j).*(x_i-x_j) / r * (w / r);
					atomicAdd(&divergence[pId], div_ij);
					atomicAdd(&divergence[j], div_ij);
				}
				else
				{

					Coord normal_j = normals[j];

					Coord dVel = vel_i - velocity[j];
					Real magNVel = dVel.dot(normal_j);
					Coord nVel = magNVel * normal_j;
					Coord tVel = dVel - nVel;


					if (magNVel < -EPSILON)
					{
						Real div_ij = g.dot(2.0f * (nVel + tangential * tVel)) * restDensity / dt;
						//						printf("Boundary div: %f \n", div_ij);
						atomicAdd(&divergence[pId], div_ij);
					}
					else
					{
						Real div_ij = g.dot(2.0f * (separation * nVel + tangential * tVel)) * restDensity / dt;
						atomicAdd(&divergence[pId], div_ij);
					}

				}
			}
		}
	}

	template <typename Real, typename Coord>
	__global__ void SIMPLE_CompensateSource
	(
		DArray<Real> divergence,
		DArray<Real> density,
		DArray<Attribute> attribute,
		DArray<Coord> position,
		Real restDensity,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= density.size()) return;
		if (!attribute[pId].isFluid()) return;

		Coord pos_i = position[pId];
		if (density[pId] > restDensity)
		{
			Real ratio = (density[pId] - restDensity) / restDensity;
			atomicAdd(&divergence[pId], 100000.0f * ratio / dt);
		}
	}


	// compute Ax;
	template <typename Real, typename Coord>
	__global__ void SIMPLE_ComputeAx
	(
		DArray<Real> residual,
		DArray<Real> pressure,
		DArray<Real> aiiSymArr,
		DArray<Real> alpha,
		DArray<Coord> position,
		DArray<Attribute> attribute,
		DArrayList<int> neighbor,
		Real smoothingLength
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isFluid()) return;

		Coord pos_i = position[pId];
		Real invAlpha_i = 1.0f / alpha[pId];

		atomicAdd(&residual[pId], aiiSymArr[pId] * pressure[pId]);
		Real con1 = 1.0f;// PARAMS.mass / PARAMS.restDensity / PARAMS.restDensity;

		//int nbSize = neighbor.getNeighborSize(pId);
		List<int>& list_i = neighbor[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			//int j = neighbor.getElement(pId, ne);

			Real r = (pos_i - position[j]).norm();

			if (r > EPSILON && attribute[j].isFluid())
			{
				Real wrr_ij = kernWRR(r, smoothingLength);
				Real a_ij = -invAlpha_i * wrr_ij;
				//				residual += con1*a_ij*preArr[j];
				atomicAdd(&residual[pId], con1 * a_ij * pressure[j]);
				atomicAdd(&residual[j], con1 * a_ij * pressure[pId]);
			}
		}
	}


	template <typename Real, typename Coord>
	__global__ void SIMPLE_P_dv(
		DArray<Coord> P_dv,
		DArray<Real> pressure,
		DArray<Real> alpha,
		DArray<bool> bSurface,
		DArray<Coord> position,
		DArray<Coord> velocity,
		DArray<Coord> normal,
		DArray<Attribute> attribute,
		DArrayList<int> neighbor,
		Real restDensity,
		Real airPressure,
		Real sliding,
		Real separation,
		Real smoothingLength,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		if (attribute[pId].isFluid())
		{
			Coord pos_i = position[pId];
			Real p_i = pressure[pId];

			//int nbSize = neighbor.getNeighborSize(pId);

			Real total_weight = 0.0f;

			Real ceo = 1.6f;

			Real invAlpha = 1.0f / alpha[pId];
			Coord vel_i = velocity[pId];
			Coord dv_i(0.0f);
			Real scale = 1.0f * dt / restDensity;
			Real acuP = 0.0f;
			total_weight = 0.0f;


			List<int>& list_i = neighbor[pId];
			int nbSize = list_i.size();
			for (int ne = 0; ne < nbSize; ne++)
			{
				//int j = neighbor.getElement(pId, ne);
				int j = list_i[ne];
				Real r = (pos_i - position[j]).norm();

				Attribute att_j = attribute[j];
				if (r > EPSILON)
				{
					Real weight = -invAlpha * kernWR(r, smoothingLength);
					Coord dnij = (pos_i - position[j]) * (1.0f / r);
					Coord corrected = dnij;
					if (corrected.norm() > EPSILON)
					{
						corrected = corrected.normalize();
					}
					corrected = -scale * weight * corrected;

					Coord dvij = (pressure[j] - pressure[pId]) * corrected;
					Coord dvjj = (pressure[j] + airPressure) * corrected;
					Coord dvij_sym = 0.5f * (pressure[pId] + pressure[j]) * corrected;


					if (att_j.isFluid())
					{
						if (bSurface[pId])
						{
							dv_i += dvjj;
						}
						else
						{
							dv_i += dvij;
						}

						if (bSurface[j])
						{
							Coord dvii = -(pressure[pId] + airPressure) * corrected;
							atomicAdd(&P_dv[j][0], ceo * dvii[0]);
							atomicAdd(&P_dv[j][1], ceo * dvii[1]);
							atomicAdd(&P_dv[j][2], ceo * dvii[2]);
						}
						else
						{

							atomicAdd(&P_dv[j][0], ceo * dvij[0]);
							atomicAdd(&P_dv[j][1], ceo * dvij[1]);
							atomicAdd(&P_dv[j][2], ceo * dvij[2]);
						}
					}
					else
					{
						Coord dvii = 2.0f * (pressure[pId]) * corrected;
						if (bSurface[pId])
						{
							dv_i += dvii;
						}

						float weight = 2.0f * invAlpha * kernWeight(r, smoothingLength);
						Coord nij = (pos_i - position[j]);
						if (nij.norm() > EPSILON)
						{
							nij = nij.normalize();
						}
						else
							nij = Coord(1.0f, 0.0f, 0.0f);

						Coord normal_j = normal[j];
						Coord dVel = velocity[j] - vel_i;
						Real magNVel = dVel.dot(normal_j);
						Coord nVel = magNVel * normal_j;
						Coord tVel = dVel - nVel;
						if (magNVel > EPSILON)
						{
							dv_i += weight * nij.dot(nVel + sliding * tVel) * nij;
						}
						else
						{
							dv_i += weight * nij.dot(separation * nVel + sliding * tVel) * nij;
						}
					}
				}
			}
			dv_i *= ceo;
			atomicAdd(&P_dv[pId][0], dv_i[0]);
			atomicAdd(&P_dv[pId][1], dv_i[1]);
			atomicAdd(&P_dv[pId][2], dv_i[2]);
		}
	}



	template <typename Real>
	__global__ void UpdatePressure(
		DArray<Real> totalPressure,
		DArray<Real> deltaPressure)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= totalPressure.size()) return;

		totalPressure[pId] += deltaPressure[pId];
	}


	template <typename Coord>
	__global__ void SIMPLE_VelUpdate(
		DArray<Coord> velocity,
		DArray<Coord> vel_old,
		DArray<Coord> P_dv
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velocity.size()) return;
		velocity[pId] = vel_old[pId] + P_dv[pId];
	}


	template <typename Real, typename Coord, typename Matrix>
	__global__ void SIMPLE_VisComput
	(
		DArray<Coord> velNew,
		DArray<Coord> velBuf,
		DArray<Coord> velOld,
		DArray<Coord> velDp,
		DArray<Coord> position,
		DArray<Real> alpha,
		DArray<Attribute> attribute,
		DArrayList<int> neighbor,
		Real rest_density,
		Real h,
		Real dt,
		DArray<Real> vis
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isFluid()) return;

		Real tempValue = 0;
		Coord Avj(0);
		Matrix Mii(0);
		Real invAlpha_i = 1.0f / alpha[pId];
		//int nbSize = neighbor.getNeighborSize(pId);
		List<int>& list_i = neighbor[pId];
		int nbSize = list_i.size();

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			//int j = neighbor.getElement(pId, ne);
			Real r = (position[pId] - position[j]).norm();
			if (r > EPSILON)
			{
				Real wrr_ij = kernWRR(r, h);
				Real ai = 1.0f / alpha[pId];
				Real aj = 1.0f / alpha[j];
				Coord nij = (position[j] - position[pId]) / r;
				tempValue = 0.25 * dt * (vis[pId] + vis[j]) * (ai + aj) * wrr_ij / rest_density;
				Avj += tempValue * velBuf[j].dot(nij) * nij;
				Matrix Mij(0);
				Mij(0, 0) += nij[0] * nij[0];	Mij(0, 1) += nij[0] * nij[1];	Mij(0, 2) += nij[0] * nij[2];
				Mij(1, 0) += nij[1] * nij[0];	Mij(1, 1) += nij[1] * nij[1];	Mij(1, 2) += nij[1] * nij[2];
				Mij(2, 0) += nij[2] * nij[0];	Mij(2, 1) += nij[2] * nij[1];	Mij(2, 2) += nij[2] * nij[2];
				Mii += Mij * tempValue;
			}
		}
		Mii += Matrix::identityMatrix();
		velNew[pId] = Mii.inverse() * (velOld[pId] + velDp[pId] + Avj);
	}

	template <typename Real, typename Coord>
	__global__ void SIMPLE_Vis_AxComput
	(
		DArray<Real> v_y,
		DArray<Coord> velBuf,
		DArray<Coord> position,
		DArray<Real> alpha,
		DArray<Attribute> attribute,
		DArrayList<int> neighbor,
		Real rest_density,
		Real h,
		Real dt,
		DArray<Real> vis
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isFluid()) return;

		Real tempValue = 0;
		Coord Avi(0);
		Real invAlpha_i = 1.0f / alpha[pId];
		//int nbSize = neighbor.getNeighborSize(pId);
		List<int>& list_i = neighbor[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			//int j = neighbor.getElement(pId, ne);
			Real r = (position[pId] - position[j]).norm();
			if (r > EPSILON)
			{
				Real wrr_ij = kernWRR(r, h);
				Real ai = 1.0f / alpha[pId];
				Real aj = 1.0f / alpha[j];
				Coord nij = (position[j] - position[pId]) / r;
				Coord vij = velBuf[pId] - velBuf[j];
				tempValue = 2.5 * dt * (vis[pId] + vis[j]) * (ai + aj) * wrr_ij / rest_density;
				Avi += tempValue * vij.dot(nij) * nij;
			}
		}
		Avi += velBuf[pId];
		v_y[3 * pId] = Avi[0];
		v_y[3 * pId + 1] = Avi[1];
		v_y[3 * pId + 2] = Avi[2];


	}


	template <typename Real, typename Coord>
	__global__ void SIMPLE_Vis_r_Comput
	(
		DArray<Real> v_r,
		DArray<Real> v_y,
		DArray<Coord> vel_old,
		DArray<Coord> Dvel,
		DArray<Coord> position,
		DArray<Attribute> attribute
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isFluid()) return;
		Coord temp_vel = Dvel[pId] + vel_old[pId];
		v_r[3 * pId] = temp_vel[0] - v_y[3 * pId];
		v_r[3 * pId + 1] = temp_vel[1] - v_y[3 * pId + 1];
		v_r[3 * pId + 2] = temp_vel[2] - v_y[3 * pId + 2];
	}

	template <typename Real, typename Coord>
	__global__ void SIMPLE_Vis_pToVector
	(
		DArray<Real> v_p,
		DArray<Coord> pv,
		DArray<Attribute> attribute
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= attribute.size()) return;
		if (!attribute[pId].isFluid()) return;
		pv[pId][0] = v_p[3 * pId];
		pv[pId][1] = v_p[3 * pId + 1];
		pv[pId][2] = v_p[3 * pId + 2];
	}


	template <typename Real, typename Coord>
	__global__ void SIMPLE_Vis_CoordToReal
	(
		DArray<Real> veloReal,
		DArray<Coord> vel,
		DArray<Attribute> attribute
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= attribute.size()) return;
		if (!attribute[pId].isFluid()) return;
		veloReal[3 * pId] = vel[pId][0];
		veloReal[3 * pId + 1] = vel[pId][1];
		veloReal[3 * pId + 2] = vel[pId][2];
	}

	template <typename Real, typename Coord>
	__global__ void SIMPLE_Vis_RealToVeloctiy
	(
		DArray<Coord> vel,
		DArray<Real> veloReal,
		DArray<Attribute> attribute
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= attribute.size()) return;
		if (!attribute[pId].isFluid()) return;
		vel[pId][0] = veloReal[3 * pId];
		vel[pId][1] = veloReal[3 * pId + 1];
		vel[pId][2] = veloReal[3 * pId + 2];
	}

	template <typename Real, typename Coord>
	__global__ void SIMPLE_CrossVis
	(
		DArray<Real> crossVis,
		DArray<Coord> velBuf,
		DArray<Coord> position,
		DArray<Real> alpha,
		DArray<Attribute> attribute,
		DArrayList<int> neighbor,
		Real smoothingLength,
		Real visTop,
		Real visFloor,
		Real K,
		Real N
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isFluid()) return;

		Coord pos_i = position[pId];
		Coord vel_i = velBuf[pId];
		Real invAlpha_i = 1.0f / alpha[pId];

		Coord dv(0);
		Coord vij(0);

		List<int>& list_i = neighbor[pId];
		int nbSize = list_i.size();
		//int nbSize = neighbor.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			//int j = neighbor.getElement(pId, ne);
			int j = list_i[ne];
			Real r = (position[pId] - position[j]).norm();
			if (r > EPSILON && attribute[j].isFluid())
			{
				Real wr_ij = kernWR(r, smoothingLength);
				Coord g = -invAlpha_i * (pos_i - position[j]) * wr_ij * (1.0f / r);
				vij = 0.5 * (velBuf[pId] - velBuf[j]);
				dv[0] += vij[0] * g[0];
				dv[1] += vij[1] * g[1];
				dv[2] += vij[2] * g[2];
			}

		}
		Real Norm = VB_FNorm(dv[0], dv[1], dv[2]);
		crossVis[pId] = VB_Viscosity(visTop, visFloor, Norm, K, N);
		if (pId == 100) printf("viscosity : %f. \r\n", crossVis[pId]);
	}


	__global__ void SIMPLE_AttributeInit
	(
		DArray<Attribute> atts
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= atts.size()) return;

		atts[pId].setFluid();
		atts[pId].setDynamic();
	}


	template<typename TDataType>
	SimpleVelocityConstraint<TDataType>::SimpleVelocityConstraint()
		: ConstraintModule()
		, m_airPressure(Real(0))
		, m_reduce(NULL)
		, m_arithmetic(NULL)
	{
		this->inSmoothingLength()->setValue(Real(0.0125));
		this->varRestDensity()->setValue(Real(1000));

		m_densitySum = std::make_shared<SummationDensity<TDataType>>();
		this->varRestDensity()->connect(m_densitySum->varRestDensity());
		this->inSmoothingLength()->connect(m_densitySum->inSmoothingLength());
		this->inSamplingDistance()->connect(m_densitySum->inSamplingDistance());

		this->inPosition()->connect(m_densitySum->inPosition());
		this->inNeighborIds()->connect(m_densitySum->inNeighborIds());
		SIMPLE_IterNum = 5;

		this->inAttribute()->tagOptional(true);
		this->inNormal()->tagOptional(true);
	};

	template<typename TDataType>
	SimpleVelocityConstraint<TDataType>::~SimpleVelocityConstraint()
	{
		m_alpha.clear();
		m_Aii.clear();
		m_AiiFluid.clear();
		m_AiiTotal.clear();
		m_pressure.clear();
		m_divergence.clear();
		m_bSurface.clear();

		m_y.clear();
		m_r.clear();
		m_p.clear();

		m_pressure.clear();

		if (m_reduce)
		{
			delete m_reduce;
		}
		if (m_arithmetic)
		{
			delete m_arithmetic;
		}
	};


	template<typename TDataType>
	void SimpleVelocityConstraint<TDataType>::constrain()
	{

		if ((init_flag == false) || (this->inTimeStep()->getValue() == 0))
		{
			initialize();
		}

		if (this->inPosition()->size() != m_viscosity.size())
		{
			resizeVector();
		}

		cuSynchronize();

		auto& m_position = this->inPosition()->getData();
		auto& m_velocity = this->inVelocity()->getData();
		auto& m_neighborhood = this->inNeighborIds()->getData();
		auto& m_attribute = this->inAttribute()->getData();
		auto& m_normal = this->inNormal()->getData();
		auto m_smoothingLength = this->inSmoothingLength()->getValue();
		auto m_restDensity = this->varRestDensity()->getValue();

		//Real dt = getParent()->getDt();
		Real dt = this->inTimeStep()->getData();

		int num = this->inPosition()->size();
		uint pDims = cudaGridSize(this->inPosition()->size(), BLOCK_SIZE);


		m_alpha.reset();
		//compute alpha_i = sigma w_j and A_i = sigma w_ij / r_ij / r_ij
		SIMPLE_ComputeAlpha << <pDims, BLOCK_SIZE >> > (
			m_alpha,
			m_position,
			m_attribute,
			m_neighborhood,
			m_smoothingLength);

		SIMPLE_CorrectAlpha << <pDims, BLOCK_SIZE >> > (
			m_alpha,
			m_maxAlpha);


		//compute the diagonal elements of the coefficient matrix
		m_AiiFluid.reset();
		m_AiiTotal.reset();

		SIMPLE_ComputeDiagonalElement << <pDims, BLOCK_SIZE >> > (
			m_AiiFluid,
			m_AiiTotal,
			m_alpha,
			m_position,
			m_attribute,
			m_neighborhood,
			m_smoothingLength);

		m_bSurface.reset();
		m_Aii.reset();
		SIMPLE_DetectSurface << <pDims, BLOCK_SIZE >> > (
			m_Aii,
			m_bSurface,
			m_AiiFluid,
			m_AiiTotal,
			m_position,
			m_attribute,
			m_neighborhood,
			m_smoothingLength,
			m_maxA);


		m_densitySum->compute();
		m_densitySum->outDensity()->getData();

		//IF Cross Model is active, viscous coefficients should be computed.
		if (IsCrossReady)
		{

			SIMPLE_CrossVis << <pDims, BLOCK_SIZE >> > (
				m_viscosity,
				m_velocity,
				m_position,
				m_alpha,
				m_attribute,
				m_neighborhood,
				m_smoothingLength,
				CrossVisCeil,
				CrossVisFloor,
				Cross_K,
				Cross_N
				);
		}


		int totalIter = 0;

		m_pressure.reset();

		velOld.assign(m_velocity);

		Real Old_temp = 0;

		if (this->varSimpleIterationEnable()->getValue() == false)
		{
			SIMPLE_IterNum = 1;
		}

		//SIMPLE Algorithm / Outer Iterations
		while (totalIter < SIMPLE_IterNum)
		{
			printf("Iteration : %d *****", totalIter);
			totalIter++;

			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//Incopressibility Solver -- Begin ¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			m_divergence.reset();
			SIMPLE_ComputeDivergence << <pDims, BLOCK_SIZE >> > (
				m_divergence,
				m_alpha,
				m_position,
				m_velocity,
				m_normal,
				m_attribute,
				m_neighborhood,
				m_separation,
				m_tangential,
				m_restDensity,
				m_smoothingLength,
				dt);

			SIMPLE_CompensateSource << <pDims, BLOCK_SIZE >> > (
				m_divergence,
				m_densitySum->outDensity()->getData(),
				m_attribute,
				m_position,
				m_restDensity,
				dt);

			m_deltaPressure.reset();
			m_y.reset();
			SIMPLE_ComputeAx << <pDims, BLOCK_SIZE >> > (
				m_y,
				m_deltaPressure,
				m_Aii,
				m_alpha,
				m_position,
				m_attribute,
				m_neighborhood,
				m_smoothingLength);

			m_r.reset();
			Function2Pt::subtract(m_r, m_divergence, m_y);

			m_p.assign(m_r);
			Real rr = m_arithmetic->Dot(m_r, m_r);
			Real err = sqrt(rr / m_r.size());

			Real initErr = err;

			int itor = 0;
			while (itor < 1000 && err / initErr > 0.00001f)
			{
				m_y.reset();
				SIMPLE_ComputeAx << <pDims, BLOCK_SIZE >> > (
					m_y,
					m_p,
					m_Aii,
					m_alpha,
					m_position,
					m_attribute,
					m_neighborhood,
					m_smoothingLength);

				float alpha = rr / m_arithmetic->Dot(m_p, m_y);
				Function2Pt::saxpy(m_deltaPressure, m_p, m_deltaPressure, alpha);
				Function2Pt::saxpy(m_r, m_y, m_r, -alpha);

				Real rr_old = rr;

				rr = m_arithmetic->Dot(m_r, m_r);

				Real beta = rr / rr_old;
				Function2Pt::saxpy(m_p, m_p, m_r, beta);

				err = sqrt(rr / m_r.size());
				itor++;
			}

			UpdatePressure << <pDims, BLOCK_SIZE >> > (
				m_pressure,
				m_deltaPressure);

			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//Incopressibility Solver -- End ¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



			P_dv.reset();
			SIMPLE_P_dv << <pDims, BLOCK_SIZE >> > (
				P_dv,
				m_pressure,
				m_alpha,
				m_bSurface,
				m_position,
				m_velocity,
				m_normal,
				m_attribute,
				m_neighborhood,
				m_restDensity,
				m_airPressure,
				m_tangential,
				m_separation,
				m_smoothingLength,
				dt);

			SIMPLE_VelUpdate << <pDims, BLOCK_SIZE >> > (
				m_velocity,
				velOld,
				P_dv
				);


			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//Viscosity Solver -- Begin ¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý¡ý
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			v_y.reset();
			SIMPLE_Vis_AxComput<Real, Coord> << <pDims, BLOCK_SIZE >> > (
				v_y,
				m_velocity,
				m_position,
				m_alpha,
				m_attribute,
				m_neighborhood,
				m_restDensity,
				m_smoothingLength,
				dt,
				m_viscosity
				);

			v_r.reset();
			SIMPLE_Vis_r_Comput << <pDims, BLOCK_SIZE >> > (
				v_r,
				v_y,
				velOld,
				P_dv,
				m_position,
				m_attribute
				);
			v_p.assign(v_r);


			Real Vrr = m_arithmetic_v->Dot(v_r, v_r);
			Real Verr = sqrt(Vrr / v_r.size());
			int VisItor = 0;
			initErr = Verr;

			while (VisItor < 1000 && Verr / initErr > 0.00001f)
			{
				VisItor++;
				//The type of "v_p" should convert to DArray<Coord>
				SIMPLE_Vis_pToVector<Real, Coord> << <pDims, BLOCK_SIZE >> > (
					v_p,
					v_pv,
					m_attribute
					);

				v_y.reset();
				SIMPLE_Vis_AxComput<Real, Coord> << <pDims, BLOCK_SIZE >> > (
					v_y,
					v_pv,
					//m_velocity.getValue(),
					m_position,
					m_alpha,
					m_attribute,
					m_neighborhood,
					m_restDensity,
					m_smoothingLength,
					dt,
					m_viscosity
					);

				float alpha = Vrr / m_arithmetic_v->Dot(v_p, v_y);

				//The type of "velocity" should convert to DArray<Real>
				SIMPLE_Vis_CoordToReal<Real, Coord> << <pDims, BLOCK_SIZE >> > (
					m_VelocityReal,
					m_velocity,
					m_attribute
					);

				Function2Pt::saxpy(m_VelocityReal, v_p, m_VelocityReal, alpha);

				SIMPLE_Vis_RealToVeloctiy<Real, Coord> << <pDims, BLOCK_SIZE >> > (
					m_velocity,
					m_VelocityReal,
					m_attribute
					);

				Function2Pt::saxpy(v_r, v_y, v_r, -alpha);
				Real Vrr_old = Vrr;

				Vrr = m_arithmetic_v->Dot(v_r, v_r);
				Real beta = Vrr / Vrr_old;
				Function2Pt::saxpy(v_p, v_p, v_r, beta);

				Verr = sqrt(Vrr / v_r.size());

			}
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//Viscosity Solver -- End ¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü¡ü
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			Real ppp = m_arithmetic->Dot(m_pressure, m_pressure);
			ppp = sqrt(ppp / m_pressure.size());
			Real testP = (ppp - Old_temp) ;
			printf("%f  \r\n", testP);
			Old_temp = ppp;
		}
	};

	template<typename TDataType>
	void SimpleVelocityConstraint<TDataType>::initialAttributes() {

		if (this->inAttribute()->isEmpty() || this->inAttribute()->size() != this->inPosition()->size())
		{
			this->inAttribute()->allocate();
			this->inAttribute()->resize(this->inPosition()->size());
			this->inNormal()->allocate();
			this->inNormal()->resize(this->inPosition()->size());
			this->inNormal()->getData().reset();
			cuExecute(this->inPosition()->size(), SIMPLE_AttributeInit, this->inAttribute()->getData());
		}
	}


	template<typename TDataType>
	bool SimpleVelocityConstraint<TDataType>::resizeVector()
	{

		int num = this->inPosition()->size();
		m_alpha.resize(num);
		m_Aii.resize(num);
		m_AiiFluid.resize(num);
		m_AiiTotal.resize(num);
		m_pressure.resize(num);
		m_divergence.resize(num);
		m_bSurface.resize(num);


		m_y.resize(num);
		m_r.resize(num);
		m_p.resize(num);

		v_y.resize(3 * num);
		v_r.resize(3 * num);
		v_p.resize(3 * num);
		v_pv.resize(num);
		m_VelocityReal.resize(3 * num);

		m_pressure.resize(num);

		P_dv.resize(num);
		velOld.resize(num);
		velBuf.resize(num);
		m_viscosity.resize(num);
		m_deltaPressure.resize(num);
		m_pressBuf.resize(num);
		m_crossViscosity.resize(num);


		m_reduce = Reduction<float>::Create(num);
		m_arithmetic = Arithmetic<float>::Create(num);
		m_arithmetic_v = Arithmetic<float>::Create(3 * num);

		visValueSet();

		initialAttributes();

		return true;
	}



	template<typename TDataType>
	bool SimpleVelocityConstraint<TDataType>::initialize()
	{
		cuSynchronize();
		init_flag = true;

		initialAttributes();

		auto& m_position = this->inPosition()->getData();
		auto& m_velocity = this->inVelocity()->getData();
		auto& m_neighborhood = this->inNeighborIds()->getData();
		auto& m_attribute = this->inAttribute()->getData();
		auto m_smoothingLength = this->inSmoothingLength()->getValue();
		auto m_restDensity = this->varRestDensity()->getValue();
		auto& m_normal = this->inNormal()->getData();

		resizeVector();


		uint pDims = cudaGridSize(this->inPosition()->size(), BLOCK_SIZE);

		m_alpha.reset();

		SIMPLE_ComputeAlpha << <pDims, BLOCK_SIZE >> > (
			m_alpha,
			m_position,
			m_attribute,
			m_neighborhood,
			m_smoothingLength);

		m_maxAlpha = m_reduce->maximum(m_alpha.begin(), m_alpha.size());

		SIMPLE_CorrectAlpha << <pDims, BLOCK_SIZE >> > (
			m_alpha,
			m_maxAlpha);

		m_AiiFluid.reset();
		SIMPLE_ComputeDiagonalElement << <pDims, BLOCK_SIZE >> > (
			m_AiiFluid,
			m_alpha,
			m_position,
			m_attribute,
			m_neighborhood,
			m_smoothingLength);

		m_maxA = m_reduce->maximum(m_AiiFluid.begin(), m_AiiFluid.size());



		std::cout << "Max alpha: " << m_maxAlpha << std::endl;
		std::cout << "Max A: " << m_maxA << std::endl;

		return true;
	};


	DEFINE_CLASS(SimpleVelocityConstraint);

}