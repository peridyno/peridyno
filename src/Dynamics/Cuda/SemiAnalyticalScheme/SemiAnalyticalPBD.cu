/**
 * @author     : Yue Chang (yuechang@pku.edu.cn)
 * @date       : 2021-08-04
 * @description: Implementation of DensityPBDMesh class, which implements the position-based part of semi-analytical boundary conditions
 *               introduced in the paper <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
 * @version    : 1.1
 */
#include "SemiAnalyticalPBD.h"
#include "SemiAnalyticalSummationDensity.h"
#include "IntersectionArea.h"

namespace dyno
{
	IMPLEMENT_TCLASS(SemiAnalyticalPBD, TDataType)

		__device__ inline float kernGradientMeshPBD(const float r, const float h)
	{
		const Real q = r / h;
		if (q > 1.0f)
			return 0.0;
		else
		{
			//G(r) in equation 6
			const Real d = 1.0 - q;
			const Real hh = h * h;
			return -45.0f / ((Real)M_PI * hh * h) * (1.0f / 3.0f * (hh * h - r * r * r) - 1.0f / 2.0f / h * (hh * hh - r * r * r * r) + 1.0f / 5.0f / hh * (hh * hh * h - r * r * r * r * r));
		}
	}

	template <typename Real,
		typename Coord>
		__global__ void K_InitKernelFunctionMesh(
			DArray<Real>  weights,
			DArray<Coord> posArr,
			DArrayList<int>  neighbors,
			SpikyKernel<Real>  kernel,
			Real               smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= weights.size())
			return;

		Coord pos_i = posArr[pId];

		List<int>& list_i = neighbors[pId];
		int  nbSize = list_i.size();
		Real total_weight = Real(0);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int  j = list_i[ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				total_weight += kernel.Weight(r, smoothingLength);
			}
		}

		weights[pId] = total_weight;
	}

	template <typename Real, typename Coord>
	__global__ void K_ComputeLambdasMesh(
		DArray<Real>                     lambdaArr,
		DArray<Real>                     rhoArr,
		DArray<Coord>                    posArr,
		DArray<TopologyModule::Triangle> Tri,
		DArray<Coord>                    positionTri,
		DArrayList<int>                     neighbors,
		DArrayList<int>                     neighborsTri,
		SpikyKernel<Real>                     kern,
		Real                                  smoothingLength,
		Real                                  sampling_distance,
		int                                   use_mesh,
		int                                   use_ghost,
		int                                   Start)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size())
			return;
		/*printf("rho size %d %d %d %d %d %d %d\n", 
			rhoArr.size(),
			lambdaArr.size(), 
			posArr.size(),
			Tri.size(),
			positionTri.size(),
			neighbors.size(),
			neighborsTri.size());*/

		Coord pos_i = posArr[pId];

		Real  lamda_i = Real(0);
		Coord grad_ci(0);

		List<int>& list_i = neighbors[pId];
		int  nbSize = list_i.size();

		List<int>& triList_i = neighborsTri[pId];
		int  nbSizeTri = triList_i.size();
		Real sum_aij;
		Real dis_n = 10000.0;
		int  nearest_T = 1;

		//semi-analytical boundary integration
		if (use_mesh && pId < Start)
			for (int ne = 0; ne < nbSizeTri; ne++)
			{
				//printf("Yes\n");
				int j = triList_i[ne];
				if (j >= 0)
					continue;
				j *= -1;
				j--;

				Triangle3D t3d(positionTri[Tri[j][0]], positionTri[Tri[j][1]], positionTri[Tri[j][2]]);
				Plane3D    PL(positionTri[Tri[j][0]], t3d.normal());
				Point3D    p3d(pos_i);
				//Point3D nearest_pt = p3d.project(t3d);
				Point3D nearest_pt = p3d.project(PL);
				Real    r = (nearest_pt.origin - pos_i).norm();

				Real  AreaSum = calculateIntersectionArea(p3d, t3d, smoothingLength);  //A_s in equation 10
				Real  MinDistance = (p3d.distance(t3d));                     //d_n (scalar) in equation 10
				Coord Min_Pt = (p3d.project(t3d)).origin - pos_i;       //d_n (vector) in equation 10
				Coord Min_Pos = p3d.project(t3d).origin;
				if (ne < nbSizeTri - 1 && triList_i[ne + 1] < 0)
				{
					//triangle clustering
					int jn;
					do
					{
						jn = triList_i[ne + 1];
						if (jn >= 0)
							break;
						jn *= -1;
						jn--;

						Triangle3D t3d_n(positionTri[Tri[jn][0]], positionTri[Tri[jn][1]], positionTri[Tri[jn][2]]);
						if ((t3d.normal().cross(t3d_n.normal())).norm() > EPSILON)
							break;

						AreaSum += calculateIntersectionArea(p3d, t3d_n, smoothingLength);

						if (abs(p3d.distance(t3d_n)) < abs(MinDistance))
						{
							MinDistance = (p3d.distance(t3d_n));
							Min_Pt = (p3d.project(t3d_n)).origin - pos_i;
							Min_Pos = (p3d.project(t3d_n)).origin;
						}
						//printf("%d %d\n", j, jn);
						ne++;
					} while (ne < nbSizeTri - 1);
				}
				if (abs(MinDistance) < abs(dis_n))
					dis_n = MinDistance;
				Min_Pt /= (-Min_Pt.norm());

				float d = p3d.distance(PL);
				d = abs(d);

				// equation 6
				if (smoothingLength - d > EPSILON && smoothingLength * smoothingLength - d * d > EPSILON && d > EPSILON)
				{

					Real a_ij =
						kernGradientMeshPBD(r, smoothingLength)
						/ (sampling_distance * sampling_distance * sampling_distance)
						* 2.0 * (M_PI) * (1 - d / smoothingLength)                //eq 11
						* AreaSum                                                 //p3d.areaTriangle(t3d, smoothingLength)
						/ ((M_PI) * (smoothingLength * smoothingLength - d * d))  //eq 11
						* t3d.normal().dot(Min_Pt) / t3d.normal().norm();

					{
						Coord g = a_ij * (pos_i - nearest_pt.origin) / r;
						grad_ci += g;
						lamda_i += g.dot(g);
					}
				}
			}
		grad_ci *= (dis_n / abs(dis_n));
		lamda_i *= (dis_n / abs(dis_n));

		//traditional integration position based fluids
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			if (j >= Start && !use_ghost)
				continue;
			//if (j >= Start ) continue;

			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord g = kern.Gradient(r, smoothingLength) * (pos_i - posArr[j]) * (1.0f / r);
				grad_ci += g;
				lamda_i += g.dot(g);
			}
		}

		lamda_i += grad_ci.dot(grad_ci);

		Real rho_i = rhoArr[pId];

		lamda_i = -(rho_i - 1000.0f) / (lamda_i + 0.1f);

		lambdaArr[pId] = lamda_i > 0.0f ? 0.0f : lamda_i;
	}

	template <typename Real, typename Coord>
	__global__ void K_ComputeLambdasMesh(
		DArray<Real>  lambdaArr,
		DArray<Real>  rhoArr,
		DArray<Coord> posArr,
		DArray<Real>  massInvArr,
		DArrayList<int>  neighbors,
		SpikyKernel<Real>  kern,
		Real               smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size())
			return;

		Coord pos_i = posArr[pId];

		Real  lamda_i = Real(0);
		Coord grad_ci(0);

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int  j = list_i[ne];
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord g = kern.Gradient(r, smoothingLength) * (pos_i - posArr[j]) * (1.0f / r);
				grad_ci += g;
				lamda_i += g.dot(g) * massInvArr[j];
			}
		}

		lamda_i += grad_ci.dot(grad_ci) * massInvArr[pId];

		Real rho_i = rhoArr[pId];

		lamda_i = -(rho_i - 1000.0f) / (lamda_i + 0.1f);

		lambdaArr[pId] = lamda_i > 0.0f ? 0.0f : lamda_i;
	}

	template <typename Real, typename Coord>
	__global__ void K_ComputeDisplacementMesh(
		DArray<Coord>                    dPos,
		DArray<Real>                     lambdas,
		DArray<Coord>                    posArr,
		DArray<TopologyModule::Triangle> Tri,
		DArray<Coord>                    positionTri,
		DArrayList<int>                     neighbors,
		DArrayList<int>                     neighborsTri,
		SpikyKernel<Real>                     kern,
		Real                                  smoothingLength,
		Real                                  dt,
		Real                                  sampling_distance,
		int                                   use_mesh,
		int                                   use_ghost,
		int                                   Start)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size())
			return;
		if (pId >= Start)
			return;

		Coord pos_i = posArr[pId];
		Real  lamda_i = lambdas[pId];

		Coord dP_i(0);
		List<int>& list_i = neighbors[pId];
		int   nbSize = list_i.size();

		List<int>& triList_i = neighborsTri[pId];
		int   nbSizeTri = triList_i.size();
		Real  dis_n = 10000.0;
		for (int ne = 0; ne < nbSizeTri; ne++)
		{
			int j = triList_i[ne];
			if (j >= 0)
				continue;
			j *= -1;
			j--;

			Triangle3D t3d(positionTri[Tri[j][0]], positionTri[Tri[j][1]], positionTri[Tri[j][2]]);
			Plane3D    PL(positionTri[Tri[j][0]], t3d.normal());
			Point3D    p3d(pos_i);
			if (abs(p3d.distance(t3d)) < abs(dis_n))
			{
				dis_n = p3d.distance(t3d);
			}
		}
		//semi-analytical boundary integration
		if (use_mesh)
			for (int ne = 0; ne < nbSizeTri; ne++)
			{
				int j = triList_i[ne];
				if (j >= 0)
					continue;
				j *= -1;
				j--;

				Triangle3D t3d(positionTri[Tri[j][0]], positionTri[Tri[j][1]], positionTri[Tri[j][2]]);
				Plane3D    PL(positionTri[Tri[j][0]], t3d.normal());
				Point3D    p3d(pos_i);
				//Point3D nearest_pt = p3d.project(t3d);
				Point3D nearest_pt = p3d.project(PL);
				Real    r = (nearest_pt.origin - pos_i).norm();

				Real  tmp = 1.0;
				float d = p3d.distance(PL);

				Coord ttm = PL.normal;
				//if(pId % 10000 == 0)
				//printf("%.3lf %.3lf %.3lf\n", ttm[0], ttm[1], ttm[2]);
				//	if (d < 0) tmp *= 0.0;

				Real  AreaSum = calculateIntersectionArea(p3d, t3d, smoothingLength);  //A_s in equation 10
				Real  MinDistance = abs(p3d.distance(t3d));                  //d_n (scalar) in equation 10
				Coord Min_Pt = (p3d.project(t3d)).origin - pos_i;       //d_n (vector) in equation 10
				Coord Min_Pos = p3d.project(t3d).origin;
				if (ne < nbSizeTri - 1 && triList_i[ne + 1] < 0)
				{
					//triangle clustering
					int jn;
					do
					{
						jn = triList_i[ne + 1];
						if (jn >= 0)
							break;
						jn *= -1;
						jn--;

						Triangle3D t3d_n(positionTri[Tri[jn][0]], positionTri[Tri[jn][1]], positionTri[Tri[jn][2]]);
						if ((t3d.normal().cross(t3d_n.normal())).norm() > EPSILON)
							break;

						AreaSum += calculateIntersectionArea(p3d, t3d_n, smoothingLength);

						if (abs(p3d.distance(t3d_n)) < abs(MinDistance))
						{
							MinDistance = (p3d.distance(t3d_n));
							Min_Pt = (p3d.project(t3d_n)).origin - pos_i;
							Min_Pos = (p3d.project(t3d_n)).origin;
						}
						//printf("%d %d\n", j, jn);
						ne++;
					} while (ne < nbSizeTri - 1);
				}

				Min_Pt /= (-Min_Pt.norm());

				d = abs(d);
				//r = max((r - sampling_distance / 2.0), 0.0);
				if (smoothingLength - d > EPSILON && smoothingLength * smoothingLength - d * d > EPSILON && d > EPSILON)
				{
					//equaltion 6
					Real a_ij =
						kernGradientMeshPBD(r, smoothingLength)
						* 2.0 * (M_PI) * (1 - d / smoothingLength)                //eq 11
						* AreaSum                                                 //p3d.areaTriangle(t3d, smoothingLength)
						/ ((M_PI) * (smoothingLength * smoothingLength - d * d))  // eq11
						* t3d.normal().dot(Min_Pt) / t3d.normal().norm()          // / (p3d.project(t3d).origin - p3d.origin).norm()
						/ (sampling_distance * sampling_distance * sampling_distance);
					//a_ij *= (dis_n / abs(dis_n));

					Coord dp_ij = 40.0f * (pos_i - nearest_pt.origin) * (lamda_i)*a_ij * (1.0 / (pos_i - nearest_pt.origin).norm());

					//if (a_ij < 0)
					{
						dp_ij *= tmp;
						dP_i += dp_ij;
						atomicAdd(&dPos[pId][0], dp_ij[0]);

						if (Coord::dims() >= 2)
							atomicAdd(&dPos[pId][1], dp_ij[1]);

						if (Coord::dims() >= 3)
							atomicAdd(&dPos[pId][2], dp_ij[2]);
					}
				}
			}
		//traditional integration position based fluids
		for (int ne = 0; ne < nbSize; ne++)
		{
			int  j = list_i[ne];
			Real r = (pos_i - posArr[j]).norm();
			if (r > EPSILON)
				if (j < Start)
				{
					Coord dp_ij = 10.0f * (pos_i - posArr[j]) * (lamda_i + lambdas[j]) * kern.Gradient(r, smoothingLength) * (1.0 / r);
					dP_i += dp_ij;

					atomicAdd(&dPos[pId][0], dp_ij[0]);
					atomicAdd(&dPos[j][0], -dp_ij[0]);

					if (Coord::dims() >= 2)
					{
						atomicAdd(&dPos[pId][1], dp_ij[1]);
						atomicAdd(&dPos[j][1], -dp_ij[1]);
					}

					if (Coord::dims() >= 3)
					{
						atomicAdd(&dPos[pId][2], dp_ij[2]);
						atomicAdd(&dPos[j][2], -dp_ij[2]);
					}
				}
				else if (use_ghost)
				{
					Coord dp_ij = 20.0f * (pos_i - posArr[j]) * (lamda_i + lambdas[j]) * kern.Gradient(r, smoothingLength) * (1.0 / r);
					dP_i += dp_ij;

					atomicAdd(&dPos[pId][0], dp_ij[0]);
					if (Coord::dims() >= 2)
					{
						atomicAdd(&dPos[pId][1], dp_ij[1]);
					}
					if (Coord::dims() >= 3)
					{
						atomicAdd(&dPos[pId][2], dp_ij[2]);
					}
				}
		}
	}

	template <typename Real, typename Coord>
	__global__ void K_ComputeDisplacementMesh(
		DArray<Coord> dPos,
		DArray<Real>  lambdas,
		DArray<Coord> posArr,
		DArray<Real>  massInvArr,
		DArrayList<int>  neighbors,
		SpikyKernel<Real>  kern,
		Real               smoothingLength,
		Real               dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size())
			return;

		Coord pos_i = posArr[pId];
		Real  lamda_i = lambdas[pId];

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int  j = list_i[ne];
			Real r = (pos_i - posArr[j]).norm();
			if (r > EPSILON)
			{
				Coord dp_ij = 10.0f * (pos_i - posArr[j]) * (lamda_i + lambdas[j]) * kern.Gradient(r, smoothingLength) * (1.0 / r);
				Coord dp_ji = -dp_ij * massInvArr[j];
				dp_ij = dp_ij * massInvArr[pId];
				atomicAdd(&dPos[pId][0], dp_ij[0]);
				atomicAdd(&dPos[j][0], dp_ji[0]);

				if (Coord::dims() >= 2)
				{
					atomicAdd(&dPos[pId][1], dp_ij[1]);
					atomicAdd(&dPos[j][1], dp_ji[1]);
				}

				if (Coord::dims() >= 3)
				{
					atomicAdd(&dPos[pId][2], dp_ij[2]);
					atomicAdd(&dPos[j][2], dp_ji[2]);
				}
			}
		}
	}

	template <typename Real, typename Coord>
	__global__ void K_UpdatePositionMesh(
		DArray<Coord> posArr,
		DArray<Coord> velArr,
		DArray<Coord> dPos,
		Real               dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size())
			return;

		posArr[pId] += dPos[pId];
		/*if (dPos[pId].norm() > 0.002)
				printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Yes\n");*/
	}

	template <typename TDataType>
	SemiAnalyticalPBD<TDataType>::SemiAnalyticalPBD()
		: ConstraintModule()
		, m_maxIteration(3)
	{
		m_restDensity.setValue(Real(1000));

		this->inSamplingDistance()->setValue(0.005f);
		use_mesh.setValue(1);
		use_ghost.setValue(0);

		mCalculateDensity = std::make_shared<SemiAnalyticalSummationDensity<TDataType>>();
		this->inSmoothingLength()->connect(mCalculateDensity->inSmoothingLength());
		this->inSamplingDistance()->connect(mCalculateDensity->inSamplingDistance());
		this->inPosition()->connect(mCalculateDensity->inPosition());
		this->inNeighborParticleIds()->connect(mCalculateDensity->inNeighborIds());
		this->inNeighborTriangleIds()->connect(mCalculateDensity->inNeighborTriIds());
		this->inTriangleIndex()->connect(mCalculateDensity->inTriangleInd());
		this->inTriangleVertex()->connect(mCalculateDensity->inTriangleVer());
//		m_restDensity.connect(mCalculateDensity->varRestDensity());
	}

	template <typename TDataType>
	SemiAnalyticalPBD<TDataType>::~SemiAnalyticalPBD()
	{
		m_lamda.clear();
		m_deltaPos.clear();
		m_position_old.clear();
	}

	template <typename TDataType>
	void SemiAnalyticalPBD<TDataType>::constrain()
	{
		int num = this->inPosition()->size();
		//printf("num = %d\n", num);
		if (num == 0)
			return;

		if (m_lamda.size() != num)
			m_lamda.resize(num);
		if (m_deltaPos.size() != num)
			m_deltaPos.resize(num);

		if (num == 0)
			return;
		
		if (m_lamda.size() != num)
			m_lamda.resize(num);
		if (m_deltaPos.size() != num)
			m_deltaPos.resize(num);
		if (m_position_old.size() != num)
			m_position_old.resize(num);

		m_position_old.assign(this->inPosition()->getData());

		int it = 0;
		while (it < m_maxIteration)
		{
			//printf("one\n");
			takeOneIteration();

			it++;
		}

		updateVelocity();
		//printf("endSemi\n");
	}

	template <typename TDataType>
	void SemiAnalyticalPBD<TDataType>::takeOneIteration()
	{
		auto& inPos = this->inPosition()->getData();

		int  num = inPos.size();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		m_deltaPos.reset();
		//printf("b %d %d %d\n",m_deltaPos.size(), num, m_density.size());
		//m_densitySum->update();
		mCalculateDensity->update();
		//printf("c");

		if (m_massInv.isEmpty())
		{
			//printf("YeS\n");
			K_ComputeLambdasMesh<Real, Coord> << <pDims, BLOCK_SIZE >> > (
				m_lamda,
				mCalculateDensity->outDensity()->getData(),
				inPos,
				this->inTriangleIndex()->getData(),
				this->inTriangleVertex()->getData(),
				this->inNeighborParticleIds()->getData(),
				this->inNeighborTriangleIds()->getData(),
				m_kernel,
				this->inSmoothingLength()->getData(),
				this->inSamplingDistance()->getData(),
				use_mesh.getData(),
				use_ghost.getData(),
				Start.getData());
			cuSynchronize();
			//printf("dddf\n");
			K_ComputeDisplacementMesh<Real, Coord> << <pDims, BLOCK_SIZE >> > (
				m_deltaPos,
				m_lamda,
				inPos,
				this->inTriangleIndex()->getData(),
				this->inTriangleVertex()->getData(),
				this->inNeighborParticleIds()->getData(),
				this->inNeighborTriangleIds()->getData(),
				m_kernel,
				this->inSmoothingLength()->getData(),
				this->inTimeStep()->getData(),
				this->inSamplingDistance()->getData(),
				use_mesh.getData(),
				use_ghost.getData(),
				Start.getData());
			cuSynchronize();
		}
		else
		{
			//printf("ee\n");
			K_ComputeLambdasMesh<Real, Coord> << <pDims, BLOCK_SIZE >> > (
				m_lamda,
				m_density.getData(),
				inPos,
				m_massInv.getData(),
				this->inNeighborParticleIds()->getData(),
				m_kernel,
				this->inSmoothingLength()->getData());
			cuSynchronize();

			K_ComputeDisplacementMesh<Real, Coord> << <pDims, BLOCK_SIZE >> > (
				m_deltaPos,
				m_lamda,
				inPos,
				m_massInv.getData(),
				this->inNeighborParticleIds()->getData(),
				m_kernel,
				this->inSmoothingLength()->getData(),
				this->inTimeStep()->getData());
			cuSynchronize();
		}
		//printf("d\n");
		K_UpdatePositionMesh<Real, Coord> << <pDims, BLOCK_SIZE >> > (
			inPos,
			this->inVelocity()->getData(),
			m_deltaPos,
			this->inTimeStep()->getData());
		cuSynchronize();
	}

	template <typename Real, typename Coord>
	__global__ void DP_UpdateVelocityMesh(
		DArray<Coord> velArr,
		DArray<Coord> prePos,
		DArray<Coord> curPos,
		Real               dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.size())
			return;
		//printf("%d %d\n", curPos.size(), velArr.size());
		velArr[pId] += (curPos[pId] - prePos[pId]) / dt;
		if (velArr[pId].norm() > 2.50f)
			velArr[pId] *= (2.50f / velArr[pId].norm());
	}

	template <typename Real, typename Coord>
	__global__ void DP_NVelocityMesh(
		DArray<Coord> velArr,
		DArray<Real>  veln)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.size())
			return;
		veln[pId] = velArr[pId].norm();
	}

	template <typename TDataType>
	void SemiAnalyticalPBD<TDataType>::updateVelocity()
	{
		int  num = this->inPosition()->size();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		DP_UpdateVelocityMesh << <pDims, BLOCK_SIZE >> > (
			this->inVelocity()->getData(),
			m_position_old,
			this->inPosition()->getData(),
			this->inTimeStep()->getData());
		cuSynchronize();
	}

	DEFINE_CLASS(SemiAnalyticalPBD);
}  // namespace PhysIKA