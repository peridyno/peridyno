#include <cuda_runtime.h>
//#include "Core/Utilities/template_functions.h"
#include "Utility.h"
#include "DensityPBDMesh.h"
#include "Framework/Node.h"
#include <string>
#include "DensitySummationMesh.h"
#include "Topology/FieldNeighbor.h"
#include "Topology/Primitive3D.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(DensityPBDMesh, TDataType)

	__device__ inline float kernGradientMeshPBD(const float r, const float h)
	{
		const Real q = r / h;
		if (q > 1.0f) return 0.0;
		else {
			const Real d = 1.0 - q;
			const Real hh = h * h;
			return -45.0f / ((Real)M_PI * hh * h) *
				(
					1.0f / 3.0f * (hh * h - r * r * r)
				-	1.0f / 2.0f / h * (hh * hh - r * r * r * r)
				+	1.0f / 5.0f / hh * (hh * hh * h - r * r * r * r * r)
				);
		}
		/*
			const Real q = r / h;
			if (q > 1.0f) return 0.0;
			//else if (r==0.0f) return 0.0f;
			else {
				const Real d = 1.0 - q;
				const Real hh = h*h;
				return -45.0f / ((Real)M_PI * hh*h) *d*d * this->m_scale;
		*/
	}


	template<typename Real,
			 typename Coord>
	__global__ void K_InitKernelFunctionMesh(
		DeviceArray<Real> weights,
		DeviceArray<Coord> posArr,
		NeighborList<int> neighbors,
		SpikyKernel<Real> kernel,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= weights.size()) return;

		Coord pos_i = posArr[pId];

		int nbSize = neighbors.getNeighborSize(pId);
		Real total_weight = Real(0);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
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
		DeviceArray<Real> lambdaArr,
		DeviceArray<Real> rhoArr,
		DeviceArray<Coord> posArr,
		DeviceArray<TopologyModule::Triangle> Tri,
		DeviceArray<Coord> positionTri,
		NeighborList<int> neighbors,
		NeighborList<int> neighborsTri,
		SpikyKernel<Real> kern,
		Real smoothingLength,
		Real sampling_distance,
		int use_mesh,
		int use_ghost,
		int Start)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord pos_i = posArr[pId];

		Real lamda_i = Real(0);
		Coord grad_ci(0);

		int nbSize = neighbors.getNeighborSize(pId);
		int nbSizeTri = neighborsTri.getNeighborSize(pId);
		Real sum_aij;
		Real dis_n = 10000.0;
		int nearest_T = 1;

		if (use_mesh && pId < Start)
			for (int ne = 0; ne < nbSizeTri; ne++)
			{
				//printf("Yes\n");
				int j = neighborsTri.getElement(pId, ne);
				//if (j >= 0) continue;
				//j *= -1;
				//j--;

				Triangle3D t3d(positionTri[Tri[j][0]], positionTri[Tri[j][1]], positionTri[Tri[j][2]]);
				Plane3D PL(positionTri[Tri[j][0]], t3d.normal());
				Point3D p3d(pos_i);
				//Point3D nearest_pt = p3d.project(t3d);
				Point3D nearest_pt = p3d.project(PL);
				Real r = (nearest_pt.origin - pos_i).norm();

				Real AreaSum = p3d.areaTriangle(t3d, smoothingLength);
				Real MinDistance = (p3d.distance(t3d));
				Coord Min_Pt = (p3d.project(t3d)).origin - pos_i;
				Coord Min_Pos = p3d.project(t3d).origin;
				if (ne < nbSizeTri - 1 )
				{
					int jn;
					do
					{
						jn = neighborsTri.getElement(pId, ne + 1);
						//if (jn >= 0) break;
						//jn *= -1; jn--;

						Triangle3D t3d_n(positionTri[Tri[jn][0]], positionTri[Tri[jn][1]], positionTri[Tri[jn][2]]);
						if ((t3d.normal().cross(t3d_n.normal())).norm() > EPSILON) break;

						AreaSum += p3d.areaTriangle(t3d_n, smoothingLength);

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
				if (smoothingLength - d > EPSILON&& smoothingLength* smoothingLength - d * d > EPSILON&& d > EPSILON)
				{
					//if (r > sampling_distance / 2)
					//	r -= sampling_distance / 2;
					Real a_ij =
						kernGradientMeshPBD(r, smoothingLength)
						/ (sampling_distance * sampling_distance * sampling_distance)
						* 2.0 * (M_PI) * (1 - d / smoothingLength)
						* AreaSum//p3d.areaTriangle(t3d, smoothingLength)
						/ ((M_PI) * (smoothingLength * smoothingLength - d * d))
						* t3d.normal().dot(Min_Pt) / t3d.normal().norm() ;

					//printf("densityPBDMesh, %3lf %.13lf %.3lf \n", 1.0f - r / smoothingLength,
					//	a_ij, kern.Gradient(r + sampling_distance / 2.0f,smoothingLength));
					//if (a_ij > 0)
					{
						Coord g = a_ij * (pos_i - nearest_pt.origin) / r;
						grad_ci += g;
						lamda_i += g.dot(g);
					}

				}
			}
		grad_ci *= (dis_n / abs(dis_n));
		lamda_i *= (dis_n / abs(dis_n));

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			if (j >= Start && !use_ghost) continue;
			//if (j >= Start ) continue;

			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord g = kern.Gradient(r, smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
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
		DeviceArray<Real> lambdaArr,
		DeviceArray<Real> rhoArr,
		DeviceArray<Coord> posArr,
		DeviceArray<Real> massInvArr,
		NeighborList<int> neighbors,
		SpikyKernel<Real> kern,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord pos_i = posArr[pId];

		Real lamda_i = Real(0);
		Coord grad_ci(0);

		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				Coord g = kern.Gradient(r, smoothingLength)*(pos_i - posArr[j]) * (1.0f / r);
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
		DeviceArray<Coord> dPos, 
		DeviceArray<Real> lambdas, 
		DeviceArray<Coord> posArr, 
		DeviceArray<TopologyModule::Triangle> Tri,
		DeviceArray<Coord> positionTri,
		NeighborList<int> neighbors, 
		NeighborList<int> neighborsTri,
		SpikyKernel<Real> kern,
		Real smoothingLength,
		Real dt,
		Real sampling_distance,
		int use_mesh,
		int use_ghost,
		int Start)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;
		if (pId >= Start) return;

		Coord pos_i = posArr[pId];
		Real lamda_i = lambdas[pId];

		Coord dP_i(0);
		int nbSize = neighbors.getNeighborSize(pId);
		int nbSizeTri = neighborsTri.getNeighborSize(pId);
		Real dis_n = 10000.0;
		for (int ne = 0; ne < nbSizeTri; ne++)
		{
			int j = neighborsTri.getElement(pId, ne);
			//if (j >= 0) continue;
		//	j *= -1;
			//j--;

			Triangle3D t3d(positionTri[Tri[j][0]], positionTri[Tri[j][1]], positionTri[Tri[j][2]]);
			Plane3D PL(positionTri[Tri[j][0]], t3d.normal());
			Point3D p3d(pos_i);
			if (abs(p3d.distance(t3d)) < abs(dis_n))
			{
				dis_n = p3d.distance(t3d);
			}
		}
		if (use_mesh)
			for (int ne = 0; ne < nbSizeTri; ne++)
			{
				int j = neighborsTri.getElement(pId, ne);
				//if (j >= 0) continue;
				//j *= -1;
				//j--;

				Triangle3D t3d(positionTri[Tri[j][0]], positionTri[Tri[j][1]], positionTri[Tri[j][2]]);
				Plane3D PL(positionTri[Tri[j][0]], t3d.normal());
				Point3D p3d(pos_i);
				//Point3D nearest_pt = p3d.project(t3d);
				Point3D nearest_pt = p3d.project(PL);
				Real r = (nearest_pt.origin - pos_i).norm();

				Real tmp =1.0;
				float d = p3d.distance(PL);

				Coord ttm = PL.normal;
				//if(pId % 10000 == 0)
				//printf("%.3lf %.3lf %.3lf\n", ttm[0], ttm[1], ttm[2]);
			//	if (d < 0) tmp *= 0.0;

				Real AreaSum = p3d.areaTriangle(t3d, smoothingLength);
				Real MinDistance = abs(p3d.distance(t3d));
				Coord Min_Pt = (p3d.project(t3d)).origin - pos_i;
				Coord Min_Pos = p3d.project(t3d).origin;
				if (ne < nbSizeTri - 1)
				{
					int jn;
					do
					{
						jn = neighborsTri.getElement(pId, ne + 1);
						//if (jn >= 0) break;
						//jn *= -1; jn--;

						Triangle3D t3d_n(positionTri[Tri[jn][0]], positionTri[Tri[jn][1]], positionTri[Tri[jn][2]]);
						if ((t3d.normal().cross(t3d_n.normal())).norm() > EPSILON) break;

						AreaSum += p3d.areaTriangle(t3d_n, smoothingLength);

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
				if (smoothingLength - d > EPSILON&& smoothingLength* smoothingLength - d * d > EPSILON&& d > EPSILON)
				{

					Real a_ij =
						kernGradientMeshPBD(r, smoothingLength)
						* 2.0 * (M_PI) * (1 - d / smoothingLength)
						* AreaSum//p3d.areaTriangle(t3d, smoothingLength)
						/ ((M_PI) * (smoothingLength * smoothingLength - d * d))
						* t3d.normal().dot(Min_Pt) / t3d.normal().norm()// / (p3d.project(t3d).origin - p3d.origin).norm()
						/
						(sampling_distance * sampling_distance * sampling_distance);
					//a_ij *= (dis_n / abs(dis_n));

					Coord dp_ij = 40.0f * (pos_i - nearest_pt.origin) * (lamda_i) * a_ij * (1.0 / (pos_i - nearest_pt.origin).norm());


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


		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - posArr[j]).norm();
			if (r > EPSILON)
				if (j < Start)
				{
					Coord dp_ij = 10.0f*(pos_i - posArr[j])*(lamda_i + lambdas[j])*kern.Gradient(r, smoothingLength)* (1.0 / r);
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
		DeviceArray<Coord> dPos,
		DeviceArray<Real> lambdas,
		DeviceArray<Coord> posArr,
		DeviceArray<Real> massInvArr,
		NeighborList<int> neighbors,
		SpikyKernel<Real> kern,
		Real smoothingLength,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord pos_i = posArr[pId];
		Real lamda_i = lambdas[pId];

		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - posArr[j]).norm();
			if (r > EPSILON)
			{
				Coord dp_ij = 10.0f*(pos_i - posArr[j])*(lamda_i + lambdas[j])*kern.Gradient(r, smoothingLength)* (1.0 / r);
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
		DeviceArray<Coord> posArr, 
		DeviceArray<Coord> velArr, 
		DeviceArray<Coord> dPos, 
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		posArr[pId] += dPos[pId];
	}


	template<typename TDataType>
	DensityPBDMesh<TDataType>::DensityPBDMesh()
		: ConstraintModule()
		, m_maxIteration(3)
	{
		m_restDensity.setValue(Real(1000));
		m_smoothingLength.setValue(Real(0.011));

		attachField(&m_restDensity, "rest_density", "Reference density", false);
		attachField(&m_smoothingLength, "smoothing_length", "The smoothing length in SPH!", false);
		attachField(&m_position, "position", "Storing the particle positions!", false);
		attachField(&m_velocity, "velocity", "Storing the particle velocities!", false);
		attachField(&m_density, "density", "Storing the particle densities!", false);
		attachField(&m_neighborhood, "neighborhood", "Storing neighboring particles' ids!", false);
	}

	template<typename TDataType>
	DensityPBDMesh<TDataType>::~DensityPBDMesh()
	{
		m_lamda.release();
		m_deltaPos.release();
		m_position_old.release();
	}

	template<typename TDataType>
	bool DensityPBDMesh<TDataType>::initializeImpl()
	{
		if (!m_position.isEmpty() && m_density.isEmpty())
		{
			m_density.setElementCount(m_position.getElementCount());
		}

		if (!isAllFieldsReady())
		{
			std::cout << "Exception: " << std::string("DensityPBD's fields are not fully initialized!") << std::endl;
			//return false;
		}

		sampling_distance.setValue(0.005);
		use_mesh.setValue(1);
		use_ghost.setValue(0);

		m_densitySum = std::make_shared<DensitySummationMesh<TDataType>>();

		m_restDensity.connect(&m_densitySum->m_restDensity);
		m_smoothingLength.connect(&m_densitySum->m_smoothingLength);
		m_position.connect(&m_densitySum->m_position);
		m_density.connect(&m_densitySum->m_density);
		m_neighborhood.connect(&m_densitySum->m_neighborhood);
		m_neighborhoodTri.connect(&m_densitySum->m_neighborhoodTri);
		Tri.connect(&m_densitySum->Tri);
		TriPoint.connect(&m_densitySum->TriPoint);
		sampling_distance.connect(&m_densitySum->sampling_distance);
		use_mesh.connect(&m_densitySum->use_mesh);
		use_ghost.connect(&m_densitySum->use_ghost);
		Start.connect(&m_densitySum->Start);

		//printf("INITIALIZE DENSITY\n");
		m_densitySum->initialize();
		//printf("INITIALIZE DENSITY OUT\n");

		int num = m_position.getElementCount();

		if (num == 0)
			return true;

		if (m_lamda.size() != num)
			m_lamda.resize(num);
		if (m_deltaPos.size() != num)
			m_deltaPos.resize(num);
		
		m_position_old.resize(num);

// 		uint pDims = cudaGridSize(num, BLOCK_SIZE);
// 		K_InitKernelFunction << <pDims, BLOCK_SIZE >> > (
// 			m_lamda, 
// 			m_position.getValue(), 
// 			m_neighborhood.getValue(), 
// 			m_kernel,
// 			m_smoothingLength.getValue());
// 
// 		Reduction<Real> reduce;
// 		Real max_weight = reduce.maximum(m_lamda.getDataPtr(), m_lamda.size());
// 		m_kernel.m_scale = 1.0 / max_weight;

		return true;
	}

	template<typename TDataType>
	bool DensityPBDMesh<TDataType>::constrain()
	{
		int num = m_position.getElementCount();

		if (num == 0)
			return true;

		if (m_lamda.size() != num)
			m_lamda.resize(num);
		if (m_deltaPos.size() != num)
			m_deltaPos.resize(num);
		if (m_position_old.size() != num)
			m_position_old.resize(num);

		Function1Pt::copy(m_position_old, m_position.getValue());
		
		int it = 0;
		while (it < m_maxIteration)
		{
			//printf("one\n");
			takeOneIteration();

			it++;
		}

		updateVelocity();

		return true;
	}


	template<typename TDataType>
	void DensityPBDMesh<TDataType>::takeOneIteration()
	{
		Real dt = this->getParent()->getDt();
		//printf("a");
		int num = m_position.getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		m_deltaPos.reset();
		//printf("b %d %d %d\n",m_deltaPos.size(), num, m_density.getElementCount());
		m_densitySum->compute();
		//printf("c");

		if (m_massInv.isEmpty())
		{
			//printf("YeS\n");
			K_ComputeLambdasMesh <Real, Coord> << <pDims, BLOCK_SIZE >> > (
				m_lamda,
				m_density.getValue(),
				m_position.getValue(),
				Tri.getValue(),
				TriPoint.getValue(),
				m_neighborhood.getValue(),
				m_neighborhoodTri.getValue(),
				m_kernel,
				m_smoothingLength.getValue(),
				sampling_distance.getValue(),
				use_mesh.getValue(),
				use_ghost.getValue(),
				Start.getValue());
			cuSynchronize();

			K_ComputeDisplacementMesh <Real, Coord> << <pDims, BLOCK_SIZE >> > (
				m_deltaPos,
				m_lamda,
				m_position.getValue(),
				Tri.getValue(),
				TriPoint.getValue(),
				m_neighborhood.getValue(),
				m_neighborhoodTri.getValue(),
				m_kernel,
				m_smoothingLength.getValue(),
				dt,
				sampling_distance.getValue(),
				use_mesh.getValue(),
				use_ghost.getValue(), 
				Start.getValue());
			cuSynchronize();
		}
		else
		{
			K_ComputeLambdasMesh <Real, Coord> << <pDims, BLOCK_SIZE >> > (
				m_lamda,
				m_density.getValue(),
				m_position.getValue(),
				m_massInv.getValue(),
				m_neighborhood.getValue(),
				m_kernel,
				m_smoothingLength.getValue());
			cuSynchronize();

			K_ComputeDisplacementMesh <Real, Coord> << <pDims, BLOCK_SIZE >> > (
				m_deltaPos,
				m_lamda,
				m_position.getValue(),
				m_massInv.getValue(),
				m_neighborhood.getValue(),
				m_kernel,
				m_smoothingLength.getValue(),
				dt);
			cuSynchronize();
		}
		
		K_UpdatePositionMesh <Real, Coord> << <pDims, BLOCK_SIZE >> > (
			m_position.getValue(),
			m_velocity.getValue(),
			m_deltaPos,
			dt);
		cuSynchronize();
		

	}

	template <typename Real, typename Coord>
	__global__ void DP_UpdateVelocityMesh(
		DeviceArray<Coord> velArr,
		DeviceArray<Coord> prePos,
		DeviceArray<Coord> curPos,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.size()) return;
		//printf("%d %d\n", curPos.size(), velArr.size());
		velArr[pId] += (curPos[pId] - prePos[pId]) / dt;
	}

	template <typename Real, typename Coord>
	__global__ void DP_NVelocityMesh(
		DeviceArray<Coord> velArr,
		DeviceArray<Real> veln)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.size()) return;
		veln[pId] = velArr[pId].norm();
	}

	template<typename TDataType>
	void DensityPBDMesh<TDataType>::updateVelocity()
	{
		int num = m_position.getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		Real dt = this->getParent()->getDt();

		DP_UpdateVelocityMesh << <pDims, BLOCK_SIZE >> > (
			m_velocity.getValue(),
			m_position_old,
			m_position.getValue(),
			dt);
		cuSynchronize();
	}

#ifdef PRECISION_FLOAT
	template class DensityPBDMesh<DataType3f>;
#else
 	template class DensityPBDMesh<DataType3d>;
#endif
}