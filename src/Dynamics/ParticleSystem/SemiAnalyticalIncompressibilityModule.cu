#include <cuda_runtime.h>
#include "SemiAnalyticalIncompressibilityModule.h"
#include "Framework/Node.h"
#include "Utility.h"
#include "SummationDensity.h"
#include "Attribute.h"
#include "Kernel.h"
#include "Topology/Primitive3D.h"



namespace dyno
{
	__device__ inline float kernWeight(const float r, const float h)
	{
		const float q = r / h;
		if (q > 1.0f) return 0.0f;
		else {
			const float d = 1.0f - q;
			const float hh = h * h;
			//			return 45.0f / ((float)M_PI * hh*h) *d*d;
			return (1.0 - pow(q, 4.0f));
			//			return (1.0 - q)*(1.0 - q)*h*h;
		}
	}


	__device__ inline float kernWeightMesh(const float r, const float h)
	{
		if (r > h) return 0.0f;
		return 4.0 / 21.0 * pow(h, 3.0f) - pow(r, 3.0f) / 3.0 + pow(r, 7.0f) / 7.0 / pow(h, 4.0f);
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

	__device__ inline float kernWRMesh(const float r, const float h)
	{
		const float q = r / h;
		const float h04 = 0.4f * h;
		if (q < 0.4f)
			return //(pow(h04, 3.0f) - pow(r, 3.0f)) / 3.0 - (pow(h, 7.0f) - pow(r, 7.0f)) / 7.0 / pow(h, 4.0f) + 0.28 * h * h;
			h * h / 3.0f - (h04 * h04 / 2.0f - pow(h04, 6.0f) / (6.0f * pow(h, 4.0f)))
			+
			((pow(h04, 3.0f) - pow(r, 3.0f)) / 3.0f - (pow(h04, 7.0f) - pow(r, 7.0f)) / 7.0f / pow(h, 4.0f)) / h04
			;
		else
			return h * h / 3.0f - (r * r / 2.0f - pow(r, 6.0f) / (6.0f * pow(h, 4.0f)));//(h * h - r * r) / 3.0;

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

	__device__ inline float kernWRRMesh(const float r, const float h)
	{
		const float q = r / h;
		const float h04 = 0.4f * h;
		if (q < 0.4f)
			return 0.6f * h - 1.0f / 5.0f / pow(h, 4.0f) * (pow(h, 5.0f) - pow(h04, 5.0f)) +
			1.0 / (0.16 * h * h) * (1.0 / 3.0 * (pow(h04, 3.0f) - pow(r, 3.0f)) - 1.0 / (7.0 * pow(h, 4.0f)) * (pow(h04, 7.0f) - pow(r, 7.0f)));
		else
			return h - r - (1.0 / 5.0 / pow(h, 4.0f) * (pow(h, 5.0f) - pow(r, 5.0f)));
	}

	
	
	template <typename Coord>
	__global__ void VC_Sort_Neighbors
	(
		DeviceArray<Coord> position,
		DeviceArray<TopologyModule::Triangle> m_triangle_index,
		DeviceArray<Coord> positionTri,
		NeighborList<int> neighborsTri
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		int nbSizeTri = neighborsTri.getNeighborSize(pId);

		Coord pos_i = position[pId];

		for (int ne = nbSizeTri / 2 - 1; ne >= 0 ; ne--)
		{
			int start = ne;
			int end = nbSizeTri - 1;
			int c = start;           // 当前(current)节点的位置
			int l = 2 * c + 1;        // 左(left)孩子的位置
			int tmp = neighborsTri.getElement(pId, c);            // 当前(current)节点的大小
			for (; l <= end; c = l, l = 2 * l + 1)
			{
				// "l"是左孩子，"l+1"是右孩子
				if (l < end)
				{ 
					bool judge = false;
					{
						int idx1, idx2;
						idx1 = neighborsTri.getElement(pId, l);
						idx2 = neighborsTri.getElement(pId, l + 1);
					//	if (neighborsTri.getElement(pId, l) < 0 && neighborsTri.getElement(pId, l + 1) < 0)
						{
							/*
							idx1 *= -1;
							idx1--;
							idx2 *= -1;
							idx2--;
							*/
							Triangle3D t3d1(positionTri[m_triangle_index[idx1][0]], positionTri[m_triangle_index[idx1][1]], positionTri[m_triangle_index[idx1][2]]);
							Triangle3D t3d2(positionTri[m_triangle_index[idx2][0]], positionTri[m_triangle_index[idx2][1]], positionTri[m_triangle_index[idx2][2]]);

							Coord normal1 = t3d1.normal().normalize();
							Coord normal2 = t3d2.normal().normalize();

							Point3D p3d(pos_i);

							Plane3D PL1(positionTri[m_triangle_index[idx1][0]], t3d1.normal());
							Plane3D PL2(positionTri[m_triangle_index[idx2][0]], t3d2.normal());

							Real dis1 = p3d.distance(PL1);
							Real dis2 = p3d.distance(PL2);

							if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON && abs(normal1[1] - normal2[1]) < EPSILON)
								judge = normal1[2] < normal2[2] ? true : false;
							else if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON)
								judge = normal1[1] < normal2[1] ? true : false;
							else if (abs(dis1 - dis2) < EPSILON)
								judge = normal1[0] < normal2[0] ? true : false;
							else
								judge = dis1 <= dis2 ? true : false;

						}
						//else judge = neighborsTri.getElement(pId, l) < neighborsTri.getElement(pId, l + 1) ? true : false;
					}
					if(judge)
						l++;
				}
				bool judge = false;
				{
					int idx1, idx2;
					idx1 = neighborsTri.getElement(pId, l);
					idx2 = tmp;
					//if (neighborsTri.getElement(pId, l) < 0 && tmp < 0)
					{
						/*
						idx1 *= -1;
						idx1--;
						idx2 *= -1;
						idx2--;*/
						Triangle3D t3d1(positionTri[m_triangle_index[idx1][0]], positionTri[m_triangle_index[idx1][1]], positionTri[m_triangle_index[idx1][2]]);
						Triangle3D t3d2(positionTri[m_triangle_index[idx2][0]], positionTri[m_triangle_index[idx2][1]], positionTri[m_triangle_index[idx2][2]]);

						Coord normal1 = t3d1.normal().normalize();
						Coord normal2 = t3d2.normal().normalize();

						Point3D p3d(pos_i);

						Plane3D PL1(positionTri[m_triangle_index[idx1][0]], t3d1.normal());
						Plane3D PL2(positionTri[m_triangle_index[idx2][0]], t3d2.normal());

						Real dis1 = p3d.distance(PL1);
						Real dis2 = p3d.distance(PL2);

						if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON && abs(normal1[1] - normal2[1]) < EPSILON)
							judge = normal1[2] <= normal2[2] ? true : false;
						else if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON)
							judge = normal1[1] <= normal2[1] ? true : false;
						else if (abs(dis1 - dis2) < EPSILON)
							judge = normal1[0] <= normal2[0] ? true : false;
						else
							judge = dis1 <= dis2 ? true : false;

					}
					//else judge = neighborsTri.getElement(pId, l) <= tmp ? true : false;
				}
				if (judge)
					break;        
				else            
				{
					neighborsTri.setElement(pId, c, neighborsTri.getElement(pId, l));
					neighborsTri.setElement(pId, l, tmp);
				}
			}
		}
		for (int ne = nbSizeTri - 1; ne > 0; ne--)
		{
			int swap_tmp = neighborsTri.getElement(pId, 0);
			neighborsTri.setElement(pId, 0, neighborsTri.getElement(pId, ne));
			neighborsTri.setElement(pId, ne, swap_tmp);
			int start = 0;
			int end = ne - 1;
			int c = start;           
			int l = 2 * c + 1;        
			int tmp = neighborsTri.getElement(pId, c);           
			for (; l <= end; c = l, l = 2 * l + 1)
			{
				// "l"是左孩子，"l+1"是右孩子
				if (l < end)
				{
					bool judge = false;
					{
						int idx1, idx2;
						idx1 = neighborsTri.getElement(pId, l);
						idx2 = neighborsTri.getElement(pId, l + 1);
						//if (neighborsTri.getElement(pId, l) < 0 && neighborsTri.getElement(pId, l + 1) < 0)
						{
							/*
							idx1 *= -1;
							idx1--;
							idx2 *= -1;
							idx2--;*/
							Triangle3D t3d1(positionTri[m_triangle_index[idx1][0]], positionTri[m_triangle_index[idx1][1]], positionTri[m_triangle_index[idx1][2]]);
							Triangle3D t3d2(positionTri[m_triangle_index[idx2][0]], positionTri[m_triangle_index[idx2][1]], positionTri[m_triangle_index[idx2][2]]);

							Coord normal1 = t3d1.normal().normalize();
							Coord normal2 = t3d2.normal().normalize();

							Point3D p3d(pos_i);

							Plane3D PL1(positionTri[m_triangle_index[idx1][0]], t3d1.normal());
							Plane3D PL2(positionTri[m_triangle_index[idx2][0]], t3d2.normal());

							Real dis1 = p3d.distance(PL1);
							Real dis2 = p3d.distance(PL2);

							if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON && abs(normal1[1] - normal2[1]) < EPSILON)
								judge = normal1[2] < normal2[2] ? true : false;
							else if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON)
								judge = normal1[1] < normal2[1] ? true : false;
							else if (abs(dis1 - dis2) < EPSILON)
								judge = normal1[0] < normal2[0] ? true : false;
							else
								judge = dis1 < dis2 ? true : false;

						}
						//else judge = neighborsTri.getElement(pId, l) < neighborsTri.getElement(pId, l + 1) ? true : false;
					}
					if (judge)
						l++;
				}
				bool judge = false;
				{
					int idx1, idx2;
					idx1 = neighborsTri.getElement(pId, l);
					idx2 = tmp;
					//if (neighborsTri.getElement(pId, l) < 0 && tmp < 0)
					{
						/*
						idx1 *= -1;
						idx1--;
						idx2 *= -1;
						idx2--;*/
						Triangle3D t3d1(positionTri[m_triangle_index[idx1][0]], positionTri[m_triangle_index[idx1][1]], positionTri[m_triangle_index[idx1][2]]);
						Triangle3D t3d2(positionTri[m_triangle_index[idx2][0]], positionTri[m_triangle_index[idx2][1]], positionTri[m_triangle_index[idx2][2]]);

						Plane3D PL1(positionTri[m_triangle_index[idx1][0]], t3d1.normal());
						Plane3D PL2(positionTri[m_triangle_index[idx2][0]], t3d2.normal());

						Coord normal1 = t3d1.normal().normalize();
						Coord normal2 = t3d2.normal().normalize();

						Point3D p3d(pos_i);

						Real dis1 = p3d.distance(PL1);
						Real dis2 = p3d.distance(PL2);

						if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON && abs(normal1[1] - normal2[1]) < EPSILON)
							judge = normal1[2] <= normal2[2] ? true : false;
						else if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON)
							judge = normal1[1] <= normal2[1] ? true : false;
						else if (abs(dis1 - dis2) < EPSILON)
							judge = normal1[0] <= normal2[0] ? true : false;
						else
							judge = dis1 <= dis2 ? true : false;

					}
					//else judge = neighborsTri.getElement(pId, l) <= tmp ? true : false;
				}
				if (judge)
					break;
				else
				{
					neighborsTri.setElement(pId, c, neighborsTri.getElement(pId, l));
					neighborsTri.setElement(pId, l, tmp);
				}
			}
		}
	}


	template <typename Real, typename Coord>
	__global__ void VC_ComputeAlphaTmp
	(
		DeviceArray<Real> alpha,
		DeviceArray<Real> rho_alpha,
		DeviceArray<Real> mass,
		DeviceArray<Coord> position,
		DeviceArray<TopologyModule::Triangle> m_triangle_index,
		DeviceArray<Coord> positionTri,
		DeviceArray<Attribute> attribute,
		NeighborList<int> neighbors,
		NeighborList<int> neighborsTri,
		Real smoothingLength,
		Real m_sampling_distance,
		DeviceArray<int> flip
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (pId >= position.size()) return;
		if (!attribute[pId].IsDynamic()) return;

		

		Coord pos_i = position[pId];
		Real alpha_i = 0.0f;
		Real ra = 0.0f;
		int nbSize = neighbors.getNeighborSize(pId);
		int nbSizeTri;
		
		nbSizeTri = neighborsTri.getNeighborSize(pId);
		
		Real debug_r = smoothingLength;
		bool tmp = false;
		Real AreaB;
		for (int ne = 0; ne < nbSizeTri; ne++)
		{
			int j = neighborsTri.getElement(pId, ne);
			//if (j >= 0) continue;
			//j *= -1;
			//j--;
			//	Real m_sampling_distance = 0.015;

			Triangle3D t3d(positionTri[m_triangle_index[j][0]], positionTri[m_triangle_index[j][1]], positionTri[m_triangle_index[j][2]]);
			Plane3D PL(positionTri[m_triangle_index[j][0]], t3d.normal());
			Point3D p3d(pos_i);
			//Point3D nearest_pt = p3d.project(t3d);
			Point3D nearest_pt = p3d.project(PL);
			Real r = (nearest_pt.origin - pos_i).norm();
			tmp = true;
			float d = p3d.distance(PL);

			Real AreaSum = p3d.areaTriangle(t3d, smoothingLength);
			Real MinDistance = abs(p3d.distance(t3d));
			Coord Min_Pt = (p3d.project(t3d)).origin - pos_i;
			if (ne < nbSizeTri - 1)
			{
				int jn;
				do
				{
					jn = neighborsTri.getElement(pId, ne + 1);
					//if (jn >= 0) break;
					//jn *= -1; jn--;

					Triangle3D t3d_n(positionTri[m_triangle_index[jn][0]], positionTri[m_triangle_index[jn][1]], positionTri[m_triangle_index[jn][2]]);
					if ((t3d.normal().cross(t3d_n.normal())).norm() > EPSILON) break;

					AreaSum += p3d.areaTriangle(t3d_n, smoothingLength);
					
					if (abs(p3d.distance(t3d_n)) < MinDistance)
					{
						MinDistance = abs(p3d.distance(t3d_n));
						Min_Pt = (p3d.project(t3d_n)).origin - pos_i;
					}
					//printf("%d %d\n", j, jn);
					ne++;
				} while (ne < nbSizeTri - 1);
			}
			Min_Pt /= Min_Pt.norm();


			d = abs(d);
			if (smoothingLength - d > EPSILON&& smoothingLength* smoothingLength - d * d > EPSILON&& d > EPSILON)
			{
				Coord n_PL = - t3d.normal();
				if (flip[pId] < 0)  n_PL *= -1;
				Coord n_TR = (p3d.project(t3d)).origin - pos_i;
				n_PL = n_PL / n_PL.norm();
				n_TR = n_TR / n_TR.norm();
				
				AreaB = M_PI * (smoothingLength * smoothingLength - d * d);

				Real a_ij =
					kernWeightMesh(r, smoothingLength)
					* 2.0 * (M_PI) * (1 - d / smoothingLength)
					//* p3d.areaTriangle(t3d, smoothingLength) * n_PL.dot(n_TR)
					* AreaSum * n_PL.dot(Min_Pt)
					/ ((M_PI) * (smoothingLength * smoothingLength - d * d)) /
					(m_sampling_distance * m_sampling_distance * m_sampling_distance);
				debug_r = min(r, debug_r);
				alpha_i += a_ij;
			}
		}
		
		Real alpha_solid = alpha_i;
		Real GT_alpha = Real(0);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - position[j]).norm();;

			//if (r > EPSILON)
			if (r > EPSILON&& attribute[j].IsDynamic())
			{
				Real a_ij = kernWeight(r, smoothingLength);
				alpha_i += a_ij;
			}
			else if (r > EPSILON)
			{
				//printf("Alpha r: %.3lf posj: %.3lf %.3lf %.3lf \n", r,position[j][0], position[j][1],position[j][2]);
				Real a_ij = kernWeight(r, smoothingLength);
				GT_alpha += a_ij;
			}
		}

		alpha[pId] = alpha_i;
		rho_alpha[pId] = ra;
	}

	template <typename Real>
	__global__ void VC_CorrectAlphaTmp
	(
		DeviceArray<Real> alpha,
		DeviceArray<Real> rho_alpha,
		DeviceArray<Real> mass,
		Real maxAlpha)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= alpha.size()) return;

		Real alpha_i = alpha[pId];
		Real ra = rho_alpha[pId];
		if (alpha_i < maxAlpha)
		{
			ra += (maxAlpha - alpha_i) * mass[pId];
			alpha_i = maxAlpha;
		}
		alpha[pId] = alpha_i;
		rho_alpha[pId] = ra;
	}

	template <typename Real, typename Coord>
	__global__ void VC_ComputeDiagonalElementTmp
	(
		DeviceArray<Real> AiiFluid,
		DeviceArray<Real> AiiTotal,
		DeviceArray<Real> alpha,
		DeviceArray<Coord> position,
		DeviceArray<TopologyModule::Triangle> m_triangle_index,
		DeviceArray<Coord> positionTri,
		DeviceArray<Attribute> attribute,
		NeighborList<int> neighbors,
		NeighborList<int> neighborsTri,
		Real smoothingLength,
		Real m_sampling_distance,
		DeviceArray<int> flip
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		//printf("Yes\n");

		if (!attribute[pId].IsDynamic()) return;

		Real invAlpha = 1.0f / alpha[pId];

		
		Real diaA_total = 0.0f;
		Real diaA_fluid = 0.0f;
		Real diaA_test = 0.0f;
		Coord pos_i = position[pId];

		bool bNearWall = false;
		int nbSize = neighbors.getNeighborSize(pId);

		int nbSizeTri;
		nbSizeTri = neighborsTri.getNeighborSize(pId);
		float dd;
		for (int ne = 0; ne < nbSizeTri; ne++)
		{
			int j = neighborsTri.getElement(pId, ne);
			//if (j >= 0) continue;
			//j *= -1;
			//j--;
			//Real m_sampling_distance = 0.015;

			Triangle3D t3d(positionTri[m_triangle_index[j][0]], positionTri[m_triangle_index[j][1]], positionTri[m_triangle_index[j][2]]);
			Plane3D PL(positionTri[m_triangle_index[j][0]], t3d.normal());
			Point3D p3d(pos_i);
			//Point3D nearest_pt = p3d.project(t3d);
			Point3D nearest_pt = p3d.project(PL);
			Real r = (nearest_pt.origin - pos_i).norm();
			float d = p3d.distance(PL);
			//if (d < 0) continue;
			d = abs(d);
			dd = d;

			Real AreaSum = p3d.areaTriangle(t3d, smoothingLength);
			Real MinDistance = abs(p3d.distance(t3d));
			Coord Min_Pt = (p3d.project(t3d)).origin - pos_i;
			if (ne < nbSizeTri - 1 )
			{
				int jn;
				do
				{
					jn = neighborsTri.getElement(pId, ne + 1);
					//if (jn >= 0) break;
					//jn *= -1; jn--;

					Triangle3D t3d_n(positionTri[m_triangle_index[jn][0]], positionTri[m_triangle_index[jn][1]], positionTri[m_triangle_index[jn][2]]);
					if ((t3d.normal().cross(t3d_n.normal())).norm() > EPSILON) break;

					AreaSum += p3d.areaTriangle(t3d_n, smoothingLength);

					if (abs(p3d.distance(t3d_n)) < MinDistance)
					{
						MinDistance = abs(p3d.distance(t3d_n));
						Min_Pt = (p3d.project(t3d_n)).origin - pos_i;
					}
					//printf("%d %d\n", j, jn);
					ne++;
				} while (ne < nbSizeTri - 1);
			}
			Min_Pt /= Min_Pt.norm();
			if (smoothingLength - d > EPSILON&& smoothingLength* smoothingLength - d * d > EPSILON&& d > EPSILON)
			{

				//Coord n_PL = nearest_pt.origin - pos_i;
				Coord n_PL = - t3d.normal();
				if (flip[pId] < 0)  n_PL *= -1;
				Coord n_TR = (p3d.project(t3d)).origin - pos_i;
				n_PL = n_PL / n_PL.norm();
				n_TR = n_TR / n_TR.norm();
				Real wrr_ij = invAlpha *
					kernWRRMesh(r, smoothingLength)
					* 2.0 * (M_PI) * (1 - d / smoothingLength)
					//* p3d.areaTriangle(t3d, smoothingLength) * n_PL.dot(n_TR)
					* AreaSum * n_PL.dot(Min_Pt)
					/ ((M_PI) * (smoothingLength * smoothingLength - d * d))
					/ (m_sampling_distance * m_sampling_distance * m_sampling_distance);
				//	printf("++++++++++++++++++++++++++++++++++++%.3lf\n", wrr_ij);
				if (abs(wrr_ij) > 3000)
				{
					//	printf("WRR: %.3lf ANGLE: %.7lf distance:%.3lf weight:%.3lf\n", kernWRRMesh(r, smoothingLength), (smoothingLength * smoothingLength - d * d),
					//		d,wrr_ij);
				}
				diaA_total += 2.0f * wrr_ij;
				diaA_test += 2.0f * wrr_ij;
			}
		}
		Real diaA_GT = Real(0);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - position[j]).norm();

			Attribute att_j = attribute[j];
			if (r > EPSILON)
			{
				Real wrr_ij = invAlpha * kernWRR(r, smoothingLength);
				if (att_j.IsDynamic())
				{
					diaA_total += wrr_ij;
					diaA_fluid += wrr_ij;
					atomicAdd(&AiiFluid[j], wrr_ij);
					atomicAdd(&AiiTotal[j], wrr_ij);
				}
				else
				{
					//diaA_total += 2.0f * wrr_ij;
					diaA_GT += 2.0f * wrr_ij;
				}
			}
		}

		atomicAdd(&AiiFluid[pId], diaA_fluid);
		atomicAdd(&AiiTotal[pId], diaA_total);
		//if (abs(diaA_test) > EPSILON)
		//	printf("========diaA_FOR_TEST: %.3lf GT:%.3lf DISTANCE: %.3lf\n",diaA_test, diaA_GT, dd);
	}

	template <typename Real, typename Coord>
	__global__ void VC_ComputeDiagonalElementTmp
	(
		DeviceArray<Real> diaA,
		DeviceArray<Real> alpha,
		DeviceArray<Coord> position,
		DeviceArray<Attribute> attribute,
		NeighborList<int> neighbors,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].IsDynamic()) return;

		Coord pos_i = position[pId];
		Real invAlpha_i = 1.0f / alpha[pId];
		Real A_i = 0.0f;

		int nbSize = neighbors.getNeighborSize(pId);

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - position[j]).norm();

			if (r > EPSILON&& attribute[j].IsDynamic())
			{
				Real wrr_ij = invAlpha_i * kernWRR(r, smoothingLength);
				A_i += wrr_ij;
				atomicAdd(&diaA[j], wrr_ij);
			}
		}

		atomicAdd(&diaA[pId], A_i);
	}

	template <typename Real, typename Coord>
	__global__ void VC_DetectSurfaceTmp
	(
		DeviceArray<Real> Aii,
		DeviceArray<bool> bSurface,
		DeviceArray<Real> AiiFluid,
		DeviceArray<Real> AiiTotal,
		DeviceArray<Coord> position,
		DeviceArray<Attribute> attribute,
		NeighborList<int> neighbors,
		Real smoothingLength,
		Real maxA
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].IsDynamic()) return;

		Real total_weight = 0.0f;
		Coord div_i = Coord(0);

		SmoothKernel<Real> kernSmooth;

		Coord pos_i = position[pId];
		int nbSize = neighbors.getNeighborSize(pId);
		bool bNearWall = false;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - position[j]).norm();

			if (r > EPSILON&& attribute[j].IsDynamic())
			{
				float weight = -kernSmooth.Gradient(r, smoothingLength);
				total_weight += weight;
				div_i += (position[j] - pos_i) * (weight / r);
			}

			if (!attribute[j].IsDynamic())
			{
				//bNearWall = true;
			}
		}



		total_weight = total_weight < EPSILON ? 1.0f : total_weight;
		Real absDiv = div_i.norm() / total_weight;

		bool bSurface_i = false;
		Real diagF_i = AiiFluid[pId];
		Real diagT_i = AiiTotal[pId];

		if (abs(diagT_i - diagF_i) > EPSILON)bNearWall = true;

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
	__global__ void VC_ComputeDivergenceTmp
	(
		DeviceArray<Real> divergence,
		DeviceArray<Real> alpha,
		DeviceArray<Real> density,
		DeviceArray<Coord> position,
		DeviceArray<Coord> velocity,
		DeviceArray<Coord> velocityTri,
		DeviceArray<TopologyModule::Triangle> m_triangle_index,
		DeviceArray<Coord> positionTri,
		DeviceArray<bool> bSurface,
		DeviceArray<Attribute> attribute,
		DeviceArray<Real> mass,
		DeviceArray<Real> m_triangle_vertex_mass,
		NeighborList<int> neighbors,
		NeighborList<int> neighborsTri,
		Real separation,
		Real tangential,
		Real restDensity,
		Real smoothingLength,
		Real m_sampling_distance,
		Real dt,
		DeviceArray<int> flip
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].IsDynamic()) return;
		

		Coord pos_i = position[pId];
		Coord vel_i = velocity[pId];

		Real div_vi = 0.0f;

		Real invAlpha_i = 1.0f / alpha[pId];
		Real mass_i = mass[pId];

		//mass_i = 10.0;
		Real mass_i0 = mass_i;

		int nbSizeTri;
		
		nbSizeTri = neighborsTri.getNeighborSize(pId);
		

		float div_debug, ddb;
		div_debug = float(0.0);
		int pop = 0;

		Real DIV_GT, DIV_FL;
		DIV_GT = Real(0);
		DIV_FL = Real(0);


		Real sum_weight_norm = Real(0);
		Coord average_normal_j = Coord(0);
		for (int ne = 0; ne < nbSizeTri; ne++)
		{
			int j = neighborsTri.getElement(pId, ne);

			//if (j >= 0) continue;
			//j *= -1; j--;

			Triangle3D t3d(positionTri[m_triangle_index[j][0]], positionTri[m_triangle_index[j][1]], positionTri[m_triangle_index[j][2]]);
			Plane3D PL(positionTri[m_triangle_index[j][0]], t3d.normal());
			Point3D p3d(pos_i);
			//Point3D nearest_pt = p3d.project(t3d);
			Point3D nearest_pt = p3d.project(PL);
			Point3D plnpt = p3d.project(PL);
			Real r = (nearest_pt.origin - pos_i).norm();
			float d = p3d.distance(PL);
			d = abs(d);
			ddb = d;
			if (r < EPSILON) continue;
			if (smoothingLength - d > EPSILON&& smoothingLength* smoothingLength - d * d > EPSILON&& d > EPSILON)
			{
				pop = 1;
				Coord n_PL = nearest_pt.origin - pos_i;
				Coord n_TR = (p3d.project(t3d)).origin - pos_i;
				n_PL = n_PL / n_PL.norm();
				n_TR = n_TR / n_TR.norm();

				float wr_ij =
					kernWeightMesh(r, smoothingLength)
					* 2.0 * (M_PI) * (1 - d / smoothingLength)
					* p3d.areaTriangle(t3d, smoothingLength) //* n_PL.dot(n_TR)
					/ ((M_PI) * (smoothingLength * smoothingLength - d * d))
					/ (m_sampling_distance * m_sampling_distance * m_sampling_distance);
				Coord normal_j = t3d.normal().normalize();
				//sum_weight_norm += wr_ij;
				average_normal_j += wr_ij * normal_j;
			}
		}
		average_normal_j = average_normal_j.normalize();

		for (int ne = 0; ne < nbSizeTri; ne++)
		{
			int j = neighborsTri.getElement(pId, ne);
			//printf("%d\n",j);
			//if (j >= 0) continue;
			//j *= -1; j--;
			//Real m_sampling_distance = 0.015;
			//printf("YESSSSSSSSSSSSSS\n");
			Triangle3D t3d(positionTri[m_triangle_index[j][0]], positionTri[m_triangle_index[j][1]], positionTri[m_triangle_index[j][2]]);
			Plane3D PL(positionTri[m_triangle_index[j][0]], t3d.normal());
			Point3D p3d(pos_i);
			//Point3D nearest_pt = p3d.project(t3d);
			Point3D nearest_pt = p3d.project(PL);
			Point3D plnpt = p3d.project(PL);
			Real r = (nearest_pt.origin - pos_i).norm();
			float d = p3d.distance(PL);
			d = abs(d);
			ddb = d;

			Real AreaSum = p3d.areaTriangle(t3d, smoothingLength);
			Real MinDistance = abs(p3d.distance(t3d));
			Coord Min_Pt = (p3d.project(t3d)).origin - pos_i;
			if (ne < nbSizeTri - 1)
			{
				int jn;
				do
				{
					jn = neighborsTri.getElement(pId, ne + 1);
					//if (jn >= 0) break;
					//jn *= -1; jn--;

					Triangle3D t3d_n(positionTri[m_triangle_index[jn][0]], positionTri[m_triangle_index[jn][1]], positionTri[m_triangle_index[jn][2]]);
					if ((t3d.normal().cross(t3d_n.normal())).norm() > EPSILON) break;

					AreaSum += p3d.areaTriangle(t3d_n, smoothingLength);

					if (abs(p3d.distance(t3d_n)) < MinDistance)
					{
						MinDistance = abs(p3d.distance(t3d_n));
						Min_Pt = (p3d.project(t3d_n)).origin - pos_i;
					}
					//printf("%d %d\n", j, jn);
					ne++;
				} while (ne < nbSizeTri - 1);
			}
			Min_Pt /= Min_Pt.norm();

			if (r < EPSILON) continue;
			if (smoothingLength - d > EPSILON&& smoothingLength* smoothingLength - d * d > EPSILON&& d > EPSILON)
			{
				pop = 1;

				//Coord n_PL = nearest_pt.origin - pos_i;
				Coord n_PL = - t3d.normal();
				if (flip[pId] < 0)  n_PL *= -1;
				Coord n_TR = (p3d.project(t3d)).origin - pos_i;
				n_PL = n_PL / n_PL.norm();
				n_TR = n_TR / n_TR.norm();

				float wr_ij =
					kernWRMesh(r, smoothingLength)
					* 2.0 * (M_PI) * (1 - d / smoothingLength)
				//	* p3d.areaTriangle(t3d, smoothingLength) * n_PL.dot(n_TR)
					* AreaSum * n_PL.dot(Min_Pt)
					/ ((M_PI) * (smoothingLength * smoothingLength - d * d))
					/ (m_sampling_distance * m_sampling_distance * m_sampling_distance);
				//printf("WRIJ IN DIV: %.3lf DISTANCE: %.3lf Kern:%.3lf areaTri: %.3lf\n", wr_ij,d, kernWRMesh(r, smoothingLength), p3d.areaTriangle(t3d, smoothingLength));

				Coord g = -invAlpha_i * (pos_i - nearest_pt.origin) * wr_ij * (1.0f / r);
				if (r < EPSILON)
					g = -invAlpha_i * wr_ij * t3d.normal().normalize();



				if (!((g.norm()) < 100000000000.0))
				{
					printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %.10lf kermMesh:%.3lf areaTri: %.3lf AreaSphere%.3lf~~~~~~~~~~~~~~\n",
						wr_ij,
						kernWRMesh(r, smoothingLength),
						p3d.areaTriangle(t3d, smoothingLength),
						(smoothingLength * smoothingLength - d * d));
				}

				Coord Velj = (velocityTri[m_triangle_index[j][0]] + velocityTri[m_triangle_index[j][1]] + velocityTri[m_triangle_index[j][2]]) / 3.0;
				//printf("J: %d Velj: %.3lf %.3lf %.3lf Pos:%.3lf %.3lf %.3lf\n",j ,positionTri[m_triangle_index[j][0]][0], positionTri[m_triangle_index[j][0]][1], positionTri[m_triangle_index[j][0]][2], 
				//	position[pId][0], position[pId][1], position[pId][2]);

				Real mass_j = m_triangle_vertex_mass[m_triangle_index[j][0]];


				Coord normal_j = t3d.normal();
				normal_j = normal_j.normalize();
				//normal_j = average_normal_j;
				//if (normal_j.dot(pos_i - nearest_pt.origin) < 0) 
				if (flip[pId] < 0)  n_PL *= -1;
					normal_j *= -1;
				//
				//	printf("NORMAL_J: %.3lf %.3lf %.3lf       nij: %.3lf %.3lf %.3lf\n",normal_j[0], normal_j[1], normal_j[2], nij[0], nij[1], nij[2]);


				Coord dVel = vel_i - Velj;
				Real magNVel = dVel.dot(normal_j);
				Coord nVel = magNVel * normal_j;
				Coord tVel = dVel - nVel;

				if (magNVel < -EPSILON)
				{
					Real div_ij = g.dot(2.0f * (nVel + tangential * tVel)) * restDensity * mass_i / dt;
					atomicAdd(&divergence[pId], div_ij);
					div_debug += div_ij;
					//	if (pos_i[1] < 0.035 && pId % 300 == 0)
					//		printf("down, %.3lf %.3lf %.3lf \n", nVel[0], nVel[1], nVel[2]);

				}
				else
				{
					Real div_ij = g.dot(2.0f * ((separation)*nVel + tangential * tVel)) * mass_i * restDensity / dt;
					atomicAdd(&divergence[pId], div_ij);
					div_debug += div_ij;
					//	if (pos_i[1] < 0.035 && pId % 300 == 0)
					//		printf("up, %.3lf %.3lf %.3lf \n",nVel[0],nVel[1],nVel[2]);
				}

			}

		}

		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - position[j]).norm();
			Real mass_j = mass[j];
			

			if (r > EPSILON)
			{
				Real wr_ij = kernWR(r, smoothingLength);
				Coord g = -invAlpha_i * (pos_i - position[j]) * wr_ij * (1.0f / r);
				//mass_i = min(mass_i0, mass_j);
				if ((attribute[j].IsDynamic() && attribute[j].IsRigid() && attribute[pId].IsRigid())) {}
				//else if(attribute[j].IsDynamic())
				else if (attribute[j].IsFluid() && attribute[pId].IsFluid())
				{
					Real div_ij = 0.5f * (vel_i - velocity[j]).dot(g) * (mass_i)*restDensity / dt;

					atomicAdd(&divergence[pId], div_ij);
					atomicAdd(&divergence[j], div_ij);
				}

			}

		}

		if ((!(abs(div_debug) < EPSILON)))
		{
			//	printf("FROM DIV: %.3lf GT: %.3lf\n", div_debug, divergence[pId] - div_debug);
			
		}
	}

	template <typename Real, typename Coord>
	__global__ void VC_CompensateSourceTmp
	(
		DeviceArray<Real> divergence,
		DeviceArray<Real> density,
		DeviceArray<Attribute> attribute,
		DeviceArray<Coord> position,
		Real restDensity,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= density.size()) return;
		if (!attribute[pId].IsDynamic()) return;

		Coord pos_i = position[pId];
		if (density[pId] > restDensity)
		{
			Real ratio = (density[pId] - restDensity) / restDensity;
			atomicAdd(&divergence[pId], 1000000.0f * ratio / dt);
		}
	}

	// compute Ax;
	template <typename Real, typename Coord>
	__global__ void VC_ComputeAxTmp
	(
		DeviceArray<Real> residual,
		DeviceArray<Real> pressure,
		DeviceArray<Real> aiiSymArr,
		DeviceArray<Real> alpha,
		DeviceArray<Coord> position,
		DeviceArray<Attribute> attribute,
		NeighborList<int> neighbor,
		Real smoothingLength
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].IsDynamic()) return;

		Coord pos_i = position[pId];
		Real invAlpha_i = 1.0f / alpha[pId];

		atomicAdd(&residual[pId], aiiSymArr[pId] * pressure[pId]);
		Real con1 = 1.0f;// PARAMS.mass / PARAMS.restDensity / PARAMS.restDensity;

		int nbSize = neighbor.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbor.getElement(pId, ne);
			Real r = (pos_i - position[j]).norm();

			if (r > EPSILON&& attribute[j].IsDynamic())
			{
				Real wrr_ij = kernWRR(r, smoothingLength);
				Real a_ij = -invAlpha_i * wrr_ij;
				//				residual += con1*a_ij*preArr[j];
				atomicAdd(&residual[pId], con1 * a_ij * pressure[j]);
				atomicAdd(&residual[j], con1 * a_ij * pressure[pId]);
			}
		}
	}


	template <typename Real>
	__global__ void VC_InitAttrTmp(
		DeviceArray<Attribute> attribute,
		DeviceArray<Real> mass
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= attribute.size()) return;
		attribute[pId].SetDynamic();
		attribute[pId].SetFluid();
		mass[pId] = 10.0;
	}

	template <typename Coord, typename Real>
	__global__ void VC_TriVelTmp(
		DeviceArray<Coord> oldPos,
		DeviceArray<Coord> newPos,
		DeviceArray<Coord> vels,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		//printf("YES\n");
		if (pId >= oldPos.size()) return;
		vels[pId] = (newPos[pId] - oldPos[pId]) / dt;
	}

	template <typename Real, typename Coord>
	__global__ void VC_UpdateVelocityBoundaryCorrectedTmp(
		DeviceArray<Real> pressure,
		DeviceArray<Real> alpha,
		DeviceArray<bool> bSurface,
		DeviceArray<Coord> position,
		DeviceArray<Coord> velocity,
		DeviceArray<Coord> velocityTri,
		DeviceArray<TopologyModule::Triangle> m_triangle_index,
		DeviceArray<Coord> positionTri,
		DeviceArray<Attribute> attribute,
		DeviceArray<Real> mass,
		DeviceArray<Real> m_triangle_vertex_mass,
		NeighborList<int> neighbor,
		NeighborList<int> neighborTri,
		Real restDensity,
		Real airPressure,
		Real sliding,
		Real separation,
		Real smoothingLength,
		Real m_sampling_distance,
		Real dt,
		DeviceArray<int> flip)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		//printf("%d\n", position.size());

		if (attribute[pId].IsDynamic())
		{
			Attribute att_i = attribute[pId];
			
			Coord pos_i = position[pId];
			Real p_i = pressure[pId];

			int nbSize = neighbor.getNeighborSize(pId);

			int nbSizeTri;
			//			if (neighbor2 != NULL)
			nbSizeTri = neighborTri.getNeighborSize(pId);
			Real total_weight = 0.0f;

			Real ceo = 1.6f;

			Real invAlpha = 1.0f / alpha[pId];
			Real invAlpha_i = invAlpha;

			Real mass_i = mass[pId];
			
			Real mass_i0 = mass_i;
			Coord vel_i = velocity[pId];
			Coord dv_i(0.0f);
			Real scale = 1.0f * dt / restDensity;
			Real acuP = 0.0f;
			total_weight = 0.0f;


			Real sum_weight_norm = Real(0);
			

			bool tmp = false;
			for (int ne = 0; ne < nbSizeTri; ne++)
			{
				int j = neighborTri.getElement(pId, ne);
				//if (j >= 0) continue;
				//j *= -1; j--;

				Triangle3D t3d(positionTri[m_triangle_index[j][0]], positionTri[m_triangle_index[j][1]], positionTri[m_triangle_index[j][2]]);
				Plane3D PL(positionTri[m_triangle_index[j][0]], t3d.normal());
				Point3D p3d(pos_i);
				Point3D nearest_ptt = p3d.project(t3d);
				Point3D nearest_pt = p3d.project(PL);
				Point3D plnpt = p3d.project(PL);
				Real r = (nearest_pt.origin - pos_i).norm();

				if (r < EPSILON) continue;
				float d = p3d.distance(PL);
				//if (d < 0) continue;
				d = abs(d);

				Real AreaSum = p3d.areaTriangle(t3d, smoothingLength);
				Real MinDistance = abs(p3d.distance(t3d));
				Coord Min_Pt = (p3d.project(t3d)).origin - pos_i;
				if (ne < nbSizeTri - 1 )
				{
					int jn;
					do
					{
						jn = neighborTri.getElement(pId, ne + 1);
					//	if (jn >= 0) break;
					//	jn *= -1; jn--;

						Triangle3D t3d_n(positionTri[m_triangle_index[jn][0]], positionTri[m_triangle_index[jn][1]], positionTri[m_triangle_index[jn][2]]);
						if ((t3d.normal().cross(t3d_n.normal())).norm() > EPSILON) break;

						AreaSum += p3d.areaTriangle(t3d_n, smoothingLength);

						if (abs(p3d.distance(t3d_n)) < MinDistance)
						{
							MinDistance = abs(p3d.distance(t3d_n));
							Min_Pt = (p3d.project(t3d_n)).origin - pos_i;
						}
						//printf("%d %d\n", j, jn);
						ne++;
					} while (ne < nbSizeTri - 1);
				}
				Min_Pt /= Min_Pt.norm();

				//Coord n_PL = nearest_pt.origin - pos_i;
				Coord n_PL = - t3d.normal();
				//if (flip[pId] < 0)  n_PL *= -1;
				Coord n_TR = (p3d.project(t3d)).origin - pos_i;
				n_PL = n_PL / n_PL.norm();
				n_TR = n_TR / n_TR.norm();


			//	if (pos_i[2] > 0.98 && pos_i[0] < 0.9)
			//		printf("%.3lf %.3lf %.3lf %d\n", t3d.normal()[0], t3d.normal()[1], t3d.normal()[2], flip[pId]);


				Real weight = -invAlpha * kernWRMesh(r, smoothingLength)
					* 2.0 * (M_PI) * (1 - d / smoothingLength)
					//* p3d.areaTriangle(t3d, smoothingLength) * n_PL.dot(n_TR)
					* AreaSum * n_PL.dot(Min_Pt)
					/ ((M_PI) * (smoothingLength * smoothingLength - d * d))
					/ (m_sampling_distance * m_sampling_distance * m_sampling_distance);

				Coord dnij = (pos_i - nearest_pt.origin) * (1.0f / r);
				if (r < EPSILON)
					dnij = t3d.normal().normalize();
				

				Coord corrected = dnij;
				if (corrected.norm() > EPSILON)
				{
					corrected = corrected.normalize();
				}

				Real mass_j = m_triangle_vertex_mass[m_triangle_index[j][0]];
				corrected = -scale * weight * corrected / (mass_i);
				d = p3d.distance(PL);
				//if (d < 0) continue;
				d = abs(d);
				if (smoothingLength - d > EPSILON&& smoothingLength* smoothingLength - d * d > EPSILON&& d > EPSILON)
				{
					Coord dvii = 2.0f * (pressure[pId]) * corrected;
					if (bSurface[pId])
					{
						dv_i += dvii;
					}
					float weight = 2.0f * invAlpha
						* kernWeightMesh(r, smoothingLength)
						* 2.0 * (M_PI) * (1 - d / smoothingLength)
						* AreaSum * n_PL.dot(Min_Pt)
						/ ((M_PI) * (smoothingLength * smoothingLength - d * d))
						/ (m_sampling_distance * m_sampling_distance * m_sampling_distance);
					
					
					Coord nij = (pos_i - nearest_pt.origin);
					if (nij.norm() > EPSILON)
					{
						nij = nij.normalize();
					}
					else
						nij = t3d.normal().normalize();

					Coord normal_j = t3d.normal();
					normal_j = normal_j.normalize();
					//if (flip[pId] < 0) normal_j *= -1;

					

					Coord Velj = Coord(0);//(velocityTri[m_triangle_index[j][0]] + velocityTri[m_triangle_index[j][1]] + velocityTri[m_triangle_index[j][2]]) / 3.0;

					Coord dVel = Velj - vel_i;//!!!!!!!!!!
					Real magNVel = dVel.dot(normal_j);
					Coord nVel = magNVel * normal_j;
					Coord tVel = dVel - nVel;

					//if (pos_i[2] > 0.98 && pos_i[0] < 0.9 && pos_i[2] < 1.0)
					//			printf("%.3lf %.3lf %.3lf %.3lf\n", nij[0], nij[1], nij[2], weight);

					if (magNVel > EPSILON)
					{
						dv_i += weight * nij.dot(nVel + sliding * tVel) * nij;
						//dv_i += weight * (nVel + sliding * tVel);
					}
					else
					{
						dv_i += weight * nij.dot(separation * nVel + sliding * tVel) * nij;
						//dv_i += weight * (separation * nVel + sliding * tVel);
					}
				}

			}
			for (int ne = 0; ne < nbSize; ne++)
			{
				int j = neighbor.getElement(pId, ne);
				Real r = (pos_i - position[j]).norm();

				Attribute att_j = attribute[j];

				Real mass_j = mass[j];
				


				if (r > EPSILON)
				{
					Real weight = -invAlpha * kernWR(r, smoothingLength);
					Coord dnij = (pos_i - position[j]) * (1.0f / r);
					Coord corrected = dnij;
					if (corrected.norm() > EPSILON)
					{
						corrected = corrected.normalize();
					}

					Real mass_j = mass[j];
					corrected = -scale * weight * corrected / (mass_i);


					Coord dvij = (pressure[j] - pressure[pId]) * corrected;
					Coord dvjj = (pressure[j] + airPressure) * corrected;

					Coord dvij_sym = 0.5f * (pressure[pId] + pressure[j]) * corrected;

					Coord dVel = (velocity[j] - vel_i);

					Real ratio_mass = mass_j / (mass_i + mass_j);
					float weight_m = invAlpha * kernWeight(r, smoothingLength);

						if (att_j.IsDynamic())
						{
							if (bSurface[pId])
									dv_i += dvjj;
							else
									dv_i += dvij;

							if (bSurface[j])
							{
								Coord dvii = -(pressure[pId] + airPressure) * corrected;
								atomicAdd(&velocity[j][0], ceo * dvii[0]);
								atomicAdd(&velocity[j][1], ceo * dvii[1]);
								atomicAdd(&velocity[j][2], ceo * dvii[2]);
							}
							else
							{
								atomicAdd(&velocity[j][0], ceo * dvij[0]);
								atomicAdd(&velocity[j][1], ceo * dvij[1]);
								atomicAdd(&velocity[j][2], ceo * dvij[2]);
							}
						}

						
				}
			}

			dv_i *= ceo;
			if (attribute[pId].IsFluid())
			{
				atomicAdd(&velocity[pId][0], dv_i[0]);
				atomicAdd(&velocity[pId][1], dv_i[1]);
				atomicAdd(&velocity[pId][2], dv_i[2]);
			}
		}


	}


	template<typename TDataType>
	SemiAnalyticalIncompressibilityModule<TDataType>::SemiAnalyticalIncompressibilityModule()
		: ConstraintModule()
		, m_airPressure(Real(00.0))
		, m_reduce(NULL)
		, m_arithmetic(NULL)
	{
		m_smoothing_length.setValue(Real(0.011));

		attachField(&m_smoothing_length, "smoothing_length", "The smoothing length in SPH!", false);

		attachField(&m_particle_position, "position", "Storing the particle positions!", false);
		attachField(&m_particle_velocity, "velocity", "Storing the particle velocities!", false);
//		attachField(&m_particle_normal, "normal", "Storing the particle normals!", false);
		attachField(&m_particle_attribute, "attribute", "Storing the particle attributes!", false);
		attachField(&m_neighborhood_particles, "neighborhood", "Storing neighboring particles' ids!", false);
	}

	template<typename TDataType>
	SemiAnalyticalIncompressibilityModule<TDataType>::~SemiAnalyticalIncompressibilityModule()
	{
		m_alpha.release();
		m_Aii.release();
		m_AiiFluid.release();
		m_AiiTotal.release();
		m_pressure.release();
		//m_pressure2.release();
		m_divergence.release();
		m_bSurface.release();

		m_y.release();
		m_r.release();
		m_p.release();

		//m_pressure.release();

		if (m_reduce)
		{
			delete m_reduce;
		}
		if (m_arithmetic)
		{
			delete m_arithmetic;
		}
	}

	template<typename TDataType>
	bool SemiAnalyticalIncompressibilityModule<TDataType>::constrain()
	{
		//return true; 

		Real dt = getParent()->getDt();

//		int start_f = Start.getValue();
//		cudaMemcpy(m_velocityAll.getValue().getDataPtr() + start_f, m_particle_velocity.getValue().getDataPtr(), num_f * sizeof(Coord), cudaMemcpyDeviceToDevice);

		std::cout << "Element Count: " << m_particle_position.getElementCount() << std::endl;

		uint pDims = cudaGridSize(m_particle_position.getElementCount(), BLOCK_SIZE);

		//compute alpha_i = sigma w_j and A_i = sigma w_ij / r_ij / r_ij
		
		printf("inside VC constraint NEW %d %d %d %d\n", m_particle_velocity.getValue().size(),m_particle_mass.getElementCount(), m_particle_attribute.getElementCount() , m_particle_position.getElementCount());

		int numTri = m_triangle_vertex.getElementCount();
		uint pDimsT = cudaGridSize(numTri, BLOCK_SIZE);

		if (!m_particle_position.isEmpty())
		{
			printf("warning from second step!");
			int num = m_particle_position.getElementCount();
			uint pDims = cudaGridSize(num, BLOCK_SIZE);

			//m_particle_attribute.setElementCount(num);
			//m_particle_mass.setElementCount(num);

			//			m_particle_normal.setElementCount(num);
						//if (m_particle_attribute.isEmpty())printf("???\n");
			//m_particle_attribute.getReference()->reset();
			//VC_InitAttrTmp << <pDims, BLOCK_SIZE >> > (
			//	m_particle_attribute.getValue(),
			//	m_particle_mass.getValue()
			//	);

		}

		int num = m_particle_position.getElementCount();
		if (m_AiiFluid.isEmpty() || m_AiiFluid.size() != num)
		{
			//printf("RESIZW\n");
			m_alpha.resize(num);
			Rho_alpha.resize(num);
			m_Aii.resize(num);
			m_AiiFluid.resize(num);
			m_AiiTotal.resize(num);
			m_pressure.resize(num);
			m_divergence.resize(num);
			m_bSurface.resize(num);
			//m_density.resize(num);

			m_y.resize(num);
			m_r.resize(num);
			m_p.resize(num);

			m_flip.setElementCount(num);

			m_reduce = Reduction<float>::Create(num);
			m_arithmetic = Arithmetic<float>::Create(num);
		}


		m_flip.getValue().reset();
		m_alpha.reset();
		//printf("sampling_distance = %.10lf; smoothing_length = %.10lf\n", m_sampling_distance.getValue(), m_smoothing_length.getValue());

		VC_TriVelTmp << <pDimsT, BLOCK_SIZE >> > (
			m_triangle_vertex_old.getValue(),
			m_triangle_vertex.getValue(),
			m_meshVel,
			dt
			);
		



		VC_Sort_Neighbors << <pDims, BLOCK_SIZE >> > (
			m_particle_position.getValue(),
			m_triangle_index.getValue(),
			m_triangle_vertex.getValue(),
			m_neighborhood_triangles.getValue()
			);
			
		VC_ComputeAlphaTmp << <pDims, BLOCK_SIZE >> > (
			m_alpha,
			Rho_alpha,
			m_particle_mass.getValue(),
			m_particle_position.getValue(),
			m_triangle_index.getValue(),
			m_triangle_vertex.getValue(),
			m_particle_attribute.getValue(),
			m_neighborhood_particles.getValue(),
			m_neighborhood_triangles.getValue(),
			m_smoothing_length.getValue(),
			m_sampling_distance.getValue(),
			m_flip.getValue());
		cuSynchronize();

		//Real m_maxAlpha2 = m_reduce->maximum(m_alpha.getDataPtr(), m_alpha.size());
		//m_maxAlpha = max(m_maxAlpha2, m_maxAlpha);

		VC_CorrectAlphaTmp << <pDims, BLOCK_SIZE >> > (
			m_alpha,
			Rho_alpha,
			m_particle_mass.getValue(),
			m_maxAlpha);

		
		//compute the diagonal elements of the coefficient matrix
		m_AiiFluid.reset();
		m_AiiTotal.reset();
		VC_ComputeDiagonalElementTmp << <pDims, BLOCK_SIZE >> > (
			m_AiiFluid,
			m_AiiTotal,
			m_alpha,
			m_particle_position.getValue(),
			m_triangle_index.getValue(),
			m_triangle_vertex.getValue(),
			m_particle_attribute.getValue(),
			m_neighborhood_particles.getValue(),
			m_neighborhood_triangles.getValue(),
			m_smoothing_length.getValue(),
			m_sampling_distance.getValue(),
			m_flip.getValue());

		m_bSurface.reset();
		m_Aii.reset();
		VC_DetectSurfaceTmp << <pDims, BLOCK_SIZE >> > (
			m_Aii,
			m_bSurface,
			m_AiiFluid,
			m_AiiTotal,
			m_particle_position.getValue(),
			m_particle_attribute.getValue(),
			m_neighborhood_particles.getValue(),
			m_smoothing_length.getValue(),
			m_maxA);

		int itor = 0;

		//compute the source term
		//m_densitySum->compute(m_density);
		m_densitySum->compute();
		//m_density = m_densitySum->outDensity()->getValue();

		m_divergence.reset();
		VC_ComputeDivergenceTmp << <pDims, BLOCK_SIZE >> > (
			m_divergence,
			m_alpha,
			m_densitySum->outDensity()->getValue(),
			m_particle_position.getValue(),
			m_particle_velocity.getValue(),
			m_meshVel,
			m_triangle_index.getValue(),
			m_triangle_vertex.getValue(),
			m_bSurface,
			m_particle_attribute.getValue(),
			m_particle_mass.getValue(),
			m_triangle_vertex_mass.getValue(),
			m_neighborhood_particles.getValue(),
			m_neighborhood_triangles.getValue(),
			m_separation,
			m_tangential,
			m_restDensity,
			m_smoothing_length.getValue(),
			m_sampling_distance.getValue(),
			dt,
			m_flip.getValue());
		
		VC_CompensateSourceTmp << <pDims, BLOCK_SIZE >> > ( // no need
			m_divergence,
			m_densitySum->outDensity()->getValue(),
			m_particle_attribute.getValue(),
			m_particle_position.getValue(),
			m_restDensity,
			dt);

		//solve the linear system of equations with a conjugate gradient method.
		m_y.reset();
		m_pressure.reset();
		VC_ComputeAxTmp << <pDims, BLOCK_SIZE >> > (
			m_y,
			m_pressure,
			m_Aii,
			m_alpha,
			m_particle_position.getValue(),
			m_particle_attribute.getValue(),
			m_neighborhood_particles.getValue(),
			m_smoothing_length.getValue());

		m_r.reset();
		Function2Pt::subtract(m_r, m_divergence, m_y);
		Function1Pt::copy(m_p, m_r);
		Real rr = m_arithmetic->Dot(m_r, m_r);
		Real err = sqrt(rr / m_r.size());

		printf("ERROR: %.10lf\n",err);

		while (itor < 1000 && err > 1.0f)
		{
			m_y.reset();
			//VC_ComputeAx << <pDims, BLOCK_SIZE >> > (*yArr, *pArr, *aiiArr, *alphaArr, *posArr, *attArr, *neighborArr);
			VC_ComputeAxTmp << <pDims, BLOCK_SIZE >> > (
				m_y,
				m_p,
				m_Aii,
				m_alpha,
				m_particle_position.getValue(),
				m_particle_attribute.getValue(),
				m_neighborhood_particles.getValue(),
				m_smoothing_length.getValue());

			float alpha = rr / m_arithmetic->Dot(m_p, m_y);
			Function2Pt::saxpy(m_pressure, m_p, m_pressure, alpha);
			Function2Pt::saxpy(m_r, m_y, m_r, -alpha);

			Real rr_old = rr;

			rr = m_arithmetic->Dot(m_r, m_r);

			Real beta = rr / rr_old;
			Function2Pt::saxpy(m_p, m_p, m_r, beta);

			err = sqrt(rr / m_r.size());
			printf("err: %.3lf\n:", err);
			itor++;
		}
		//return true;
		//update the each particle's velocity
		VC_UpdateVelocityBoundaryCorrectedTmp << <pDims, BLOCK_SIZE >> > (
			m_pressure,
			m_alpha,
			m_bSurface,
			m_particle_position.getValue(),
			m_particle_velocity.getValue(),
			m_meshVel,
			m_triangle_index.getValue(),
			m_triangle_vertex.getValue(),
			m_particle_attribute.getValue(),
			m_particle_mass.getValue(),
			m_triangle_vertex_mass.getValue(),
			m_neighborhood_particles.getValue(),
			m_neighborhood_triangles.getValue(),
			m_restDensity,
			m_airPressure,
			m_tangential,
			m_separation,
			m_smoothing_length.getValue(),
			m_sampling_distance.getValue(),
			dt,
			m_flip.getValue());

		//printf("pressure size: %d\n", m_pressure.size());
		//cudaMemcpy(m_pressure2.getValue().getDataPtr(), m_pressure.getDataPtr() + start_f, num_f * sizeof(Real), cudaMemcpyDeviceToDevice);
//		cudaMemcpy(m_particle_velocity.getValue().getDataPtr(), m_velocityAll.getValue().getDataPtr() + start_f, num_f * sizeof(Coord), cudaMemcpyDeviceToDevice);
//		cudaMemcpy(PressureFluid.getValue().getDataPtr(), m_pressure.getDataPtr() + start_f, num_f * sizeof(Real), cudaMemcpyDeviceToDevice);
		//cudaMemcpy(PressureFluid.getValue().getDataPtr(), m_divergence_Tri.getDataPtr() + start_f, num_f * sizeof(Real), cudaMemcpyDeviceToDevice);

		return true;
	}


	template<typename TDataType>
	bool SemiAnalyticalIncompressibilityModule<TDataType>::initializeImpl()
	{
		first_step = true;

		if (m_particle_position.isEmpty())
			printf("OKKKKKKKKKK\n");
		m_sampling_distance.setValue(0.005);

		//		printf("%d %d %d %d\n", m_position.getElementCount(), m_position1.getElementCount(), m_velocity.getElementCount(), m_velocity2.getElementCount());


		if (!m_particle_position.isEmpty())
		{
			printf("warning from second step!");
			int num = m_particle_position.getElementCount();
			uint pDims = cudaGridSize(num, BLOCK_SIZE);

			m_particle_attribute.setElementCount(num);
			m_particle_mass.setElementCount(num);

			//			m_particle_normal.setElementCount(num);
						//if (m_particle_attribute.isEmpty())printf("???\n");
			m_particle_attribute.getReference()->reset();
			VC_InitAttrTmp << <pDims, BLOCK_SIZE >> > (
				m_particle_attribute.getValue(),
				m_particle_mass.getValue()
				);

		}
		else
		{
			printf("YES~ m_triangle_index Size: %d\n", m_triangle_index.getElementCount());
		}
		Real dt = getParent()->getDt();
		int numt = m_triangle_vertex.getElementCount();

		m_meshVel.resize(numt);
		uint pDims = cudaGridSize(numt, BLOCK_SIZE);

		VC_TriVelTmp << <pDims, BLOCK_SIZE >> > (
			m_triangle_vertex_old.getValue(),
			m_triangle_vertex.getValue(),
			m_meshVel,
			dt
			);


		//m_neighborhood->initialize();
		/*
		m_densitySum = std::make_shared<DensitySummation<TDataType>>();
		m_smoothing_length.connect(&m_densitySum->m_smoothingLength);
		m_particle_position.connect(&m_densitySum->m_position);
		m_neighborhood_particles.connect(&m_densitySum->m_neighborhood);
		m_densitySum->initialize();
		*/

		m_densitySum = std::make_shared<SummationDensity<TDataType>>();
		m_smoothing_length.connect(m_densitySum->varSmoothingLength());
		m_particle_position.connect(m_densitySum->inPosition());
		m_neighborhood_particles.connect(m_densitySum->inNeighborIndex());
		m_densitySum->initialize();

		int num = m_particle_position.getElementCount();

		if (num == 0)
		{
			m_maxAlpha = 36.8;
			m_maxA = 35205.6;
			return true;
		}

		num_f = m_particle_position.getElementCount();
		//int start_f = 11286;//5130;

		m_alpha.resize(num);
		Rho_alpha.resize(num);
		m_Aii.resize(num);
		m_AiiFluid.resize(num);
		m_AiiTotal.resize(num);
		m_pressure.resize(num);
		m_divergence.resize(num);
		//m_divergence_Tri.resize(num);
		m_bSurface.resize(num);
		m_density.resize(num);

		m_y.resize(num);
		m_r.resize(num);
		m_p.resize(num);

		//	m_pressure.resize(num);

		m_reduce = Reduction<float>::Create(num);
		m_arithmetic = Arithmetic<float>::Create(num);


		pDims = cudaGridSize(num, BLOCK_SIZE);
/*
		VC_Sort_Neighbors << <pDims, BLOCK_SIZE >> > (
			m_particle_position.getValue(),
			m_triangle_index.getValue(),
			m_triangle_vertex.getValue(),
			m_neighborhood_triangles.getValue()
			);

		VC_Calc_SolidAngle << <pDims, BLOCK_SIZE >> > (
			m_particle_position.getValue(),
			m_triangle_index.getValue(),
			m_triangle_vertex.getValue(),
			m_neighborhood_triangles.getValue(),
			m_smoothing_length.getValue()
		);
*/

		//printf("TRI:%d\n", m_triangle_index.getValue().ge);
//		printf("NEI1:%d\n", m_neighborhood.isEmpty());
		m_alpha.reset();
		printf("FLIP: %d\n", m_flip.getValue().size());
		VC_ComputeAlphaTmp << <pDims, BLOCK_SIZE >> > (
			m_alpha,
			Rho_alpha,
			m_particle_mass.getValue(),
			m_particle_position.getValue(),
			m_triangle_index.getValue(),
			m_triangle_vertex.getValue(),
			m_particle_attribute.getValue(),
			m_neighborhood_particles.getValue(),
			m_neighborhood_triangles.getValue(),
			m_smoothing_length.getValue(),
			m_sampling_distance.getValue(),
			m_flip.getValue()
			);

		m_maxAlpha = m_reduce->maximum(m_alpha.begin(), m_alpha.size());

		VC_CorrectAlphaTmp << <pDims, BLOCK_SIZE >> > (
			m_alpha,
			Rho_alpha,
			m_particle_mass.getValue(),
			m_maxAlpha);

		m_AiiFluid.reset();
		VC_ComputeDiagonalElementTmp << <pDims, BLOCK_SIZE >> > (
			m_AiiFluid,
			m_alpha,
			m_particle_position.getValue(),
			m_particle_attribute.getValue(),
			m_neighborhood_particles.getValue(),
			m_smoothing_length.getValue());

		m_maxA = m_reduce->maximum(m_AiiFluid.begin(), m_AiiFluid.size());

		std::cout << "Max alpha: " << m_maxAlpha << std::endl;
		printf("%.10lf\n", m_maxAlpha);
		std::cout << "Max A: " << m_maxA << std::endl;


		return true;
	}
}