#include <cuda_runtime.h>
#include "UnifiedVelocityConstraint.h"
#include "Framework/Node.h"
#include "Utility.h"
#include "SummationDensity.h"
#include "DensitySummationMesh.h"
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



	template <typename Real, typename Coord>
	__global__ void UVC_ComputeAlphaTmp
	(
		GArray<Real> alpha,
		GArray<Coord> position,
		GArray<Attribute> attribute,
		NeighborList<int> neighbors,
		Real smoothingLength
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (pId >= position.size()) return;
		if (!attribute[pId].IsDynamic()) return;

		Coord pos_i = position[pId];
		Real alpha_i = 0.0f;
		Real ra = 0.0f;
		int nbSize = neighbors.getNeighborSize(pId);
		

		Real alpha_solid = alpha_i;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			Real r = (pos_i - position[j]).norm();;

			//if (r > EPSILON)
			if (r > EPSILON && attribute[j].IsDynamic())
			{
				Real a_ij = kernWeight(r, smoothingLength);
				alpha_i += a_ij;
			}
			else if (r > EPSILON)
			{
				Real a_ij = kernWeight(r, smoothingLength);
				
			}
		}

		alpha[pId] = alpha_i;
	}

	template <typename Real>
	__global__ void UVC_CorrectAlphaTmp
	(
		GArray<Real> alpha,
		GArray<Real> rho_alpha,
		GArray<Real> mass,
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
	__global__ void UVC_ComputeDivergence
	(
		GArray<Real> divergence,
		GArray<Real> density,
		GArray<int> mapping,
		GArray<Real> alpha,
		GArray<Coord> position,
		GArray<Coord> velocity,
		GArray<Attribute> attribute,
		NeighborList<int> neighbors,
		Real restDensity,
		Real smoothingLength,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (pId >= neighbors.getElementSize()) return;

		int i = mapping[pId];
		int offset = pId - neighbors.getElementIndex(i, 0);
		int j = neighbors.getElement(i, offset);

		if (i >= j) return;
		Coord pos_i = position[i];
		Coord pos_j = position[j];

		Real r0 = (pos_i - pos_j).norm();
		if (r0 < EPSILON)
			r0 = 1.0f;
		Coord n_ji = (pos_i - pos_j) / r0;


		Coord vel_i = Coord(0);
		Coord vel_j = Coord(0);

		int nbr_i = neighbors.getNeighborSize(i);
		int nbr_j = neighbors.getNeighborSize(j);

		vel_j = velocity[j];
		vel_i = velocity[i];
		Real div = (vel_j - vel_i).dot(n_ji) * restDensity / dt;

		divergence[pId] += div;// *kernWeight(r0, smoothingLength) * (1.0 / alpha[pId] + 1.0 / alpha[j]) / 2.0f;

		if (density[i] > restDensity)
		{
			Real ratio = (density[i] - restDensity) / restDensity;
			atomicAdd(&divergence[pId], 100000 * ratio);
		}
		if (density[j] > restDensity)
		{
			Real ratio = (density[j] - restDensity) / restDensity;
			atomicAdd(&divergence[pId], 100000 * ratio);
		}

	}

	template <typename Real, typename Coord>
	__global__ void UVC_CompensateSourceTmp
	(
		GArray<Real> divergence,
		GArray<Real> density,
		GArray<Attribute> attribute,
		GArray<Coord> position,
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
			atomicAdd(&divergence[pId], 100000.0f * ratio / dt);
		}
	}

	

	template <typename Real, typename Coord>
	__global__ void UVC_ComputeGradient
	(
		GArray<Coord> velocity_inside_iteration,
		GArray<Real> gradient,
		GArray<int> index_sym,
		GArray<Real> density,
		GArray<Real> force,
		GArray<Real> alpha,
		GArray<Coord> position,
		GArray<Attribute> attribute,
		NeighborList<int> neighbor,
		GArray<int> mapping,
		Real smoothingLength,
		Real dt,
		Real restDensity
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= neighbor.getElementSize()) return;


		Real sampling_distance = 0.005f;
		int i = mapping[pId];
		int offset = pId - neighbor.getElementIndex(i, 0);
		//printf("OFFSESET: %d\n", offset);
		int j = neighbor.getElement(i, offset);

		if (i >= j)
		{
			return;
		}

		Coord pos_i = position[i];
		Coord pos_j = position[j];

		Real r0 = (pos_i - pos_j).norm();
		if (r0 < EPSILON)
			r0 = 1.0f;
		Real invAlpha_i = 1.0f / alpha[i];
		Real invAlpha_j = 1.0f / alpha[j];

		//Real scale = 1.0;//dt / restDensity;

		Real mass_i = restDensity * pow(sampling_distance, 3);
		Real scale = 1.0f * dt * mass_i;
		//printf("mass fluid: %.13lf\n", mass_i);

		//Real w0_ij = kernWeight(r0, smoothingLength);
		//scale *= scale;

		Coord tmp_vel_i = velocity_inside_iteration[i];
		Coord tmp_vel_j = velocity_inside_iteration[j];

		Coord n_ji = (pos_i - pos_j) / r0;

		atomicAdd(&gradient[pId], (tmp_vel_i.dot(n_ji) + tmp_vel_j.dot(-n_ji)) * scale);

		if (density[i] > restDensity)
		{
			Real ratio = (density[i] - restDensity) / restDensity;

		//	printf("ratio: %.13lf\n", ratio);

			//atomicAdd(&gradient[pId], -0.5f * ratio * scale);
		}
		if (density[j] > restDensity)
		{
			Real ratio = (density[j] - restDensity) / restDensity;

			//printf("ratio: %.13lf\n", ratio);

			//atomicAdd(&gradient[pId], -0.5f * ratio * scale);
		}
		//printf("%.13lf\n", gradient[pId]);

		
	}



	template <typename Real>
	__global__ void UVC_NormalizeGradientPoint(
		GArray<Real> gradient_point,
		GArray<Real> invRadius,
		GArray<Real> density
		)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= invRadius.size()) return;

		if(invRadius[pId] > EPSILON)
		gradient_point[pId] /= invRadius[pId];

		Real restDensity = 1000.0f;
		Real sampling_distance = 0.005f;
		Real dt = 0.001f;
		Real mass_i = restDensity * pow(sampling_distance, 3);
		Real scale = 1.0f * dt * mass_i;

		if (density[pId] > restDensity)
		{
			Real ratio = (density[pId] - restDensity) / restDensity;

			//	printf("ratio: %.13lf\n", ratio);

			gradient_point[pId] -= 50.0f * ratio * scale;
		}
	}

	template <typename Real, typename Coord>
	__global__ void UVC_ComputeGradientPoint(
		GArray<Real> alpha,
		GArray<int> index_sym,
		GArray<Real> invRadius,
		GArray<Real> gradient_edge,
		GArray<Real> gradient_point,
		GArray<Real> pressure_point,
		GArray<Coord> position,
		GArray<Attribute> attribute,
		NeighborList<int> neighbor,
		Real restDensity,
		Real airPressure,
		Real smoothingLength,
		Real m_sampling_distance,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;


		Real invAlpha_i = 1.0 / alpha[pId];
		Real scale = 0.5f * dt / restDensity;
		int nbSize = neighbor.getNeighborSize(pId);

		Real gradient_sum(0);
		Real summ = 0.0f;

		Real invRadiusSum(0);

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbor.getElement(pId, ne);
			if (j == pId) continue;
			Real r = (position[j] - position[pId]).norm();
			Real w_ij = kernWeight(r, smoothingLength);
			Real a_ij = 0.5f * (invAlpha_i + 1.0 / alpha[j]) * w_ij;
			int idx1 = min(pId, j);
			//int idx2 = max(pId, j);
			int idx;
			Coord n_ji = (position[j] - position[pId]) / r;
			if (idx1 == pId)
			{
				idx = neighbor.getElementIndex(pId, ne);
			}
			else
			{
				idx = index_sym[neighbor.getElementIndex(pId, ne)];
			}
			gradient_sum += gradient_edge[idx] * 1.0 / r;// * a_ij;
			if (r > EPSILON)
				invRadiusSum += 1.0;
			else
				invRadiusSum += 1.0;
			//summ += 1.0f;
		}

		gradient_point[pId] = gradient_sum; // summ;
		pressure_point[pId] -= gradient_sum; // for visualize debug
		//if(pId % 10000 == 0)
			//printf("PP: %.10lf\n", pressure_point[pId]);
		invRadius[pId] = invRadiusSum;

		

	}

	template <typename Real>
	__global__ void UVC_ComputeGradientPointRigid(
		GArray<Real> invRadius,
		GArray<Real> gradient_edge,
		GArray<Real> gradient_point,
		GArray<Real> density,
		GArray<NeighborConstraints> nbc,
		Real restDensity,
		Real smoothingLength,
		Real m_sampling_distance,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nbc.size()) return;
		return;
		
		
		Coord3D n_ji = nbc[pId].pos2 - nbc[pId].pos1;
		Real r = n_ji.norm();
		if (r < EPSILON) r = 1.0f;
		
		atomicAdd(&gradient_point[nbc[pId].idx1], gradient_edge[pId] * 1.0f / r);
		atomicAdd(&invRadius[nbc[pId].idx1], 1.0);

		//SpikyKernel<Real> kern;
		//Real den = kern.Weight(r, smoothingLength) * 0.0000721597f;
		//atomicAdd(&density[nbc[pId].idx1], den);

	}

	template <typename Real>
	__global__ void UVC_CompensateRhoRigid(
		GArray<Real> invRadius,
		GArray<Real> gradient_edge,
		GArray<Real> gradient_point,
		GArray<Real> density,
		GArray<NeighborConstraints> nbc,
		Real restDensity,
		Real smoothingLength,
		Real m_sampling_distance,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nbc.size()) return;
		//return;


		Coord3D n_ji = nbc[pId].pos2 - nbc[pId].pos1;
		Real r = n_ji.norm();
		if (r < EPSILON) r = 1.0f;


		SpikyKernel<Real> kern;
		Real den = kern.Weight(r, smoothingLength) * 0.0000721597f;
		atomicAdd(&density[nbc[pId].idx1], den);

	}

	template <typename Real, typename Coord>
	__global__ void UVC_UpdateGradient
	(
		GArray<Real> gradient_point,
		GArray<Real> gradient_edge,
		GArray<Real> force,
		GArray<Coord> vel_tmp,
		GArray<Real> alpha,
		GArray<Real> invRadius,
		GArray<Coord> position,
		GArray<Attribute> attribute,
		NeighborList<int> neighbor,
		GArray<int> mapping,
		Real smoothingLength,
		Real dt,
		Real restDensity,
		Real step_i
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= neighbor.getElementSize()) return;

		int i = mapping[pId];
		int offset = pId - neighbor.getElementIndex(i, 0);
		//printf("OFFSESET: %d\n", offset);
		int j = neighbor.getElement(i, offset);
		if (i >= j)
			return;


		Coord pos_i = position[i];
		Coord pos_j = position[j];



		Real r0 = (pos_i - pos_j).norm();
		if (r0 < EPSILON)
			r0 = 1.0f;
		Real invAlpha_ij = 1.0f;
		//Real invAlpha_j = 1.0f / invRadius[j];

		Real scale = 1.0;//dt / restDensity;

		Real w0_ij = kernWeight(r0, smoothingLength);
		//scale *= scale;

		int nbSize = neighbor.getNeighborSize(i);

		Coord n_ji = (pos_i - pos_j) / r0;

		//Real a_ij = (invAlpha_i + invAlpha_j) * (1.0f / r0);//w0_ij;


		//for debug
		//Real mass_i = restDensity * pow(0.005,3);
		//if(gradient_point[i] * step * invAlpha_ij * (1.0f / r0) / mass_i > 0.1f)
		//printf("gradient change: %.13lf\n", gradient_point[i] * step * invAlpha_ij * (1.0f / r0) / mass_i);

		//Real step11 = pow((vel_tmp[j] - vel_tmp[i]).dot(n_ji), 2) / pow(((gradient_point[i] + gradient_point[j]) / r0), 2) * pow(dt, 2) * 10;
		//if (abs((gradient_point[i] + gradient_point[j])) < 0.0000000001) step11 = 0.0f;

		//Real step22 = pow((vel_tmp[j] - vel_tmp[i]).dot(n_ji), 2) / gradient_point[j] / gradient_point[j] * pow(dt, 4);
		//if (abs(gradient_point[j]) < 0.0000000001) step22 = 0.0f;

		//if(pId % 5000 == 0)
		//	printf("========= step: %.20lf\n", step11);

		force[pId] -= 1.0f * gradient_point[i] * step_i * invAlpha_ij * (1.0f / r0);
		force[pId] -= 1.0f * gradient_point[j] * step_i * invAlpha_ij * (1.0f / r0);

		//force[pId] -= 1.0f * gradient_edge[pId] * step_i * (1.0f / 0.005);

	}



	template <typename Real>
	__global__ void UVC_InitAttrTmp(
		GArray<Attribute> attribute,
		GArray<Real> mass
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= attribute.size()) return;
		attribute[pId].SetDynamic();
		attribute[pId].SetFluid();
		mass[pId] = 10.0;
	}

	

	__global__ void UVC_initialize_mapping(
		NeighborList<int> nbr,
		GArray<int> mapping
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nbr.size()) return;

		int nbSize = nbr.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbr.getElementIndex(pId, ne);
			mapping[j] = pId;
		}
	}

	__global__ void UVC_initialize_sym_mapping(
		NeighborList<int> nbr,
		GArray<int> mapping,
		GArray<int> mapping_sym
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= mapping_sym.size()) return;

		int i = mapping[pId];
		int offset = pId - nbr.getElementIndex(i, 0);
		int j = nbr.getElement(i, offset);
		

		int nbSizeJ = nbr.getNeighborSize(j);
		int idx = -1;
		for (int ne2 = 0; ne2 < nbSizeJ; ne2++)
		{
			if (nbr.getElement(j, ne2) == i)
			{
				idx = nbr.getElementIndex(j, ne2);
				break;
			}
		}

		if (idx == -1)
		{
			idx = pId;
		}
		mapping_sym[pId] = idx;
		

	}
	
	template <typename Real, typename Coord>
	__global__ void UVC_UpdateVelocityParticles(
		GArray<Real> force,
		GArray<int> index_sym,
		GArray<Real> alpha,
		GArray<Coord> position,
		GArray<Coord> velocity_old,
		GArray<Coord> velocity,
		GArray<Attribute> attribute,
		GArray<Real> density,
		NeighborList<int> neighbor,
		Real restDensity,
		Real smoothingLength,
		Real m_sampling_distance,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		//printf("%d\n", position.size());

		Real invAlpha_i = 1.0 / alpha[pId];
		//Real factor
		Real mass_i = restDensity * pow(m_sampling_distance, 3);
	//	printf("mass_i= %.10lf\n", mass_i);
		Real scale = 2.0f * dt / mass_i;
		int nbSize = neighbor.getNeighborSize(pId);
		velocity[pId] = velocity_old[pId];

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbor.getElement(pId, ne);
			if (j == pId) continue;
			Real r = (position[j] - position[pId]).norm();
			if (r < EPSILON) return;
			Real w_ij = kernWeight(r, smoothingLength);
			Real a_ij = 1.0f; //* invAlpha_i * w_ij;
			int idx1 = min(pId, j);
			int idx;
			Coord n_ji = (position[j] - position[pId]) / r;
			if (idx1 == pId)
			{
				idx = neighbor.getElementIndex(pId, ne);
			}
			else
			{
				idx = index_sym[neighbor.getElementIndex(pId, ne)];
			}

			Coord dvji = force[idx] * n_ji * scale * a_ij;

			velocity[pId] += (-dvji);

		}
	}

	template <typename Real, typename Coord>
	__global__ void UVC_UpdateVelocityParticlesRigid(
		GArray<Real> force_interface,
		GArray<Coord> velocity,
		GArray<NeighborConstraints> nbc,
		Real restDensity,
		Real smoothingLength,
		Real m_sampling_distance,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nbc.size()) return;
		
		Real mass_i = restDensity * pow(m_sampling_distance, 3);
		Real scale = 1.0f * dt / mass_i;
		
		Coord n_ji = nbc[pId].pos2 - nbc[pId].pos1;
		Real r = n_ji.norm();
		if (r < EPSILON) r = 1.0f;
		n_ji = n_ji / r;

		Coord dvji = force_interface[pId] * n_ji * scale;

		//printf("fluid norm %.10lf\n", dvji.norm());

		atomicAdd(&velocity[nbc[pId].idx1][0], - dvji[0]);
		atomicAdd(&velocity[nbc[pId].idx1][1], - dvji[1]);
		atomicAdd(&velocity[nbc[pId].idx1][2], - dvji[2]);

		
	}


	template <typename Real, typename Coord>
	__global__ void UVC_UpdateVelocityBoundaryCorrectedTmp(
		GArray<Real> force,
		GArray<int> index_sym,
		GArray<Real> alpha,
		GArray<bool> bSurface,
		GArray<Coord> position,
		GArray<Coord> velocity,
		GArray<Coord> velocity_old,
		GArray<Coord> velocityTri,
		GArray<TopologyModule::Triangle> m_triangle_index,
		GArray<Coord> positionTri,
		GArray<Attribute> attribute,
		GArray<Real> mass,
		GArray<Real> density,
		GArray<Real> m_triangle_vertex_mass,
		NeighborList<int> neighbor,
		NeighborList<int> neighborTri,
		Real restDensity,
		Real airPressure,
		Real sliding,
		Real separation,
		Real smoothingLength,
		Real m_sampling_distance,
		Real dt,
		GArray<int> flip)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		//printf("%d\n", position.size());

		Real invAlpha_i = 1.0 / alpha[pId];
		//Real factor
		Real mass_i = restDensity * pow(m_sampling_distance, 3);
		Real scale = 1.0f * dt / mass_i;
		int nbSize = neighbor.getNeighborSize(pId);



		Real force_sum(0);
		Real factor_sum(0);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbor.getElement(pId, ne);
			if (j == pId) continue;
			Real r = (position[j] - position[pId]).norm();
			Real w_ij = kernWeight(r, smoothingLength);
			Real a_ij = 1.0f; //* invAlpha_i * w_ij;
			int idx1 = min(pId, j);
			//int idx2 = max(pId, j);
			int idx;
			Coord n_ji = (position[j] - position[pId]) / r;
			if (idx1 == pId)
			{
				idx = neighbor.getElementIndex(pId, ne);
			}
			else
			{
				idx = index_sym[neighbor.getElementIndex(pId, ne)];
			}
			if(force[idx] > force_sum)
			{ 
				force_sum += force[idx];
				factor_sum += 1.0;
			}
		}
		if(factor_sum > 0)
			force_sum /= factor_sum;

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbor.getElement(pId, ne);
			if (j == pId) continue;
			Real r = (position[j] - position[pId]).norm();
			Real w_ij = kernWeight(r, smoothingLength);
			Real a_ij = 1.0f; //* invAlpha_i * w_ij;
			int idx1 = min(pId, j);
			//int idx2 = max(pId, j);
			int idx;
			Coord n_ji = (position[j] - position[pId]) / r;
			if (idx1 == pId)
			{
				idx = neighbor.getElementIndex(pId, ne);
			}
			else
			{
				idx = index_sym[neighbor.getElementIndex(pId, ne)];
			}

			Coord dvji = force[idx] * n_ji * scale * a_ij;
		//	if (!(dvji.norm() < 1000.0f))
		//		printf("%.3lf %.3lf %.3lf\n", dvji[0], dvji[1], dvji[2]);

			

			atomicAdd(&velocity[pId][0], -dvji[0]);
			atomicAdd(&velocity[pId][1], -dvji[1]);
			atomicAdd(&velocity[pId][2], -dvji[2]);

			atomicAdd(&velocity[j][0], dvji[0]);
			atomicAdd(&velocity[j][1], dvji[1]);
			atomicAdd(&velocity[j][2], dvji[2]);

		}


		int nbSizeTri = neighborTri.getNeighborSize(pId);

		Coord vel_i = velocity_old[pId];
		Coord pos_i = position[pId];
		for (int ne = 0; ne < nbSizeTri; ne++)
		{
			int j = neighborTri.getElement(pId, ne);
			//if (j >= 0) continue;
			//j *= -1; j--;

			Triangle3D t3d(positionTri[m_triangle_index[j][0]], positionTri[m_triangle_index[j][1]], positionTri[m_triangle_index[j][2]]);
			Plane3D PL(positionTri[m_triangle_index[j][0]], t3d.normal());
			Point3D p3d(pos_i);
			Point3D nearest_pt = p3d.project(PL);

			Real r = (nearest_pt.origin - pos_i).norm();

			if (r < EPSILON) continue;


			Real AreaSum = p3d.areaTriangle(t3d, smoothingLength);
			Real MinDistance = abs(p3d.distance(t3d));
			Coord Min_Pt = (p3d.project(t3d)).origin - pos_i;
			if (ne < nbSizeTri - 1)
			{
				int jn;
				do
				{
					jn = neighborTri.getElement(pId, ne + 1);
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
					ne++;
				} while (ne < nbSizeTri - 1);
			}
			Min_Pt /= Min_Pt.norm();
			Coord n_PL = -t3d.normal();
			n_PL = n_PL / n_PL.norm();

			Real d = p3d.distance(PL);
			//if (d < 0) continue;
			d = abs(d);
			if (smoothingLength - d > EPSILON && smoothingLength * smoothingLength - d * d > EPSILON && d > EPSILON)
			{

				Coord nij = (pos_i - nearest_pt.origin);
				if (nij.norm() > EPSILON)
				{
					nij = nij.normalize();
				}
				else
					nij = t3d.normal().normalize();

				Coord normal_j = t3d.normal();
				normal_j = normal_j.normalize();


				Coord Velj = Coord(0);

				Coord dVel = Velj - vel_i;
				Real magNVel = dVel.dot(normal_j);
				Coord nVel = magNVel * normal_j;
				Coord tVel = dVel - nVel;

				Coord dv_i = Coord(0);

				
				dv_i += 2.0f * nij * force_sum * scale;

				

				atomicAdd(&velocity[pId][0], dv_i[0]);
				atomicAdd(&velocity[pId][1], dv_i[1]);
				atomicAdd(&velocity[pId][2], dv_i[2]);
			}

		}
		
	}



	template<typename TDataType>
	UnifiedVelocityConstraint<TDataType>::UnifiedVelocityConstraint()
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
	UnifiedVelocityConstraint<TDataType>::~UnifiedVelocityConstraint()
	{
		m_alpha.clear();
		m_Aii.clear();
		m_AiiFluid.clear();
		m_AiiTotal.clear();
//		m_pressure.release();
		//m_pressure2.release();
		m_divergence.clear();
		m_bSurface.clear();

		m_y.clear();
		m_r.clear();
		m_p.clear();

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
	bool UnifiedVelocityConstraint<TDataType>::constrain()
	{
		Real dt = getParent()->getDt();
		pretreat(dt);
		for (int i = 0; i < 10; i++)
			take_one_iteration(dt);
		update1(dt);
		return true;
	}


	template<typename TDataType>
	void UnifiedVelocityConstraint<TDataType>::pretreat(Real dt)
	{

		//Real dt = getParent()->getDt();
		uint pDims = cudaGridSize(m_particle_position.getElementCount(), BLOCK_SIZE);

		int numTri = m_triangle_vertex.getElementCount();
		uint pDimsT = cudaGridSize(numTri, BLOCK_SIZE);
		uint pDimsN = cudaGridSize(m_neighborhood_particles.getValue().getElementSize(), BLOCK_SIZE);


		int num_ele = m_neighborhood_particles.getValue().getElementSize();
		printf("NUM_ELE; %d NUM_P: %d\n", num_ele, m_particle_position.getElementCount());
		m_mapping.resize(num_ele);
		m_pairwise_force.resize(num_ele);
		m_pressure.setElementCount(num_ele);
		m_divergence.resize(num_ele);
		m_gradient.resize(num_ele);
		m_index_sym.resize(num_ele);
		//	m_index_sym.reset();

		UVC_initialize_mapping << <pDims, BLOCK_SIZE >> > (
			m_neighborhood_particles.getValue(),
			m_mapping
			);
		cuSynchronize();

		UVC_initialize_sym_mapping << <pDimsN, BLOCK_SIZE >> > (
			m_neighborhood_particles.getValue(),
			m_mapping,
			m_index_sym
			);
		cuSynchronize();
		if (!m_particle_position.isEmpty())
		{
			printf("warning from second step!");
			int num = m_particle_position.getElementCount();
			uint pDims = cudaGridSize(num, BLOCK_SIZE);

		}

		int num = m_particle_position.getElementCount();
		if (m_AiiFluid.isEmpty() || m_AiiFluid.size() != num)
		{
			m_alpha.resize(num);
			m_bSurface.resize(num);
			m_gradient_point.setElementCount(num);
			invRadius.resize(num);
			m_velocity_inside_iteration.setElementCount(num);
			m_pressure_point.setElementCount(num);
		}

		m_alpha.reset();
		m_pressure_point.getValue().reset();

		cuSynchronize();
		UVC_ComputeAlphaTmp << <pDims, BLOCK_SIZE >> > (
			m_alpha,
			m_particle_position.getValue(),
			m_particle_attribute.getValue(),
			m_neighborhood_particles.getValue(),
			m_smoothing_length.getValue()
			);
		cuSynchronize();

		UVC_CorrectAlphaTmp << <pDims, BLOCK_SIZE >> > (
			m_alpha,
			Rho_alpha,
			m_particle_mass.getValue(),
			m_maxAlpha);
		cuSynchronize();


		m_densitySum->compute();

		m_density = m_density_field.getValue();


		int itor = 0;
		m_divergence.reset();
		
		m_y.reset();
		m_pressure.getValue().reset();
		m_gradient.reset();
		invRadius.reset();

		cuSynchronize();

		if (m_arithmetic)
		{
			delete m_arithmetic;
		}
		if (m_reduce)
		{
			delete m_reduce;
		}
		m_arithmetic = Arithmetic<float>::Create(m_particle_position.getElementCount());
		m_reduce = Reduction<float>::Create(m_particle_position.getElementCount());
		err_last = -1.0f;
		step_i = 10.0f;

		int nbc_size = m_nbrcons.getElementCount();
		if (nbc_size > 0)
			cuExecute(
				nbc_size,
				UVC_CompensateRhoRigid,
				invRadius,
				m_gradient_rigid.getValue(),
				m_gradient_point.getValue(),
				m_density,
				m_nbrcons.getValue(),
				m_restDensity,
				m_smoothing_length.getValue(),
				m_sampling_distance.getValue(),
				dt);

		cuSynchronize();

	}

	template<typename TDataType>
	void UnifiedVelocityConstraint<TDataType>::take_one_iteration(Real dt)
	{
		
			uint pDimsN = cudaGridSize(m_neighborhood_particles.getValue().getElementSize(), BLOCK_SIZE);
			uint pDims = cudaGridSize(m_particle_position.getElementCount(), BLOCK_SIZE);

			m_gradient.reset();
			m_velocity_inside_iteration.getValue().reset();


			UVC_UpdateVelocityParticles << <pDims, BLOCK_SIZE >> > (
				m_pressure.getValue(),
				m_index_sym,
				m_alpha,
				m_particle_position.getValue(),
				m_particle_velocity.getValue(),
				m_velocity_inside_iteration.getValue(),
				m_particle_attribute.getValue(),
				m_density,
				m_neighborhood_particles.getValue(),
				m_restDensity,
				m_smoothing_length.getValue(),
				m_sampling_distance.getValue(),
				dt);
			
			cuSynchronize();

			UVC_ComputeGradient << <pDimsN, BLOCK_SIZE >> > (
				m_velocity_inside_iteration.getValue(),
				m_gradient,
				m_index_sym,
				m_density,
				m_pressure.getValue(),
				m_alpha,
				m_particle_position.getValue(),
				m_particle_attribute.getValue(),
				m_neighborhood_particles.getValue(),
				m_mapping,
				m_smoothing_length.getValue(),
				dt,
				m_restDensity);
			cuSynchronize();

			UVC_ComputeGradientPoint << <pDimsN, BLOCK_SIZE >> > (
				m_alpha,
				m_index_sym,
				invRadius,
				m_gradient,
				m_gradient_point.getValue(),
				m_pressure_point.getValue(),
				m_particle_position.getValue(),
				m_particle_attribute.getValue(),
				m_neighborhood_particles.getValue(),
				m_restDensity,
				m_airPressure,
				m_smoothing_length.getValue(),
				m_sampling_distance.getValue(),
				dt);
			cuSynchronize();

			Real rr = m_arithmetic->Dot(m_gradient, m_gradient);
			err = sqrt(rr / m_gradient.size());
			printf("err: %.13lf\n", err);
			if (err_last > 0 && err > err_last)
				return;
			err_last = err;

			Real step = 1.0f;
			UVC_UpdateGradient << <pDimsN, BLOCK_SIZE >> > (
				m_gradient_point.getValue(),
				m_gradient,
				m_pressure.getValue(),
				m_velocity_inside_iteration.getValue(),
				m_alpha,
				invRadius,
				m_particle_position.getValue(),
				m_particle_attribute.getValue(),
				m_neighborhood_particles.getValue(),
				m_mapping,
				m_smoothing_length.getValue(),
				dt,
				m_restDensity,
				step
				);
			cuSynchronize();


			
		
	}

	template<typename TDataType>
	void UnifiedVelocityConstraint<TDataType>::take_one_iteration_1(Real dt)
	{

		uint pDimsN = cudaGridSize(m_neighborhood_particles.getValue().getElementSize(), BLOCK_SIZE);
		uint pDims = cudaGridSize(m_particle_position.getElementCount(), BLOCK_SIZE);

		m_gradient.reset();
		m_velocity_inside_iteration.getValue().reset();

		//printf("nbr size: %d\n",m_gradient.size());
		UVC_UpdateVelocityParticles << <pDims, BLOCK_SIZE >> > (
			m_pressure.getValue(),
			m_index_sym,
			m_alpha,
			m_particle_position.getValue(),
			m_particle_velocity.getValue(),
			m_velocity_inside_iteration.getValue(),
			m_particle_attribute.getValue(),
			m_density,
			m_neighborhood_particles.getValue(),
			m_restDensity,
			m_smoothing_length.getValue(),
			m_sampling_distance.getValue(),
			dt);

		int pDimsNBC = m_nbrcons.getElementCount();
		if (pDimsNBC > 0)
			cuExecute(
				pDimsNBC,
				UVC_UpdateVelocityParticlesRigid,
				m_force_rigid.getValue(),
				m_velocity_inside_iteration.getValue(),
				m_nbrcons.getValue(),
				m_restDensity,
				m_smoothing_length.getValue(),
				m_sampling_distance.getValue(),
				dt);

		cuSynchronize();

		UVC_ComputeGradient << <pDimsN, BLOCK_SIZE >> > (
			m_velocity_inside_iteration.getValue(),
			m_gradient,
			m_index_sym,
			m_density,
			m_pressure.getValue(),
			m_alpha,
			m_particle_position.getValue(),
			m_particle_attribute.getValue(),
			m_neighborhood_particles.getValue(),
			m_mapping,
			m_smoothing_length.getValue(),
			dt,
			m_restDensity);
		cuSynchronize();
	}


	template<typename TDataType>
	void UnifiedVelocityConstraint<TDataType>::take_one_iteration_2(Real dt)
	{

		uint pDimsN = cudaGridSize(m_neighborhood_particles.getValue().getElementSize(), BLOCK_SIZE);
		uint pDims = cudaGridSize(m_particle_position.getElementCount(), BLOCK_SIZE);
		uint pDimsR = cudaGridSize(m_gradient_rigid.getElementCount(), BLOCK_SIZE);

		UVC_ComputeGradientPoint << <pDimsN, BLOCK_SIZE >> > (
			m_alpha,
			m_index_sym,
			invRadius,
			m_gradient,
			m_gradient_point.getValue(),
			m_pressure_point.getValue(),
			m_particle_position.getValue(),
			m_particle_attribute.getValue(),
			m_neighborhood_particles.getValue(),
			m_restDensity,
			m_airPressure,
			m_smoothing_length.getValue(),
			m_sampling_distance.getValue(),
			dt);
		cuSynchronize();

		int nbc_size = m_nbrcons.getElementCount();
		if(nbc_size > 0)
		cuExecute(
			nbc_size,
			UVC_ComputeGradientPointRigid,
			invRadius,
			m_gradient_rigid.getValue(),
			m_gradient_point.getValue(),
			m_density,
			m_nbrcons.getValue(),
			m_restDensity,
			m_smoothing_length.getValue(),
			m_sampling_distance.getValue(),
			dt);

		cuSynchronize();


		UVC_NormalizeGradientPoint << <pDimsN, BLOCK_SIZE >> > (
			m_gradient_point.getValue(),
			invRadius,
			m_density
			);
		cuSynchronize();
		
		Real rr = m_arithmetic->Dot(m_gradient_point.getValue(), m_gradient_point.getValue());
		err = sqrt(rr / m_gradient_point.getValue().size());
		printf("err: %.13lf\n", err);
	    //if (err_last > 0 && err > err_last)
			
			//return;
		err_last = err;
		if (err < 0.000000000001) return;

		Real max_g = m_reduce->maximum(m_gradient_point.getValue().begin(), m_gradient_point.getValue().size());
		Real min_g = m_reduce->minimum(m_gradient_point.getValue().begin(), m_gradient_point.getValue().size());

		step_i = min(step_i,min(0.000005 / max(abs(max_g), abs(min_g)), 10.0f));

		printf("err: %.15lf				step: %.15lf\n", err, step_i);

		UVC_UpdateGradient << <pDimsN, BLOCK_SIZE >> > (
			m_gradient_point.getValue(),
			m_gradient,
			m_pressure.getValue(),
			m_velocity_inside_iteration.getValue(),
			m_alpha,
			invRadius,
			m_particle_position.getValue(),
			m_particle_attribute.getValue(),
			m_neighborhood_particles.getValue(),
			m_mapping,
			m_smoothing_length.getValue(),
			dt,
			m_restDensity,
			step_i
			);
		cuSynchronize();


	}

	template<typename TDataType>
	void UnifiedVelocityConstraint<TDataType>::update(Real dt)
	{
		

		uint pDims = cudaGridSize(m_particle_position.getElementCount(), BLOCK_SIZE);

		
		
		cudaMemcpy(m_particle_velocity_buffer.begin(), m_particle_velocity.getValue().begin(), num_f * sizeof(Coord), cudaMemcpyDeviceToDevice);

		cuSynchronize();

		UVC_UpdateVelocityBoundaryCorrectedTmp << <pDims, BLOCK_SIZE >> > (
			m_pressure.getValue(),
			m_index_sym,
			m_alpha,
			m_bSurface,
			m_particle_position.getValue(),
			m_particle_velocity.getValue(),
			m_particle_velocity_buffer,
			m_meshVel,
			m_triangle_index.getValue(),
			m_triangle_vertex.getValue(),
			m_particle_attribute.getValue(),
			m_particle_mass.getValue(),
			m_density,
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
		cuSynchronize();
		

		int pDimsNBC = m_nbrcons.getElementCount();
		if (pDimsNBC > 0)
		cuExecute(
			pDimsNBC,
			UVC_UpdateVelocityParticlesRigid, 
			m_force_rigid.getValue(),
			m_particle_velocity.getValue(),
			m_nbrcons.getValue(),
			m_restDensity,
			m_smoothing_length.getValue(),
			m_sampling_distance.getValue(),
			dt);
		

		cuSynchronize();
//		return true;
	}

	template<typename TDataType>
	void UnifiedVelocityConstraint<TDataType>::update1(Real dt)
	{


		uint pDims = cudaGridSize(m_particle_position.getElementCount(), BLOCK_SIZE);



		cudaMemcpy(m_particle_velocity_buffer.begin(), m_particle_velocity.getValue().begin(), num_f * sizeof(Coord), cudaMemcpyDeviceToDevice);

		cuSynchronize();

		UVC_UpdateVelocityBoundaryCorrectedTmp << <pDims, BLOCK_SIZE >> > (
			m_pressure.getValue(),
			m_index_sym,
			m_alpha,
			m_bSurface,
			m_particle_position.getValue(),
			m_particle_velocity.getValue(),
			m_particle_velocity_buffer,
			m_meshVel,
			m_triangle_index.getValue(),
			m_triangle_vertex.getValue(),
			m_particle_attribute.getValue(),
			m_particle_mass.getValue(),
			m_density,
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
		cuSynchronize();

		//		return true;
	}


	template<typename TDataType>
	bool UnifiedVelocityConstraint<TDataType>::initializeImpl()
	{
		first_step = true;

		if (m_particle_position.isEmpty())
			printf("OKKKKKKKKKK\n");
		m_sampling_distance.setValue(0.005);

		//		printf("%d %d %d %d\n", m_position.getElementCount(), m_position1.getElementCount(), m_velocity.getElementCount(), m_velocity2.getElementCount());


		if (!m_particle_position.isEmpty())
		{
			printf("warning from second step! %d ~~~", m_neighborhood_triangles.getElementCount());
			int num = m_particle_position.getElementCount();
			uint pDims = cudaGridSize(num, BLOCK_SIZE);

			m_particle_attribute.setElementCount(num);
			m_particle_mass.setElementCount(num);

			//			m_particle_normal.setElementCount(num);
						//if (m_particle_attribute.isEmpty())printf("???\n");
			m_particle_attribute.getReference()->reset();
			UVC_InitAttrTmp << <pDims, BLOCK_SIZE >> > (
				m_particle_attribute.getValue(),
				m_particle_mass.getValue()
				);

		}
		else
		{
			printf("YES~ m_triangle_index Size: %d\n", m_triangle_index.getElementCount());
		}
	//	Real dt = getParent()->getDt();
		int numt = m_triangle_vertex.getElementCount();

		m_meshVel.resize(numt);
		uint pDims = cudaGridSize(numt, BLOCK_SIZE);


		
		m_densitySum = std::make_shared<DensitySummationMesh<TDataType>>();
		m_smoothing_length.connect(&m_densitySum->m_smoothingLength);
		m_particle_position.connect(&m_densitySum->m_position);
		m_neighborhood_particles.connect(&m_densitySum->m_neighborhood);
		m_density_field.connect(&m_densitySum->m_density);

		m_neighborhood_triangles.connect(&m_densitySum->m_neighborhoodTri);
		m_triangle_index.connect(&m_densitySum->Tri);
		m_triangle_vertex.connect(&m_densitySum->TriPoint);
		m_sampling_distance.connect(&m_densitySum->sampling_distance);
		m_densitySum->use_mesh.setValue(1);
		m_densitySum->use_ghost.setValue(0);
		m_densitySum->Start.setValue(m_particle_position.getElementCount());

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

		m_velocity_inside_iteration.setElementCount(num);
		m_alpha.resize(num);
		Rho_alpha.resize(num);
		m_Aii.resize(num);
		m_AiiFluid.resize(num);
		m_AiiTotal.resize(num);
		m_pressure.setElementCount(num);
		m_divergence.resize(num);
		//m_divergence_Tri.resize(num);
		m_bSurface.resize(num);
		m_density.resize(num);
		m_particle_velocity_buffer.resize(num);
		m_gradient_point.setElementCount(num);
		m_pressure_point.setElementCount(num);
		invRadius.resize(num);

		m_y.resize(num);
		m_r.resize(num);
		m_p.resize(num);

		//	m_pressure.resize(num);

		m_reduce = Reduction<float>::Create(num);
		m_arithmetic = Arithmetic<float>::Create(num);


		pDims = cudaGridSize(num, BLOCK_SIZE);
		

		//printf("TRI:%d\n", m_triangle_index.getValue().ge);
//		printf("NEI1:%d\n", m_neighborhood.isEmpty());
		m_alpha.reset();
		printf("FLIP ========: %d\n", m_flip.getValue().size());
		UVC_ComputeAlphaTmp << <pDims, BLOCK_SIZE >> > (
			m_alpha,
			m_particle_position.getValue(),
			m_particle_attribute.getValue(),
			m_neighborhood_particles.getValue(),
			m_smoothing_length.getValue()
			);

		m_maxAlpha = m_reduce->maximum(m_alpha.begin(), m_alpha.size());

		UVC_CorrectAlphaTmp << <pDims, BLOCK_SIZE >> > (
			m_alpha,
			Rho_alpha,
			m_particle_mass.getValue(),
			m_maxAlpha);

		m_AiiFluid.reset();
	

		std::cout << "Max alpha: " << m_maxAlpha << std::endl;
		printf("%.10lf\n", m_maxAlpha);
		std::cout << "Max A: " << m_maxA << std::endl;


		return true;
	}
}