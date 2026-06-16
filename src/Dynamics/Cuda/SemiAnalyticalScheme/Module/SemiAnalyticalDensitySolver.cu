#include "SemiAnalyticalDensitySolver.h"

#include "SemiAnalyticalSummationDensity.h"
#include "TriangularMeshConstraint.h"
#include "Collision/DynamicNeighborTriangleQuery.h"
#include "Collision/NeighborTriangleQuery.h"
#include "Collision/Distance3D.h"
#include "Matrix/MatrixFunc.h"
#include <string>
#include "Algorithm/Function2Pt.h"
#include <cuda_runtime.h>
#include <iostream>
#include "Primitive/PrimitiveSweep3D.h"
#include "CCD/AdditiveCCD.h"
#include "CCD/LightCCD.h"
#include "Collision/ComputeGeometry.h"
#include "IntersectionArea.h"
#include "Timer.h"
#include "SceneGraph.h"
#include <curand_kernel.h>
#include <fstream>

namespace dyno {

#define TIME_LOG
#define CONSERVE_MOMETNUM
#define BOUNDARY_PARTICLE
#define CONTACT_MODE



	IMPLEMENT_TCLASS(SemiAnalyticalDensitySolver, TDataType)
		template<typename TDataType>
	SemiAnalyticalDensitySolver<TDataType>::SemiAnalyticalDensitySolver()
		: ParticleApproximation<TDataType>()
	{
		mCalculateDensity = std::make_shared<SemiAnalyticalSummationDensity<TDataType>>();
		this->inSamplingDistance()->connect(mCalculateDensity->inSamplingDistance());
		this->inSmoothingLength()->connect(mCalculateDensity->inSmoothingLength());
		this->inPosition()->connect(mCalculateDensity->inPosition());
		this->inNeighborIds()->connect(mCalculateDensity->inNeighborIds());
		this->inNeighborTriIdsMerge()->connect(mCalculateDensity->inNeighborTriIds());
		this->inTriangleSetMerge()->connect(mCalculateDensity->inTriangleSet());
		this->varRestDensity()->quote(mCalculateDensity->varRestDensity());

		this->varKernelType()->setCurrentKey(EKernelType::KT_Spiky);
		m_arithmetic_v = Arithmetic<typename TDataType::Real>::Create(3);
	};

	template<typename TDataType>
	SemiAnalyticalDensitySolver<TDataType>::~SemiAnalyticalDensitySolver()
	{
		if (m_arithmetic_v != nullptr)
		{
			delete m_arithmetic_v;
			m_arithmetic_v = nullptr;
		}

		mPosOld.clear();
		mPosBuf.clear();
		mPosStart.clear();
		mVelStart.clear();
		mEnergy.clear();
		mSignDis.clear();
		mPolyN.clear();
		mA.clear();
		mAlpha.clear();
		mKappa.clear();
	};

	__device__ __forceinline__ Quat<float> nlerpQuat(const Quat<float>& q1In, const Quat<float>& q2In, float t)
	{
		Quat<float> q1 = q1In;
		Quat<float> q2 = q2In;
		q1.normalize();
		q2.normalize();

		// Ensure shortest path
		if (q1.dot(q2) < 0.0f)
		{
			q2 = -q2;
		}

		Quat<float> r = (1.0f - t) * q1 + t * q2;
		r.normalize();
		return r;
	}

	__device__ __forceinline__ bool buildTriBasis(
		const Vector<float, 3>& v0,
		const Vector<float, 3>& v1,
		const Vector<float, 3>& v2,
		Vector<float, 3>& e1,
		Vector<float, 3>& e2,
		Vector<float, 3>& e3)
	{
		Vector<float, 3> a = v1 - v0;
		float na = a.norm();
		if (na < 1e-10f) return false;
		e1 = a * (1.0f / na);

		Vector<float, 3> b = v2 - v0;
		b = b - e1 * b.dot(e1);
		float nb = b.norm();
		if (nb < 1e-10f) return false;
		e2 = b * (1.0f / nb);

		e3 = e1.cross(e2);
		float nc = e3.norm();
		if (nc < 1e-10f) return false;
		e3 = e3 * (1.0f / nc);
		return true;
	}

	__device__ __forceinline__ bool rigidInterpTriangle(
		const Vector<float, 3>& x0,
		const Vector<float, 3>& x1,
		const Vector<float, 3>& x2,
		const Vector<float, 3>& y0,
		const Vector<float, 3>& y1,
		const Vector<float, 3>& y2,
		float t,
		Vector<float, 3>& out0,
		Vector<float, 3>& out1,
		Vector<float, 3>& out2)
	{
		Vector<float, 3> c0 = (x0 + x1 + x2) * (1.0f / 3.0f);
		Vector<float, 3> c1 = (y0 + y1 + y2) * (1.0f / 3.0f);
		Vector<float, 3> ct = c0 + (c1 - c0) * t;

		Vector<float, 3> e1, e2, e3;
		Vector<float, 3> f1, f2, f3;
		if (!buildTriBasis(x0, x1, x2, e1, e2, e3)) return false;
		if (!buildTriBasis(y0, y1, y2, f1, f2, f3)) return false;

		// R = [f1 f2 f3] * [e1 e2 e3]^T = sum_k f_k * e_k^T
		SquareMatrix<float, 3> R;
		R(0, 0) = f1.x * e1.x + f2.x * e2.x + f3.x * e3.x;
		R(0, 1) = f1.x * e1.y + f2.x * e2.y + f3.x * e3.y;
		R(0, 2) = f1.x * e1.z + f2.x * e2.z + f3.x * e3.z;
		R(1, 0) = f1.y * e1.x + f2.y * e2.x + f3.y * e3.x;
		R(1, 1) = f1.y * e1.y + f2.y * e2.y + f3.y * e3.y;
		R(1, 2) = f1.y * e1.z + f2.y * e2.z + f3.y * e3.z;
		R(2, 0) = f1.z * e1.x + f2.z * e2.x + f3.z * e3.x;
		R(2, 1) = f1.z * e1.y + f2.z * e2.y + f3.z * e3.y;
		R(2, 2) = f1.z * e1.z + f2.z * e2.z + f3.z * e3.z;

		Quat<float> qFull(R);
		Quat<float> qT = nlerpQuat(Quat<float>::identity(), qFull, t);

		out0 = ct + qT.rotate(x0 - c0);
		out1 = ct + qT.rotate(x1 - c0);
		out2 = ct + qT.rotate(x2 - c0);
		return true;
	}

	template<typename Real, typename Coord>
	__global__ void ccdPreResolveParticles(
		DArray<Coord> particle_position,
		DArray<Coord> particle_velocity,
		DArray<Coord> triangle_vertex,
		DArray<Coord> triangle_vertex_previous,
		DArray<Topology::Triangle> triangle_index,
		DArrayList<int> triangle_neighbors,
		DArray<Real> mSignDis,
		DArray<uint> mPolyN,
		uint N,
		bool warmStart,
		DArray<Real> outKappas,
		Real KappaLower,
		DArray<Real> mEnergy,
		Real d_hat,
		uint shapeSize,
		Coord g,
		Real dt)
	{
		typedef typename TPoint3D<Real> Point3D;
		typedef typename TTriangle3D<Real> Triangle3D;
		typedef typename TPointSweep3D<Real> PointSweep3D;

		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= particle_position.size()) return;
		List<int>& nbrTriIds_i = triangle_neighbors[pId];
		int nbrSize = nbrTriIds_i.size();
		mSignDis[pId] = 2 * d_hat;
		mEnergy[pId] = 0.5 * particle_velocity[pId].dot(particle_velocity[pId]) - g.y * particle_position[pId].y;
		Coord pos_i = particle_position[pId];
		Coord vel_i = particle_velocity[pId] - g * dt;//   
		Coord radius = vel_i * dt;
		Coord pos_i_old = pos_i - radius;
		Point3D start_point(pos_i_old);
		Point3D end_point(pos_i);
		Real min_toi = Real(2.0);
		Coord collision_pos_j_final(0);
		for (int ne = 0; ne < nbrSize; ne++)
		{
			int j = nbrTriIds_i[ne];
			if (j < 0 || j >= (int)triangle_index.size()) continue;

			Triangle3D start_triangle(triangle_vertex_previous[triangle_index[j][0]], triangle_vertex_previous[triangle_index[j][1]], triangle_vertex_previous[triangle_index[j][2]]);
			Triangle3D end_triangle(triangle_vertex[triangle_index[j][0]], triangle_vertex[triangle_index[j][1]], triangle_vertex[triangle_index[j][2]]);
			Real toi = Real(1.0);
			Vector<float, 3> x0 = triangle_vertex_previous[triangle_index[j][0]];
			Vector<float, 3> x1 = triangle_vertex_previous[triangle_index[j][1]];
			Vector<float, 3> x2 = triangle_vertex_previous[triangle_index[j][2]];
			Vector<float, 3> x3 = pos_i_old;
			Vector<float, 3> y0 = triangle_vertex[triangle_index[j][0]];
			Vector<float, 3> y1 = triangle_vertex[triangle_index[j][1]];
			Vector<float, 3> y2 = triangle_vertex[triangle_index[j][2]];
			Vector<float, 3> y3 = pos_i;
			bool collided;
			collided = LightCCD<float>::VertexFaceCCD(x3, x0, x1, x2, y3, y0, y1, y2, toi, pId, j);
			if (collided)
			{
				if (toi < min_toi)
				{
					Coord collision_pos_j = (pos_i - pos_i_old) * toi + pos_i_old;
					Vector<float, 3> middle_triangle_u, middle_triangle_v, middle_triangle_w;
					if (!rigidInterpTriangle(x0, x1, x2, y0, y1, y2, (float)toi, middle_triangle_u, middle_triangle_v, middle_triangle_w))
					{
						// Fallback to per-vertex linear interpolation if basis is degenerate
						middle_triangle_u = (y0 - x0) * (float)toi + x0;
						middle_triangle_v = (y1 - x1) * (float)toi + x1;
						middle_triangle_w = (y2 - x2) * (float)toi + x2;
					}
					Triangle3D middle_triangle(middle_triangle_u, middle_triangle_v, middle_triangle_w);
					typename Triangle3D::Param baryc;
					Point3D pos_j(collision_pos_j);
					middle_triangle.computeBarycentrics(pos_j.project(middle_triangle).origin, baryc);
					collision_pos_j_final = end_triangle.computeLocation(baryc) + 1000 * EPSILON * end_triangle.normal();
					min_toi = toi;
				}
			}
		}
		if (shapeSize > 1)
		{
			if (min_toi <= 1 - EPSILON && min_toi >= EPSILON)
			{
				particle_position[pId] = collision_pos_j_final;
				mSignDis[pId] = 1000 * EPSILON;
			}
		}
		else
		{
			if (min_toi <= 1 && min_toi >= 0)//cone needs more precise ccd 
			{
				particle_position[pId] = collision_pos_j_final;
				mSignDis[pId] = 1000 * EPSILON;
			}
		}
		mPolyN[pId] = N;
		// Warm-start: keep previous kappa as initial guess, but clamp to lower bound.
		if (warmStart)
		{
			outKappas[pId] = outKappas[pId] > KappaLower ? outKappas[pId] : KappaLower;
		}

	}

	template <typename Real>
	__global__ void SIIS_InitKappas(DArray<Real> kappas, Real kappaLower)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= (int)kappas.size()) return;
		kappas[pId] = kappaLower;
	}

	template <typename Real>
	__global__ void SIIS_ClampKappasLower(DArray<Real> kappas, Real kappaLower)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= (int)kappas.size()) return;
		Real v = kappas[pId];
		kappas[pId] = v > kappaLower ? v : kappaLower;
	}

	template <typename Real, typename Coord, typename Kernel>
	__global__ void SIIS_UpdateFluidParticles(
		DArray<Coord> posNext,	//out
		DArray<Real> diagnals,	//out
		DArray<Coord> Kappas,
		DArray<Coord> posBuf,	//in
		DArray<Real> rho,		//in
		DArray<Real> BoundaryDensity,
		DArrayList<int> neighbors,	//in
		Real smoothingLength,
		Real mu,
		Real dt,
		Kernel gradient,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posNext.size()) return;
		Real rho_0 = Real(1000);

		Real rho_i = rho[pId] - BoundaryDensity[pId];
		rho_i = rho_i > rho_0 ? rho_i : rho_0;
		Real bulk_rho_i = rho[pId];
		bulk_rho_i = bulk_rho_i > rho_0 ? bulk_rho_i : rho_0;
		Real A = mu * dt * dt / rho_0;
		Real B_plus = rho_i / rho_0;
		Real bulk_B_plus = bulk_rho_i / rho_0;
		Real B_minus = Real(-1);

		Coord pos_i = posBuf[pId];
		//printf("[%d] pos_i: %f %f %f; rho_i: %f;\n", pId, pos_i.x, pos_i.y, pos_i.z, rho[pId]);

		Coord posAcc_i(0);

		Real a_i = Real(0);

		List<int>& nbrIds_i = neighbors[pId];
		int nbSize = nbrIds_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbrIds_i[ne];
			Coord pos_j = posBuf[j];
			Real rho_j = max(rho[j], rho_0);
			Real r = (pos_i - pos_j).norm();
			if (r > EPSILON && r < smoothingLength)
			{
				Real a_ij = A * gradient(r, smoothingLength, scale) * (1.0f / r);
#ifdef CONSERVE_MOMETNUM
				// j -> i
				posAcc_i += B_minus * a_ij * pos_j + B_plus * a_ij * (pos_j - pos_i);
				// i -> j
				Coord posAcc_ji = B_minus * a_ij * pos_i + B_plus * a_ij * (pos_i - pos_j);
				Real a_ji = B_minus * a_ij;

				atomicAdd(&posNext[j][0], posAcc_ji[0]);
				atomicAdd(&posNext[j][1], posAcc_ji[1]);
				atomicAdd(&posNext[j][2], posAcc_ji[2]);
				atomicAdd(&diagnals[j], a_ji);

#else
				posAcc_i += B_minus * a_ij * pos_j + B_plus * a_ij * (pos_j - pos_i);
#endif // CONSERVE_MOMETNUM
				a_i += B_minus * a_ij;
				Kappas[pId] += (bulk_B_plus + B_minus) * a_ij * (pos_j - pos_i) / (dt * dt);
			}
		}
#ifdef CONSERVE_MOMETNUM
		atomicAdd(&posNext[pId][0], posAcc_i[0]);
		atomicAdd(&posNext[pId][1], posAcc_i[1]);
		atomicAdd(&posNext[pId][2], posAcc_i[2]);
		atomicAdd(&diagnals[pId], a_i);
#else
		posNext[pId] = posAcc_i;
		diagnals[pId] = a_i;
#endif // CONSERVE_MOMETNUM
	}
	template <typename Real, typename Coord, typename Kernel>
	__global__ void SIIS_UpdateSemiAnalyticalBoundaryParticles(
		DArray<Coord> posNext,
		DArray<Real> diagnals,
		DArray<Coord> Kappas,
		DArray<Coord> posBuf,
		DArray<Real> rho,
		DArray<Topology::Triangle> triIndices,
		DArray<Coord> triVertices,
		DArrayList<int> neighborTri,
		Real smoothingLength,
		Real samplingDistance,
		Real mu,
		DArray<uint> mPolyN,
		DArray<Real> SignDis,
		DArray<uint> varyflag,
		DArray<uint> count,
		DArray<Coord> nearNorm,
		DArray<int> nbrTriId,
		DArray<Real> outKappas,
		uint shape,
		Real dt,
		Kernel integral,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posNext.size()) return;
		Coord pos_i = posBuf[pId];
		Real rho_0 = Real(1000);
		Real rho_i = rho[pId];
		rho_i = maximum(rho_i, rho_0);

		Real B_plus = rho_i / rho_0;
		Real B_minus = Real(-1);
		Real A = mu * dt * dt / rho_0;

		Coord posAcc_i(0);
		Real a_i = Real(0);
		Real signed_distance = 2 * smoothingLength;
		int final_triangle = -1;
		Coord final_normal(0);
		Coord final_force(0);
		Point3D p3d(pos_i);
		Real ratio = smoothingLength * smoothingLength * smoothingLength / (samplingDistance * samplingDistance * samplingDistance);
		List<int>& nbrTriIds_i = neighborTri[pId];
		int nbSizeTri = nbrTriIds_i.size();
		for (int ne = 0; ne < nbSizeTri; ne++)
		{
			int j = nbrTriIds_i[ne];

			Triangle3D t3d(triVertices[triIndices[j][0]], triVertices[triIndices[j][1]], triVertices[triIndices[j][2]]);
			Coord t_normal = t3d.normal();
			Plane3D plane(triVertices[triIndices[j][0]], t_normal);

			Point3D nearest_pt = p3d.project(plane);

			Coord pos_j = nearest_pt.origin;

			Coord d_n = pos_i - pos_j;

			Real r = d_n.norm();
			d_n = r > EPSILON ? d_n / r : t_normal;
			Real d = p3d.distance(plane);


			//If the new triangle is closer, use the new triangle to update p3d
			if (glm::abs(d) < glm::abs(signed_distance) - EPSILON)
			{
				signed_distance = d;
				final_normal = t_normal;
				final_triangle = j;
			}

			if (d > 0 && r > EPSILON)
			{
				Real A_0 = ((M_PI) * (smoothingLength * smoothingLength - d * d));
				A_0 = A_0 < EPSILON ? EPSILON : A_0;
				Real omega_0 = 2 * M_PI * (1 - d / smoothingLength);
				Real omega = abs(d_n.dot(t3d.normal())) * calculateIntersectionArea(p3d, t3d, smoothingLength) * omega_0 / A_0;

				Real w_ij = integral(r, smoothingLength, 1) * omega;
				Real eta_ij = ratio * w_ij;
				//TODO: this is an implementation hack, use a more elegant way 
				Real a_ij = eta_ij * A * SpikyKernel<Real>::gradient(r, smoothingLength, scale) * (1.0f / r);

				// j -> i
				posAcc_i += B_minus * a_ij * pos_j + B_plus * a_ij * (pos_j - pos_i);
				a_i += B_minus * a_ij;

				final_force += (B_plus + B_minus) * a_ij * (pos_j - pos_i) / (dt * dt);
			}
		}
		if (signed_distance > EPSILON && SignDis[pId] < -EPSILON)
		{
			varyflag[pId] = 1;
		}
		else
		{
			varyflag[pId] = 0;
		}
		SignDis[pId] = signed_distance;
		if (SignDis[pId] < -EPSILON)
		{
			mPolyN[pId] *= 2;
		}
		count[pId] = shape;
		nearNorm[pId] = final_normal;
		nbrTriId[pId] = final_triangle;

#ifdef BOUNDARY_PARTICLE
		posNext[pId] += posAcc_i;
		diagnals[pId] += a_i;
		Kappas[pId] += final_force;

#endif // BOUNDARY_PARTICLE
	}


	template <typename Real, typename Coord, typename Tri2Edg>
	__global__ void SIIS_UpdateSemiAnalyticalContactPotential(
		DArrayList<Coord> posNext,
		DArrayList<Real> diagnals,
		DArray<Coord> posBuf,
		DArray<Topology::Triangle> triIndices,
		DArray<Coord> triVertices,
		DArray<Topology::Edge> edges,
		DArray<Tri2Edg> t2e,
		DArray<Coord> edgeN,
		DArray<Coord> vertexN,
		DArrayList<int> neighborTri,
		uint shape,
		DArray<uint> mPolyN,
		Real volume,
		Real d_hat,
		DArrayList<Real> mEnergy,
		DArrayList<Coord> mContactForce,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= mPolyN.size()) return;
		const Real rho_0 = Real(1000);
		Coord pos_i = posBuf[pId];

		auto& list = neighborTri[pId];
		List<Coord>& mContactForcelist = mContactForce[pId];
		List<Coord>& posNextlist = posNext[pId];
		List<Real>& diagnalslist = diagnals[pId];
		List<Real>& mEnergylist = mEnergy[pId];
		// NOTE: These per-shape lists are indexed by "shape" via operator[] and do NOT use insert().
		// List::size() tracks inserted element count, which would stay 0. So we must treat them as fixed-size buffers.
		mContactForcelist[shape] = Coord(0);
		posNextlist[shape] = Coord(0);
		diagnalslist[shape] = 0;
		mEnergylist[shape] = 0;
		if (list.size() <= 0)
		{
			return;
		}

		ProjectedPoint3D<Real> p3d;
		bool valid = calculateSignedDistance2TriangleSetFromNormal(p3d, pos_i, triVertices, edges, triIndices, t2e, edgeN, vertexN, list);
		if (valid)
		{
			Real d = p3d.signed_distance; //pos_i.y;
			Coord normal = p3d.normal;	//Coord(0, 1, 0);
			Coord pos_j = pos_i - d * normal;
			Real gamma = d / d_hat;
			Real potential = 0;
			if (glm::abs(d) < d_hat)
			{
				uint N = mPolyN[pId];
				Real force = 0;
				Real A = dt * dt / rho_0;
				Real B_plus = 0;
				Real B_minus = 0;

				uint Cnk = N;
				Real gammaOdd = 1 / d_hat;
				Real gammaEven = 1 / (d_hat * d_hat);
				for (uint k = 1; k <= N; k++)
				{
					if (k % 2 == 0)
					{
						B_plus += Cnk * gammaEven;
						gammaEven *= (gamma * gamma);
					}
					if (k % 2 == 1) {
						B_minus += Cnk * gammaOdd;
						gammaOdd *= (gamma * gamma);
					}

					//C(n, k) * gamma ^ k
					Cnk *= (N - k);
					Cnk /= (k + 1);
					potential += glm::pow(1 - gamma, k) / k;
					force += glm::pow(1 - gamma, k - 1);
				}

				Real a_i = A * B_plus;
				Coord posAcc_i = A * B_minus * normal + A * B_plus * pos_j;
				posNextlist[shape] = posAcc_i;
				diagnalslist[shape] = a_i;
				mEnergylist[shape] = volume * potential;
				mContactForcelist[shape] = A * (B_minus * normal + B_plus * (pos_j - pos_i)) / (dt * dt);
			}
		}

	}

	template <typename Real, typename Coord>
	__global__ void SIIS_SumSemiAnalyticalContactPotential(
		DArrayList<Coord> posNext,
		DArrayList<Real> diagnals,
		DArray<Coord> inPos,
		DArray<Real> mA,
		DArray<uint> varyflag,
		Real KappaLower,
		DArray<Coord> Kappas,
		DArray<Real> outKappas,
		DArray<Real> sumEnergy,
		Real d_hat,
		DArray<Real> SignDis,
		DArray<Coord> nearNorm,
		DArrayList<Real> mEnergy,
		DArrayList<Coord> mContactForce
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= mA.size()) return;

		if (glm::abs(SignDis[pId]) > d_hat || nearNorm[pId].norm() < 0.5)
		{
			return;
		}
		List<Coord>& mContactForcelist = mContactForce[pId];
		List<Coord>& posNextlist = posNext[pId];
		List<Real>& diagnalslist = diagnals[pId];
		List<Real>& mEnergylist = mEnergy[pId];
		Coord CF(0);
		// For per-shape lists, size() may remain 0 because elements are assigned via operator[] instead of insert().
		// Use max_size() (capacity) as the intended fixed slot count.
		int c1 = (int)mContactForcelist.max_size();
		int c2 = (int)posNextlist.max_size();
		int c3 = (int)diagnalslist.max_size();
		int c4 = (int)mEnergylist.max_size();
		int n = c1;
		n = n < c2 ? n : c2;
		n = n < c3 ? n : c3;
		n = n < c4 ? n : c4;
		for (int i = 0; i < n; i++)
		{
			CF += mContactForcelist[i];
		}
		Real cfNorm = CF.norm();
		if (cfNorm < EPSILON)
		{
			return;
		}
		Coord Kappas_buffer = Kappas[pId];
		Kappas[pId] += (d_hat - SignDis[pId]) * nearNorm[pId];
		Real kappa = Kappas[pId].norm() / cfNorm;
		kappa = kappa > KappaLower ? kappa : KappaLower;


		if (varyflag[pId] > EPSILON)
		{
			outKappas[pId] = kappa;
		}
		else
		{
			kappa = outKappas[pId];
		}
		for (int i = 0; i < n; i++)
		{
			inPos[pId] += kappa * posNextlist[i];
			mA[pId] += kappa * diagnalslist[i];
			sumEnergy[pId] += kappa * mEnergylist[i];
		}
		Kappas[pId] = Kappas_buffer + CF * kappa;

	}




	template <typename Real, typename Coord>
	__global__ void relaxPositionEnergy(
		DArray<Coord> posNext,
		DArray<Coord> posBuf,
		DArray<Coord> posOld,
		DArray<Real> mEnergy,
		DArray<Real> mA,
		DArray<Real> mAlpha,
		DArray<Real> rho,
		Real samplingDistance,
		Real mu,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posNext.size()) return;

		Coord grad = (mA[pId] + 1) * posBuf[pId] - (posOld[pId] + posNext[pId]);

		Coord posNext_i = (posOld[pId] + posNext[pId]) / (mA[pId] + 1);

		Coord dx = posNext_i - posBuf[pId];
		Coord ds = posBuf[pId] - posOld[pId];

		const Real rho_0 = Real(1000);
		const Real m = rho_0 * samplingDistance * samplingDistance * samplingDistance;
		const Real C = m / (dt * dt);

		Real square = C * dx.dot(grad);

		Real contactE = mEnergy[pId];
		Real rho_i = rho[pId];
		rho_i = maximum(rho_i, rho_0);
		Real bulkE_i = mu * (rho_i / rho_0 - 1) * (rho_i / rho_0 - 1) * 0.5;
		mEnergy[pId] += 0.5 * C * ds.dot(ds) + bulkE_i;

		Real alpha_i = minimum(mEnergy[pId] / (-square + EPSILON), Real(1));
		if (rho_i > rho_0)alpha_i = alpha_i;
		else alpha_i = Real(1);
		mAlpha[pId] = alpha_i;
		posNext[pId] = posBuf[pId] + alpha_i * (posNext_i - posBuf[pId]);
	}


	template <typename Real>
	__global__ void clampDensityMin(
		DArray<Real> densityIn,
		DArray<Real> densityOut,
		Real rho0)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= densityIn.size()) return;

		Real rho_i = densityIn[pId];
		rho_i = maximum(rho_i, rho0);
		densityOut[pId] = rho_i;
	}


	template <typename Real, typename Coord>
	__global__ void updateVelocityFromPositions(
		DArray<Coord> velArr,
		DArray<Coord> velStart,
		DArray<Coord> curPos,
		DArray<Coord> prePos,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.size()) return;

		Coord delta_v = (curPos[pId] - prePos[pId]) / dt;
		velArr[pId] = velStart[pId] + delta_v; // Update velocity based on current and previous positions
		//Real maxVel = h / dt;
		//if (velArr[pId].norm() > maxVel)
		//{
		//	velArr[pId] = velArr[pId].normalize() * maxVel;
		//}

		//printf("velArr[%d]: %f; delta_v[%d]: %f\n", pId, velArr[pId].y, pId, delta_v.y);


	}

	template <typename Real, typename Coord>
	__global__ void SIIS_ApplyBoundaryTangentialFriction(
		DArray<Coord> velArr,
		DArray<Coord> posArr,
		DArray<Real> signDis,
		DArray<Coord> nearNorm,
		DArray<int> nbrTriId,
		DArray<Coord> bulkForce,
		DArray<Topology::Triangle> triIndex,
		DArray<Coord> triVertex,
		DArray<Coord> triVertexPrev,
		Real d_hat,
		Real friction,
		Real invMass,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= (int)velArr.size()) return;

		if (friction <= (Real)0) return;
		int tId = nbrTriId[pId];
		if (tId < 0) return;
		if (tId >= (int)triIndex.size()) return;
		if (glm::abs(signDis[pId]) > d_hat) return;

		Coord n = nearNorm[pId];
		Real nLen = n.norm();
		if (nLen <= (Real)1e-6) return;
		n = n / nLen;

		// Estimate triangle rigid motion and evaluate face velocity at contact point.
		// For rigid motion: v(x) = v_cm + omega x (x - c).
		Topology::Triangle tri = triIndex[tId];
		int i0 = (int)tri[0];
		int i1 = (int)tri[1];
		int i2 = (int)tri[2];
		if (i0 < 0 || i1 < 0 || i2 < 0) return;
		if (i0 >= (int)triVertex.size() || i1 >= (int)triVertex.size() || i2 >= (int)triVertex.size()) return;
		if (i0 >= (int)triVertexPrev.size() || i1 >= (int)triVertexPrev.size() || i2 >= (int)triVertexPrev.size()) return;
		Real invDt = dt > (Real)0 ? (Real)1 / dt : (Real)0;
		if (invDt <= (Real)0) return;

		Coord a0 = triVertexPrev[i0];
		Coord a1 = triVertexPrev[i1];
		Coord a2 = triVertexPrev[i2];
		Coord b0 = triVertex[i0];
		Coord b1 = triVertex[i1];
		Coord b2 = triVertex[i2];
		Coord c0 = (a0 + a1 + a2) * ((Real)1 / (Real)3);
		Coord c1 = (b0 + b1 + b2) * ((Real)1 / (Real)3);
		Coord v_cm = (c1 - c0) * invDt;

		// Build orthonormal bases on prev/current triangles.
		auto safeNormalize = [](const Coord& v, Real eps, bool& ok) {
			Real l = v.norm();
			if (l <= eps) { ok = false; return v; }
			return v / l;
			};
		bool ok = true;
		Coord e1 = safeNormalize(a1 - a0, (Real)1e-8, ok);
		Coord tmpE2 = (a2 - a0);
		tmpE2 = tmpE2 - e1 * tmpE2.dot(e1);
		Coord e2 = safeNormalize(tmpE2, (Real)1e-8, ok);
		Coord e3 = e1.cross(e2);
		Coord f1 = safeNormalize(b1 - b0, (Real)1e-8, ok);
		Coord tmpF2 = (b2 - b0);
		tmpF2 = tmpF2 - f1 * tmpF2.dot(f1);
		Coord f2 = safeNormalize(tmpF2, (Real)1e-8, ok);
		Coord f3 = f1.cross(f2);
		if (!ok || e3.norm() <= (Real)1e-8 || f3.norm() <= (Real)1e-8)
		{
			// Degenerate triangle: fall back to centroid velocity (average vertex velocity ~= v_cm).
			Coord vFace = v_cm;
			Real Fn = (bulkForce[pId].dot(n) < 0) ? -bulkForce[pId].dot(n) : (Real)0;
			if (Fn <= (Real)0) return;
			Coord v = velArr[pId];
			Coord vRel = v - vFace;
			Real vn = vRel.dot(n);
			Coord vN = n * vn;
			Coord vT = vRel - vN;
			Real vt = vT.norm();
			if (vt <= (Real)1e-8) return;
			Real maxDvT = friction * Fn * dt * invMass;
			Real newVt = vt - maxDvT;
			if (newVt < (Real)0) newVt = (Real)0;
			Coord vTnew = vT * (newVt / vt);
			velArr[pId] = vFace + vN + vTnew;
			return;
		}

		// Rotation matrix R = F * E^T = f1*e1^T + f2*e2^T + f3*e3^T.
		Real R00 = f1.x * e1.x + f2.x * e2.x + f3.x * e3.x;
		Real R01 = f1.x * e1.y + f2.x * e2.y + f3.x * e3.y;
		Real R02 = f1.x * e1.z + f2.x * e2.z + f3.x * e3.z;
		Real R10 = f1.y * e1.x + f2.y * e2.x + f3.y * e3.x;
		Real R11 = f1.y * e1.y + f2.y * e2.y + f3.y * e3.y;
		Real R12 = f1.y * e1.z + f2.y * e2.z + f3.y * e3.z;
		Real R20 = f1.z * e1.x + f2.z * e2.x + f3.z * e3.x;
		Real R21 = f1.z * e1.y + f2.z * e2.y + f3.z * e3.y;
		Real R22 = f1.z * e1.z + f2.z * e2.z + f3.z * e3.z;

		Real tr = R00 + R11 + R22;
		Real cosang = (tr - (Real)1) * (Real)0.5;
		if (cosang > (Real)1) cosang = (Real)1;
		if (cosang < (Real)-1) cosang = (Real)-1;
		Real angle = glm::acos(cosang);
		Coord omega((Real)0);
		if (angle > (Real)1e-6)
		{
			Real sinang = glm::sin(angle);
			if (glm::abs(sinang) > (Real)1e-8)
			{
				Real inv2s = (Real)1 / ((Real)2 * sinang);
				Coord axis((R21 - R12) * inv2s, (R02 - R20) * inv2s, (R10 - R01) * inv2s);
				omega = axis * (angle * invDt);
			}
		}

		// Contact point on current triangle.
		typedef typename TPoint3D<Real> Point3D;
		typedef typename TTriangle3D<Real> Triangle3D;
		Triangle3D t3d(b0, b1, b2);
		Point3D p3d(posArr[pId]);
		Coord x_contact = p3d.project(t3d).origin;
		Coord vFace = v_cm + omega.cross(x_contact - c1);

		Real Fn = (bulkForce[pId].dot(n) < 0) ? -bulkForce[pId].dot(n) : (Real)0;
		if (Fn <= (Real)0) return;

		Coord v = velArr[pId];
		Coord vRel = v - vFace;
		Real vn = vRel.dot(n);
		Coord vN = n * vn;
		Coord vT = vRel - vN;
		Real vt = vT.norm();
		if (vt <= (Real)1e-8) return;

		// Coulomb friction impulse cap: |J_t| <= mu * Fn * dt
		// Convert to velocity change: |dv_t| <= mu * Fn * dt / m
		Real maxDvT = friction * Fn * dt * invMass;
		Real newVt = vt - maxDvT;
		if (newVt < (Real)0) newVt = (Real)0;
		Coord vTnew = vT * (newVt / vt);
		velArr[pId] = vFace + vN + vTnew;
	}
	__global__ void SIIS_Narrow_Count_new(
		int pnum,
		uint shapeNum,
		DArrayList<int> triangle_neighbors,
		DArray<uint> shapeIds,
		DArray<uint> count)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pnum) return;
		uint cnt = 0;

		List<int>& nbrTriIds_i = triangle_neighbors[pId];
		int nbrSize = nbrTriIds_i.size();
		for (int ne = 0; ne < nbrSize; ne++)
		{
			int j = nbrTriIds_i[ne];
			if (shapeIds[j] == shapeNum)
			{
				cnt++;
			}
		}
		count[pId] = cnt;
	}

	__global__ void SIIS_Narrow_Set_new(
		int pnum,
		uint shapeNum,
		DArrayList<int> triangle_neighbors,
		DArray<uint> shapeIds,
		DArrayList<int> triangle_neighbors_new)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pnum) return;

		List<int>& nbrTriIds_i_new = triangle_neighbors_new[pId];
		List<int>& nbrTriIds_i = triangle_neighbors[pId];
		int nbrSize = nbrTriIds_i.size();
		for (int ne = 0; ne < nbrSize; ne++)
		{
			int j = nbrTriIds_i[ne];
			if (shapeIds[j] == shapeNum)
			{
				nbrTriIds_i_new.insert(j);
			}
		}
	}




	template<typename TDataType>
	void SemiAnalyticalDensitySolver<TDataType>::compute()
	{
		updatePosition();
	}

	template<typename TDataType>
	void SemiAnalyticalDensitySolver<TDataType>::updatePosition()
	{
		int num = this->inPosition()->size();
		Real dt = this->inTimeStep()->getValue();
		Real  h = this->inSmoothingLength()->getValue();
		Real  d = this->inSamplingDistance()->getValue();
		Real  d_hat = this->varD_hat()->getValue();
		auto scn = this->getSceneGraph();
		Coord  g = scn->getGravity();
		auto& inPos = this->inPosition()->getData();
		auto tsMerge = this->inTriangleSetMerge()->constDataPtr();
		auto& triVertexMerge = tsMerge->getPoints();
		auto& triIndexMerge = tsMerge->triangleIndices();
		auto& vels = this->inVelocity()->getData();
		auto& PreTriangleVerMerge = this->inPreTriangleVerMerge()->getData();
		auto& outKappas = this->outKappas()->getData();
		const Real kappaLower = this->varKappaLower()->getValue();
		Real rho_0 = this->varRestDensity()->getData();
		Real v = glm::pow(d, 3);

		uint shapeSize = tsMerge->shapeSize();
		// Defensive: some scenes may not provide shape IDs, but contact assembly expects at least 1 shape.
		if (shapeSize == 0) shapeSize = 1;
		auto& trishapeIds = tsMerge->shapeIds();

		if (mPosOld.size() != num)
			mPosOld.resize(num);

		if (mPosBuf.size() != num)
			mPosBuf.resize(num);

		if (mPosStart.size() != num)
			mPosStart.resize(num);

		if (mVelStart.size() != num)
			mVelStart.resize(num);

		if (mEnergy.size() != num)
			mEnergy.resize(num);

		if (mSignDis.size() != num)
			mSignDis.resize(num);

		if (mPolyN.size() != num)
			mPolyN.resize(num);

		if (mA.size() != num)
			mA.resize(num);

		if (mAlpha.size() != num)
			mAlpha.resize(num);

		if (mKappa.size() != num)
			mKappa.resize(num);


		// Kappa warm-start across frames:
		// - Ensure outKappas is correctly sized
		// - Initialize once to kappaLower
		// - Clamp each frame to maintain lower bound
		if (outKappas.size() != num)
		{
			outKappas.resize(num);
			cuExecute(num, SIIS_InitKappas<Real>, outKappas, kappaLower);
		}
		else
		{
			cuExecute(num, SIIS_ClampKappasLower<Real>, outKappas, kappaLower);
		}

		GTimer timer;
		timer.start();
		mPosOld.assign(inPos);
		mVelStart.assign(vels);

		cuExecute(num, ccdPreResolveParticles,
			inPos,
			vels,
			triVertexMerge,
			PreTriangleVerMerge,
			triIndexMerge,
			this->inNeighborTriIdsMerge()->getData(),
			mSignDis,
			mPolyN,
			this->varPolynomialNumber()->getValue(),
			this->varWarmStart()->getValue(),
			outKappas,
			kappaLower,
			mEnergy,
			d_hat,
			shapeSize,
			g,
			dt);

#ifdef TIME_LOG
		timer.stop();
		printf("SIIS_CCD_pre: %fms\n", timer.getElapsedTime());
		timer.start();
#endif // TIME_LOG
		mPosStart.assign(inPos);
		auto& edgesMerge = tsMerge->edgeIndices();
		DArray<Coord> edgeNormalMerge, vertexNormalMerge;
		tsMerge->update();

		tsMerge->requestEdgeNormals(edgeNormalMerge);
		tsMerge->requestVertexNormals(vertexNormalMerge, VertexNormalWeightingOption::ANGLE_WEIGHTED);

		auto& tri2edgMerge = tsMerge->triangle2Edge();
		DArray<Real> VelocityNormal;
		VelocityNormal.resize(num);
		DArray<uint> KappaFlag;
		KappaFlag.resize(num);
		KappaFlag.reset();
		CArray<uint> KappaFlagC;
		KappaFlagC.assign(KappaFlag);
		Reduction<Real> reduce;
		DArray<Real> Velocity_sum;
		Velocity_sum.resize(3);
		DArray<Real> Height_sum;
		Height_sum.resize(1);
		Velocity_sum.reset();
		Height_sum.reset();

		DArray<uint> nbrNum;
		DArray<Coord> nearNorm;
		DArray<int> nbrTriId;
		nbrNum.resize(num);
		nearNorm.resize(num);
		nbrTriId.resize(num);
		mCalculateDensity->update();
		cuExecute(num, clampDensityMin,
			mCalculateDensity->outDensity()->getData(),
			this->outDensity()->getData(),
			rho_0
		);

		int iter = 0;
		uint maxIter = this->varIterationNumber()->getValue();
		const Real boundaryFriction = this->varBoundaryFriction()->getValue();
		const Real invMass = (Real)1 / (rho_0 * v + EPSILON);

		// Snapshot last-iteration boundary association for velocity-space friction.
		DArray<Coord> bulkForce;
		if (boundaryFriction > (Real)0)
		{
			bulkForce.resize(num);
		}


#ifdef CONTACT_MODE
		// shapeSize==1: maintain per-particle-per-neighborTriangle state across solver iterations.
		// IMPORTANT: neighbor-triangle lists may not be ready/filled when entering this function,
		// so we lazily initialize (once) right before the first contact assembly that needs it.
		DArray<uint> triNbrCount;
		DArrayList<uint> polyNPerTri;
		DArrayList<uint> varyFlagPerTri;
		DArrayList<Real> kappaPerTri;
		bool polyNPerTriReady = false;
#endif // CONTACT_MODE

		while (iter++ < maxIter)
		{
			printf("---- SIIS Iteration %d ----\n", iter);
			KappaFlag.reset();
			mKappa.reset();
			mEnergy.reset();
			mPosBuf.assign(inPos);
			inPos.reset();
			mA.reset();
			bulkForce.reset();
			nbrNum.reset();
			nearNorm.reset();
			nbrTriId.reset();

			cuFirstOrder(num, this->varKernelType()->currentKey(), this->mScalingFactor,
				SIIS_UpdateFluidParticles,
				inPos,
				mA,
				mKappa,
				mPosBuf,
				this->outDensity()->getData(),
				mCalculateDensity->outBoundaryDensity()->getData(),
				this->inNeighborIds()->getData(),
				h,
				this->varMu()->getValue(),
				dt);



			cuIntegral(num, this->varKernelType()->currentKey(), this->mScalingFactor,
				SIIS_UpdateSemiAnalyticalBoundaryParticles,
				inPos,
				mA,
				mKappa,
				mPosBuf,
				this->outDensity()->getData(),
				triIndexMerge,
				triVertexMerge,
				this->inNeighborTriIdsMerge()->getData(),
				h,
				d,
				this->varMu()->getValue(),
				mPolyN,
				mSignDis,
				KappaFlag,
				nbrNum,
				nearNorm,
				nbrTriId,
				outKappas,
				shapeSize,
				dt);



#ifdef CONTACT_MODE
			// IMPORTANT: Right/Left/CP/CF lists are indexed by "shape" inside kernels.
			// Do NOT size them from nbrNum (which is later reset/reused for per-shape triangle counts).
			DArray<uint> shapeCount;
			shapeCount.resize(num);
			// Fill on host to avoid any possibility of device-fill being optimized out / not executed in the expected stream.
			CArray<uint> hShapeCount;
			hShapeCount.resize(num);
			for (int pi = 0; pi < num; ++pi) hShapeCount[pi] = shapeSize;
			shapeCount.assign(hShapeCount);
			if (iter == 1)
			{
				CArray<uint> dbgShapeCount;
				dbgShapeCount.assign(shapeCount);
				printf("[CONTACT_MODE] shapeSize=%u; shapeCount[0]=%u\n", shapeSize, (unsigned)dbgShapeCount[0]);
			}

			DArrayList<Coord> CFList;
			DArrayList<Real> CPList;
			DArrayList<Real> LeftList;
			DArrayList<Coord> RightList;
			CFList.resize(shapeCount);
			CPList.resize(shapeCount);
			LeftList.resize(shapeCount);
			RightList.resize(shapeCount);
			shapeCount.clear();
			for (int i = 0; i < shapeSize; i++)
			{
				DArrayList<int> TriNbrNumList;
				nbrNum.reset();
				int sum;
				cuExecute(num,
					SIIS_Narrow_Count_new,
					num,
					i,
					this->inNeighborTriIdsMerge()->getData(),
					trishapeIds,
					nbrNum);
				TriNbrNumList.resize(nbrNum);
				sum = mReduce.accumulate(nbrNum.begin(), nbrNum.size());
				if (sum > 0)
				{
					cuExecute(num,
						SIIS_Narrow_Set_new,
						num,
						i,
						this->inNeighborTriIdsMerge()->getData(),
						trishapeIds,
						TriNbrNumList);
				}
				cuExecute(num,
					SIIS_UpdateSemiAnalyticalContactPotential,
					RightList,
					LeftList,
					mPosBuf,
					triIndexMerge,
					triVertexMerge,
					edgesMerge,
					tri2edgMerge,
					edgeNormalMerge,
					vertexNormalMerge,
					TriNbrNumList,
					i,
					mPolyN,
					v,
					d_hat,
					CPList,
					CFList,
					dt);
				TriNbrNumList.clear();
			}

			cuExecute(num,
				SIIS_SumSemiAnalyticalContactPotential,
				RightList,
				LeftList,
				inPos,
				mA,
				KappaFlag,
				this->varKappaLower()->getValue(),
				mKappa,
				outKappas,
				mEnergy,
				d_hat,
				mSignDis,
				nearNorm,
				CPList,
				CFList
			);
			CFList.clear();
			CPList.clear();
			LeftList.clear();
			RightList.clear();

			bulkForce.assign(mKappa);
#endif // CONTACT_MODE



			cuExecute(num,
				relaxPositionEnergy,
				inPos,
				mPosBuf,
				mPosStart,
				mEnergy,
				mA,
				mAlpha,
				this->outDensity()->getData(),
				d,
				this->varMu()->getValue(),
				dt);
			mCalculateDensity->update();
			cuExecute(num, clampDensityMin,
				mCalculateDensity->outDensity()->getData(),
				this->outDensity()->getData(),
				rho_0
			);
		}

#ifdef CONTACT_MODE
		if (polyNPerTriReady)
		{
			polyNPerTri.clear();
			varyFlagPerTri.clear();
			kappaPerTri.clear();
			triNbrCount.clear();
		}
#endif // CONTACT_MODE
#ifdef TIME_LOG
		timer.stop();
		printf("SIIS_Iteration: %fms\n", timer.getElapsedTime());
		timer.start();
#endif // TIME_LOG

		cuExecute(num, updateVelocityFromPositions,
			vels,
			mVelStart,
			inPos,
			mPosOld,
			dt);

		if (boundaryFriction > (Real)0)
		{
			cuExecute(num, SIIS_ApplyBoundaryTangentialFriction,
				vels,
				inPos,
				mSignDis,
				nearNorm,
				nbrTriId,
				bulkForce,
				triIndexMerge,
				triVertexMerge,
				PreTriangleVerMerge,
				d_hat,
				boundaryFriction,
				invMass,
				dt);
		}
		nbrNum.clear();
		bulkForce.clear();
		nearNorm.clear();
		nbrTriId.clear();

		CArray<Coord> PosC;
		PosC.assign(inPos);
		PosC.clear();
		edgeNormalMerge.clear();
		vertexNormalMerge.clear();
		KappaFlag.clear();
		KappaFlagC.clear();
		VelocityNormal.clear();
		Velocity_sum.clear();
		Height_sum.clear();


		//this->outDensity()->getData().assign(mCalculateDensity->outBoundaryDensity()->getData());
#ifdef TIME_LOG
		timer.stop();
		printf("SIIS_UpdateVelocity: %f\n", timer.getElapsedTime());
#endif // TIME_LOG
	}



	DEFINE_CLASS(SemiAnalyticalDensitySolver);
}
