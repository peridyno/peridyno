#include "SemiAnalyticalPBD.h"

#include "SemiAnalyticalSummationDensity.h"
#include "IntersectionArea.h"

namespace dyno
{
	IMPLEMENT_TCLASS(SemiAnalyticalPBD, TDataType)

	template <typename TDataType>
	SemiAnalyticalPBD<TDataType>::SemiAnalyticalPBD()
		: ConstraintModule()
	{
		this->inSamplingDistance()->setValue(0.005);

		mCalculateDensity = std::make_shared<SemiAnalyticalSummationDensity<TDataType>>();
		this->inSmoothingLength()->connect(mCalculateDensity->inSmoothingLength());
		this->inSamplingDistance()->connect(mCalculateDensity->inSamplingDistance());
		this->inPosition()->connect(mCalculateDensity->inPosition());
		this->inNeighborParticleIds()->connect(mCalculateDensity->inNeighborIds());
		this->inNeighborTriangleIds()->connect(mCalculateDensity->inNeighborTriIds());
		this->inTriangleIndex()->connect(mCalculateDensity->inTriangleInd());
		this->inTriangleVertex()->connect(mCalculateDensity->inTriangleVer());
	}

	template <typename TDataType>
	SemiAnalyticalPBD<TDataType>::~SemiAnalyticalPBD()
	{
		mLamda.clear();
		mDeltaPos.clear();
		mPosBuffer.clear();
	}

	template <typename TDataType>
	void SemiAnalyticalPBD<TDataType>::constrain()
	{
		int num = this->inPosition()->size();

		if (mLamda.size() != num) {
			mLamda.resize(num);
			mDeltaPos.resize(num);
			mPosBuffer.resize(num);
		}

		mPosBuffer.assign(this->inPosition()->getData());

		int iter = 0;
		auto maxIter = this->varInterationNumber()->getData();
		while (iter++ < maxIter) {
			takeOneIteration();
		}

		updateVelocity();
	}

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

	template <typename Real, typename Coord>
	__global__ void K_ComputeLambdasMesh(
		DArray<Real> lambdaArr,
		DArray<Real> rhoArr,
		DArray<Coord> posArr,
		DArray<TopologyModule::Triangle> Tri,
		DArray<Coord>  positionTri,
		DArrayList<int> neighbors,
		DArrayList<int> neighborsTri,
		SpikyKernel<Real> kern,
		Real smoothingLength,
		Real samplingDistance)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size())
			return;

		List<int>& triList_i = neighborsTri[pId];
		int  nbSizeTri = triList_i.size();
		Real dis_n = REAL_MAX;

		//semi-analytical boundary integration
		Real  lamda_i = Real(0);
		Coord pos_i = posArr[pId];
		Coord grad_ci(0);
		for (int ne = 0; ne < nbSizeTri; ne++)
		{
			int j = triList_i[ne];
			Triangle3D t3d(positionTri[Tri[j][0]], positionTri[Tri[j][1]], positionTri[Tri[j][2]]);
			Plane3D	PL(positionTri[Tri[j][0]], t3d.normal());
			Point3D	p3d(pos_i);
			Point3D	nearest_pt = p3d.project(PL);
			Real r = (nearest_pt.origin - pos_i).norm();

			Real  AreaSum = calculateIntersectionArea(p3d, t3d, smoothingLength);  //A_s in equation 10
			Real  MinDistance = (p3d.distance(t3d));							//d_n (scalar) in equation 10
			Coord Min_Pt = (p3d.project(t3d)).origin - pos_i;					//d_n (vector) in equation 10
			Coord Min_Pos = p3d.project(t3d).origin;
			if (ne < nbSizeTri - 1 && triList_i[ne + 1] < 0)
			{
				//triangle clustering
				int jn;
				do
				{
					jn = triList_i[ne + 1];

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
					/ (samplingDistance * samplingDistance * samplingDistance)
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
		List<int>& parList_i = neighbors[pId];
		int  nbSize = parList_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = parList_i[ne];
			
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
	__global__ void K_ComputeDisplacementMesh(
		DArray<Coord>                    dPos,
		DArray<Real>                     lambdas,
		DArray<Coord>                    posArr,
		DArray<TopologyModule::Triangle> Tri,
		DArray<Coord>                    positionTri,
		DArrayList<int>                     neighbors,
		DArrayList<int>                     neighborsTri,
		SpikyKernel<Real>                     kern,
		Real  smoothingLength,
		Real  dt,
		Real  sampling_distance)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size())
			return;

		Coord pos_i = posArr[pId];
		Real  lamda_i = lambdas[pId];

		Coord dP_i(0);
		List<int>& list_i = neighbors[pId];
		int   nbSize = list_i.size();

		List<int>& triList_i = neighborsTri[pId];
		int   nbSizeTri = triList_i.size();
		Real  dis_n = REAL_MAX;
		for (int ne = 0; ne < nbSizeTri; ne++)
		{
			int j = triList_i[ne];
			Triangle3D t3d(positionTri[Tri[j][0]], positionTri[Tri[j][1]], positionTri[Tri[j][2]]);
			Plane3D    PL(positionTri[Tri[j][0]], t3d.normal());
			Point3D    p3d(pos_i);
			if (abs(p3d.distance(t3d)) < abs(dis_n))
			{
				dis_n = p3d.distance(t3d);
			}
		}
		//semi-analytical boundary integration
		for (int ne = 0; ne < nbSizeTri; ne++)
		{
			int j = triList_i[ne];
			Triangle3D t3d(positionTri[Tri[j][0]], positionTri[Tri[j][1]], positionTri[Tri[j][2]]);
			Plane3D    PL(positionTri[Tri[j][0]], t3d.normal());
			Point3D    p3d(pos_i);
			//Point3D nearest_pt = p3d.project(t3d);
			Point3D nearest_pt = p3d.project(PL);
			Real    r = (nearest_pt.origin - pos_i).norm();

			Real  tmp = 1.0;
			float d = p3d.distance(PL);

			Coord ttm = PL.normal;
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
		}
	}

	template <typename Coord>
	__global__ void K_UpdatePositionMesh(
		DArray<Coord> posArr,
		DArray<Coord> dPos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size())
			return;

		posArr[pId] += dPos[pId];
	}

	template <typename Real, typename Coord>
	__global__ void DP_UpdateVelocityMesh(
		DArray<Coord> velArr,
		DArray<Coord> prePos,
		DArray<Coord> curPos,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.size())
			return;

		velArr[pId] += (curPos[pId] - prePos[pId]) / dt;
	}

	template <typename TDataType>
	void SemiAnalyticalPBD<TDataType>::takeOneIteration()
	{
		auto& inPos = this->inPosition()->getData();

		mDeltaPos.reset();
		mCalculateDensity->update();

		int num = inPos.size();

		cuExecute(num,
			K_ComputeLambdasMesh,
			mLamda,
			mCalculateDensity->outDensity()->getData(),
			inPos,
			this->inTriangleIndex()->getData(),
			this->inTriangleVertex()->getData(),
			this->inNeighborParticleIds()->getData(),
			this->inNeighborTriangleIds()->getData(),
			m_kernel,
			this->inSmoothingLength()->getData(),
			this->inSamplingDistance()->getData());

		cuExecute(num,
			K_ComputeDisplacementMesh,
			mDeltaPos,
			mLamda,
			inPos,
			this->inTriangleIndex()->getData(),
			this->inTriangleVertex()->getData(),
			this->inNeighborParticleIds()->getData(),
			this->inNeighborTriangleIds()->getData(),
			m_kernel,
			this->inSmoothingLength()->getData(),
			this->inTimeStep()->getData(),
			this->inSamplingDistance()->getData());

		cuExecute(num,
			K_UpdatePositionMesh,
			inPos,
			mDeltaPos);
	}

	template <typename TDataType>
	void SemiAnalyticalPBD<TDataType>::updateVelocity()
	{
		int  num = this->inPosition()->size();
		cuExecute(num,
			DP_UpdateVelocityMesh,
			this->inVelocity()->getData(),
			mPosBuffer,
			this->inPosition()->getData(),
			this->inTimeStep()->getData());
	}

	DEFINE_CLASS(SemiAnalyticalPBD);
}