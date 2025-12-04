#include "EulerianSimSplitting2D.h"
#include"../Core/Backend/Cuda/SparseMatrix/SparseMatrix.h"
#include "Algorithm/Function2Pt.h"
//#include "Volume/AdaptiveVolumeFromCircle.h"
#include <thrust/sort.h>
#include <ctime>
#include "Timer.h"

//#include <D:/CFD/eigen-3.4.0/Eigen/Sparse>
//#include <D:/CFD/eigen-3.4.0/Eigen/SparseLU>
//#include <D:/CFD/eigen-3.4.0/Eigen/SparseQR>
//#include <D:/CFD/eigen-3.4.0/Eigen/SparseCholesky>
//#include <D:/CFD/eigen-3.4.0/Eigen/IterativeLinearSolvers>
//#include <D:/CFD/eigen-3.4.0/Eigen/SuperLUSupport>

namespace dyno
{
	IMPLEMENT_TCLASS(EulerianSimSplitting2D, TDataType)

	template<typename TDataType>
	EulerianSimSplitting2D<TDataType>::EulerianSimSplitting2D()
		: EulerianSim<TDataType>()
	{
	}

	template<typename TDataType>
	EulerianSimSplitting2D<TDataType>::~EulerianSimSplitting2D()
	{
		m_node.clear();
		m_sdf.clear();
		m_neighbor.clear();
		mNode2Ver.clear();
		mVer2Node.clear();

		m_identifier.clear();
		m_velocity.clear();
		m_pressure.clear();
		m_density.clear();

		//m_NodeIndex.clear();
		//m_NodeParticleNum.clear();

		//matrix_count.clear();
	}

	template <typename Real, typename Coord, typename Coord2D>
	__global__ void ESS_ClassifyBoundary2D(
		DArray<CellType> identifier,
		DArray<Coord> zpos,
		DArray<Coord> pos,
		DArray<AdaptiveGridNode2D> octree,
		DArray<Real> sdf,
		Real dx,
		Coord2D center,
		Real radius)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= octree.size()) return;

		//octree[tId].m_position[2] = 0.0f;
		pos[tId] = Coord(octree[tId].m_position[0], octree[tId].m_position[1], 0.0f);

		Real dist = (octree[tId].m_position - center).norm();
		sdf[tId] = dist - radius;

		if (sdf[tId] < 0)
		{
			identifier[tId] = CellType::Inside;
			zpos[tId] = Coord(1.0f, 0.0f, 0.0f);
		}
		//else if (octree[tId].m_position[1] < 0.001)
		//{
		//	identifier[tId] = CellType::Inlet1;
		//	zpos[tId] = Coord(2.0f, 0.0f, 0.0f);
		//}
		else if ((octree[tId].m_position[1]) > 0.5)
		{
			identifier[tId] = CellType::Outlet1;
			zpos[tId] = Coord(3.0f, 0.0f, 0.0f);
		}
		else
		{
			identifier[tId] = CellType::Static;
			zpos[tId] = Coord(0.0f, 0.0f, 0.0f);
		}
	}

	template<typename TDataType>
	void EulerianSimSplitting2D<TDataType>::node_topology()
	{
		auto volumeSet = this->inAdaptiveVolume2D()->getDataPtr();
		volumeSet->extractLeafs(m_node, m_neighbor);
		if (m_node.size() == 0) return;
		Real dx = volumeSet->adaptiveGridDx2D();

		DArray<Coord> zpos(m_node.size());
		DArray<Coord> points(m_node.size());
		m_identifier.resize(m_node.size());
		m_sdf.resize(m_node.size());
		cuExecute(m_node.size(),
			ESS_ClassifyBoundary2D,
			m_identifier,
			zpos,
			points,
			m_node,
			m_sdf,
			dx,
			this->inCenter()->getData(),
			this->inRadius()->getData());
		points.clear();

		this->outNodeType()->allocate();
		this->outNodeType()->assign(zpos);
		zpos.clear();

		//this->outLeafNodes()->allocate();
		//this->outLeafNodes()->getDataPtr()->setPoints(points);

		m_velocity.resize(m_identifier.size());
		m_pressure.resize(m_identifier.size());
		m_density.resize(m_identifier.size());
		m_velocity.reset();
		m_pressure.reset();
		m_density.reset();

		DArray<Coord2D> vertex;
		volumeSet->extractVertexs(vertex, mNode2Ver, mVer2Node);
		vertex.clear();
	}

	template <typename Coord, typename Coord2D>
	__global__ void ESS_2DTo3D(
		DArray<Coord> outvel,
		DArray<Coord2D> vel)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vel.size()) return;

		outvel[tId] = Coord(vel[tId][0], vel[tId][1], 0.0f);
	}

	template<typename TDataType>
	void EulerianSimSplitting2D<TDataType>::resetStates()
	{
		node_topology();

		//vertex_topology();

		this->outVelocity()->allocate();
		auto& outvel = this->outVelocity()->getData();
		outvel.resize(m_velocity.size());
		cuExecute(m_velocity.size(),
			ESS_2DTo3D,
			outvel,
			m_velocity);

		this->outPressure()->allocate();
		this->outPressure()->assign(m_density);
	}

	template <typename Coord2D, typename Real>
	__global__ void ESS_ParticleToVertex(
		DArray<Coord2D> vvelocity,
		DArray<Real> vcoef,
		DArray<Real> vvol,
		DArray<Coord2D> particle,
		DArray<Coord2D> pvelocity,
		DArray<int> pindex,
		DArray<AdaptiveGridNode2D> node,
		DArray<int> node2ver,
		Real sandv,
		Real dx,
		Level level)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= particle.size()) return;

		if (pindex[tId] == EMPTY) return;

		int nind = pindex[tId];

		int vind1 = node2ver[4 * nind];
		int vind2 = node2ver[4 * nind + 1];
		int vind3 = node2ver[4 * nind + 2];
		int vind4 = node2ver[4 * nind + 3];

		Real up_dx = dx * (1 << (level - node[nind].m_level));
		Coord2D resid = particle[tId] - (node[nind].m_position - Coord2D(0.5*up_dx, 0.5*up_dx));
		Real x_resid = resid[0];
		Real y_resid = resid[1];

		Real coef1 = (x_resid / up_dx) * (y_resid / up_dx);
		atomicAdd(&(vcoef[vind1]), coef1);
		atomicAdd(&(vvol[vind1]), coef1 * sandv);
		atomicAdd(&(vvelocity[vind1][0]), coef1 * (pvelocity[tId][0]));
		atomicAdd(&(vvelocity[vind1][1]), coef1 * (pvelocity[tId][1]));

		Real coef2 = (1.0f - x_resid / up_dx) * (y_resid / up_dx);
		atomicAdd(&(vcoef[vind2]), coef2);
		atomicAdd(&(vvol[vind2]), coef2 * sandv);
		atomicAdd(&(vvelocity[vind2][0]), coef2 * (pvelocity[tId][0]));
		atomicAdd(&(vvelocity[vind2][1]), coef2 * (pvelocity[tId][1]));

		Real coef3 = (1.0f - x_resid / up_dx) *(1.0f - y_resid / up_dx);
		atomicAdd(&(vcoef[vind3]), coef3);
		atomicAdd(&(vvol[vind3]), coef3 * sandv);
		atomicAdd(&(vvelocity[vind3][0]), coef3 * (pvelocity[tId][0]));
		atomicAdd(&(vvelocity[vind3][1]), coef3 * (pvelocity[tId][1]));

		Real coef4 = (x_resid / up_dx) * (1.0f - y_resid / up_dx);
		atomicAdd(&(vcoef[vind4]), coef4);
		atomicAdd(&(vvol[vind4]), coef4 * sandv);
		atomicAdd(&(vvelocity[vind4][0]), coef4 * (pvelocity[tId][0]));
		atomicAdd(&(vvelocity[vind4][1]), coef4 * (pvelocity[tId][1]));
	}

	template <typename Coord2D, typename Real>
	__global__ void ESS_VertexNormalization(
		DArray<Real> vvol,
		DArray<Coord2D> vvelocity,
		DArray<Real> vcoef,
		DArray<AdaptiveGridNode2D> node,
		DArrayList<int> mVer2Node,
		Real dx,
		Level level)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vcoef.size()) return;

		if (vcoef[tId] < 0.01*dx*dx)
		{
			vvelocity[tId] = Coord2D(0.0f, 0.0f);
			vvol[tId] = Real(0.0f);
		}
		else
		{
			vvelocity[tId] = vvelocity[tId] / vcoef[tId];

			Real volume = 0;
			List<int>& list = mVer2Node[tId];
			for (int i = 0; i < list.size(); i++)
			{
				int nind = list[i];
				Real up_dx = dx * (1 << (level - node[nind].m_level));
				volume += up_dx * up_dx*0.25;
			}
			Real svolume = vvol[tId];
			if (svolume > volume) svolume = volume;
			vvol[tId] = svolume / volume;
		}
	}

	template <typename Coord2D, typename Real>
	__global__ void ESS_VertexDangling(
		DArray<Real> vdensity,
		DArray<Coord2D> vvelocity,
		DArray<AdaptiveGridNode2D> node,
		DArrayList<int> neighbor,
		DArray<int> node2ver)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= node.size()) return;

		if (neighbor[4 * tId].size() > 1)
		{
			int dindex = EMPTY;
			int nindex = neighbor[4 * tId][0];
			if (node[nindex].m_position[1] > node[tId].m_position[1])
				dindex = node2ver[4 * nindex + 1];
			else
				dindex = node2ver[4 * nindex + 2];

			vdensity[dindex] = (vdensity[node2ver[4 * tId]] + vdensity[node2ver[4 * tId + 3]]) / 2;
			vvelocity[dindex] = (vvelocity[node2ver[4 * tId]] + vvelocity[node2ver[4 * tId + 3]]) / 2;
		}

		if (neighbor[4 * tId + 1].size() > 1)
		{
			int dindex = EMPTY;
			int nindex = neighbor[4 * tId + 1][0];
			if (node[nindex].m_position[1] > node[tId].m_position[1])
				dindex = node2ver[4 * nindex];
			else
				dindex = node2ver[4 * nindex + 3];

			vdensity[dindex] = (vdensity[node2ver[4 * tId + 1]] + vdensity[node2ver[4 * tId + 2]]) / 2;
			vvelocity[dindex] = (vvelocity[node2ver[4 * tId + 1]] + vvelocity[node2ver[4 * tId + 2]]) / 2;
		}

		if (neighbor[4 * tId + 2].size() > 1)
		{
			int dindex = EMPTY;
			int nindex = neighbor[4 * tId + 2][0];
			if (node[nindex].m_position[0] > node[tId].m_position[0])
				dindex = node2ver[4 * nindex + 3];
			else
				dindex = node2ver[4 * nindex + 2];

			vdensity[dindex] = (vdensity[node2ver[4 * tId]] + vdensity[node2ver[4 * tId + 1]]) / 2;
			vvelocity[dindex] = (vvelocity[node2ver[4 * tId]] + vvelocity[node2ver[4 * tId + 1]]) / 2;
		}

		if (neighbor[4 * tId + 3].size() > 1)
		{
			int dindex = EMPTY;
			int nindex = neighbor[4 * tId + 3][0];
			if (node[nindex].m_position[0] > node[tId].m_position[0])
				dindex = node2ver[4 * nindex];
			else
				dindex = node2ver[4 * nindex + 1];

			vdensity[dindex] = (vdensity[node2ver[4 * tId + 2]] + vdensity[node2ver[4 * tId + 3]]) / 2;
			vvelocity[dindex] = (vvelocity[node2ver[4 * tId + 2]] + vvelocity[node2ver[4 * tId + 3]]) / 2;
		}
	}

	template <typename Coord2D, typename Real>
	__global__ void ESS_VertexPlusToNode(
		DArray<Coord2D> nvelocity,
		DArray<Real> ndensity,
		DArray<CellType> identifier,
		DArray<Coord2D> vvelocity,
		DArray<Real> vdensity,
		DArray<int> node2ver)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nvelocity.size()) return;

		if (identifier[tId] == CellType::Inside)
		{
			int vind1 = node2ver[4 * tId];
			int vind2 = node2ver[4 * tId + 1];
			int vind3 = node2ver[4 * tId + 2];
			int vind4 = node2ver[4 * tId + 3];


			Coord2D pv = (vvelocity[vind1] + vvelocity[vind2] + vvelocity[vind3] + vvelocity[vind4]) / 4;
			Real sandc = (vdensity[vind1] + vdensity[vind2] + vdensity[vind3] + vdensity[vind4]) / 4;

			nvelocity[tId] = sandc * pv + (1.0f - sandc)*nvelocity[tId];
			//nvelocity[tId] += pv;
			//nvelocity[tId][2] = 0.0f;

			ndensity[tId] = sandc;
		}
		else
		{
			nvelocity[tId] = Coord2D(0.0f, 0.0f);
			ndensity[tId] = -1.0f;
		}
	}

	template <typename Coord, typename Coord2D>
	__global__ void ESS_3DTo2D(
		DArray<Coord2D> ppos,
		DArray<Coord> particles)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= particles.size()) return;

		ppos[tId] = Coord2D(particles[tId][0], particles[tId][1]);
	}
	template<typename TDataType>
	void EulerianSimSplitting2D<TDataType>::particlesToGrids()
	{
		auto volumeSet = this->inAdaptiveVolume2D()->getDataPtr();
		Level level = volumeSet->adaptiveGridLevelMax2D();
		Real dx = volumeSet->adaptiveGridDx2D();

		auto& particles = this->statePPosition()->getData();
		DArray<Coord2D> particles_pos(particles.size());
		cuExecute(particles.size(),
			ESS_3DTo2D,
			particles_pos,
			particles);
		DArray<int> nodeIndex(particles.size());
		volumeSet->accessRandom(nodeIndex, particles_pos);

		Real sdx = this->varSamplingDistance()->getData();
		auto& pvelocity = this->statePVelocity()->getData();
		DArray<Coord2D> pvelocity2D(pvelocity.size());
		cuExecute(pvelocity.size(),
			ESS_3DTo2D,
			pvelocity2D,
			pvelocity);

		DArray<Coord2D> vvelocity(mVer2Node.size());
		vvelocity.reset();
		DArray<Real> vcoefficient(mVer2Node.size());
		vcoefficient.reset();
		DArray<Real> vvolume(mVer2Node.size());
		vvolume.reset();
		cuExecute(particles.size(),
			ESS_ParticleToVertex,
			vvelocity,
			vcoefficient,
			vvolume,
			particles_pos,
			pvelocity2D,
			nodeIndex,
			m_node,
			mNode2Ver,
			sdx*sdx,
			dx,
			level);
		nodeIndex.clear();
		particles_pos.clear();
		pvelocity2D.clear();

		cuExecute(mVer2Node.size(),
			ESS_VertexNormalization,
			vvolume,
			vvelocity,
			vcoefficient,
			m_node,
			mVer2Node,
			dx,
			level);
		vcoefficient.clear();

		cuExecute(m_node.size(),
			ESS_VertexDangling,
			vvolume,
			vvelocity,
			m_node,
			m_neighbor,
			mNode2Ver);

		cuExecute(m_node.size(),
			ESS_VertexPlusToNode,
			m_velocity,
			m_density,
			m_identifier,
			vvelocity,
			vvolume,
			mNode2Ver);

		vvelocity.clear();
		vvolume.clear();
		//printf("EulerianSimSplitting2D: particlesToGrids ok!!!  \n");
	}

	template <typename Coord2D, typename Real>
	__global__ void ESS_NodeToVertex(
		DArray<Coord2D> vvelocity,
		DArray<Real> vcoef,
		DArray<Coord2D> nvelocity,
		DArray<int> node2ver,
		DArray<Real> density)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nvelocity.size()) return;

		int vind1 = node2ver[4 * tId];
		int vind2 = node2ver[4 * tId + 1];
		int vind3 = node2ver[4 * tId + 2];
		int vind4 = node2ver[4 * tId + 3];

		Coord2D nvel = nvelocity[tId] * density[tId];

		atomicAdd(&(vcoef[vind1]), 0.25);
		atomicAdd(&(vvelocity[vind1][0]), 0.25*(nvel[0]));
		atomicAdd(&(vvelocity[vind1][1]), 0.25*(nvel[1]));

		atomicAdd(&(vcoef[vind2]), 0.25);
		atomicAdd(&(vvelocity[vind2][0]), 0.25*(nvel[0]));
		atomicAdd(&(vvelocity[vind2][1]), 0.25*(nvel[1]));

		atomicAdd(&(vcoef[vind3]), 0.25);
		atomicAdd(&(vvelocity[vind3][0]), 0.25*(nvel[0]));
		atomicAdd(&(vvelocity[vind3][1]), 0.25*(nvel[1]));

		atomicAdd(&(vcoef[vind4]), 0.25);
		atomicAdd(&(vvelocity[vind4][0]), 0.25*(nvel[0]));
		atomicAdd(&(vvelocity[vind4][1]), 0.25*(nvel[1]));
	}

	template <typename Coord2D, typename Real>
	__global__ void ESS_VertexNormalization2(
		DArray<Coord2D> vvelocity,
		DArray<Real> vcoef,
		Real dx)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vvelocity.size()) return;

		if (vcoef[tId] < 0.01*dx*dx)
			vvelocity[tId] = Coord2D(0.0f, 0.0f);
		else
			vvelocity[tId] = vvelocity[tId] / vcoef[tId];
	}

	template <typename Coord, typename Real, typename Coord2D>
	__global__ void ESS_VertexToParticle(
		DArray<Coord> pvelocity,
		DArray<Coord2D> particle,
		DArray<int> pindex,
		DArray<AdaptiveGridNode2D> node,
		DArray<int> node2ver,
		DArray<Coord2D> vvelocity,
		Real dx,
		Level level,
		Real coef)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= particle.size()) return;

		if (pindex[tId] == EMPTY) return;

		int nind = pindex[tId];
		int vind1 = node2ver[4 * nind];
		int vind2 = node2ver[4 * nind + 1];
		int vind3 = node2ver[4 * nind + 2];
		int vind4 = node2ver[4 * nind + 3];

		Real up_dx = dx * (1 << (level - node[nind].m_level));
		Coord2D resid = particle[tId] - (node[nind].m_position - Coord2D(0.5*up_dx, 0.5*up_dx));
		Real x_resid = resid[0];
		Real y_resid = resid[1];

		//atomicCAS

		Real vcoef1 = (x_resid / up_dx) * (y_resid / up_dx);
		Real vcoef2 = (1 - x_resid / up_dx) * (y_resid / up_dx);
		Real vcoef3 = (1 - x_resid / up_dx) *(1 - y_resid / up_dx);
		Real vcoef4 = (x_resid / up_dx) * (1 - y_resid / up_dx);

		Coord2D upvelocity = coef * (vcoef1 * vvelocity[vind1] + vcoef2 * vvelocity[vind2] + vcoef3 * vvelocity[vind3] + vcoef4 * vvelocity[vind4]);
		pvelocity[tId] += Coord(upvelocity[0], upvelocity[1], 0.0f);
	}

	template<typename TDataType>
	void EulerianSimSplitting2D<TDataType>::gridsToParticles(DArray<Coord2D>& velocity)
	{
		auto volumeSet = this->inAdaptiveVolume2D()->getDataPtr();
		Level level = volumeSet->adaptiveGridLevelMax2D();
		Real dx = volumeSet->adaptiveGridDx2D();

		DArray<Coord2D> vvelocity(mVer2Node.size());
		vvelocity.reset();
		DArray<Real> vcoefficient(mVer2Node.size());
		vcoefficient.reset();
		cuExecute(m_node.size(),
			ESS_NodeToVertex,
			vvelocity,
			vcoefficient,
			velocity,
			mNode2Ver,
			m_density);

		cuExecute(mVer2Node.size(),
			ESS_VertexNormalization2,
			vvelocity,
			vcoefficient,
			dx);
		cuExecute(m_node.size(),
			ESS_VertexDangling,
			vcoefficient,
			vvelocity,
			m_node,
			m_neighbor,
			mNode2Ver);
		vcoefficient.clear();

		auto& particles = this->statePPosition()->getData();
		DArray<Coord2D> particles_pos(particles.size());
		cuExecute(particles.size(),
			ESS_3DTo2D,
			particles_pos,
			particles);
		DArray<int> nodeIndex(particles.size());
		volumeSet->accessRandom(nodeIndex, particles_pos);
		cuExecute(particles.size(),
			ESS_VertexToParticle,
			this->statePVelocity()->getData(),
			particles_pos,
			nodeIndex,
			m_node,
			mNode2Ver,
			vvelocity,
			dx,
			level,
			this->varUpdateCoefficient()->getData());
		nodeIndex.clear();
		vvelocity.clear();
		particles_pos.clear();
		//printf("EulerianSimSplitting2D: gridsToParticles ok!!!  \n");
	}

	template <typename Coord2D>
	__global__ void ESS_VertexToNode(
		DArray<Coord2D> nvelocity,
		DArray<CellType> identifier,
		DArray<Coord2D> vvelocity,
		DArray<int> node2ver)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nvelocity.size()) return;

		if (identifier[tId] == CellType::Inside)
		{
			int vind1 = node2ver[4 * tId];
			int vind2 = node2ver[4 * tId + 1];
			int vind3 = node2ver[4 * tId + 2];
			int vind4 = node2ver[4 * tId + 3];

			nvelocity[tId] = (vvelocity[vind1] + vvelocity[vind2] + vvelocity[vind3] + vvelocity[vind4]) / 4;
			//nvelocity[tId][2] = 0.0f;
		}
		else
			nvelocity[tId] = Coord2D(0.0f, 0.0f);
	}

	template<typename TDataType>
	void EulerianSimSplitting2D<TDataType>::interpolate_previous(DArray<Coord2D>& position, DArray<Coord2D>& velocity)
	{
		auto volumeSet = this->inAdaptiveVolume2D()->getDataPtr();
		Level level = volumeSet->adaptiveGridLevelMax2D();
		Real dx = volumeSet->adaptiveGridDx2D();
		DArray<int> index(position.size());
		volumeSet->accessRandom(index, position);

		DArray<Coord2D> vvelocity(mVer2Node.size());
		vvelocity.reset();
		DArray<Real> vcoefficient(mVer2Node.size());
		vcoefficient.reset();
		DArray<Real> vvolume(mVer2Node.size());
		//DArray<uint> node_num(m_identifier.size());
		cuExecute(position.size(),
			ESS_ParticleToVertex,
			vvelocity,
			vcoefficient,
			vvolume,
			position,
			velocity,
			index,
			m_node,
			mNode2Ver,
			dx,
			dx,
			level);
		vvolume.clear();
		index.clear();

		cuExecute(mVer2Node.size(),
			ESS_VertexNormalization2,
			vvelocity,
			vcoefficient,
			dx);
		cuExecute(m_node.size(),
			ESS_VertexDangling,
			vcoefficient,
			vvelocity,
			m_node,
			m_neighbor,
			mNode2Ver);
		vcoefficient.clear();

		cuExecute(m_node.size(),
			ESS_VertexToNode,
			m_velocity,
			m_identifier,
			vvelocity,
			mNode2Ver);

		vvelocity.clear();
		//printf("EulerianSimSplitting2D: interpolate_previous ok!!!  \n");
	}

	template <typename Coord2D>
	__global__ void ESS_SubtractVelocity(
		DArray<Coord2D> deltav,
		DArray<Coord2D> nowv,
		DArray<Coord2D> oldv,
		DArray<CellType> identifier)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= deltav.size()) return;

		if (identifier[tId] == CellType::Inside)
		{
			deltav[tId] = nowv[tId] - oldv[tId];
			//deltav[tId][2] = 0.0f;
		}
		else
			deltav[tId] = Coord2D(0.0f, 0.0f);
	}

	template <typename Coord2D>
	__global__ void ESS_GetPreviousVelocity(
		DArray<Coord2D> pos,
		DArray<Coord2D> vel,
		DArray<AdaptiveGridNode2D> node,
		DArray<Coord2D> velocity)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= node.size()) return;

		vel[tId] = velocity[tId];
		pos[tId] = node[tId].m_position;
	}

	template<typename TDataType>
	void EulerianSimSplitting2D<TDataType>::updateStates()
	{
		//GTimer timer;

		auto dt = this->stateTimeStep()->getData();

		DArray<Coord2D> old_velocity(m_identifier.size());
		DArray<Coord2D> old_points(m_identifier.size());
		cuExecute(m_identifier.size(),
			ESS_GetPreviousVelocity,
			old_points,
			old_velocity,
			m_node,
			m_velocity);

		node_topology();

		//printf("UpdateStates:  %d  %d  \n", old_points.size(), m_identifier.size());

		interpolate_previous(old_points, old_velocity);

		DArray<Coord> vel3d(m_velocity.size());
		cuExecute(m_velocity.size(),
			ESS_2DTo3D,
			vel3d,
			m_velocity);
		Reduction<Coord> reduce;
		Coord maxvel = reduce.maximum(vel3d.begin(), vel3d.size());
		vel3d.clear();
		//Reduction<Coord2D> reduce;
		//Coord2D maxvel = reduce.maximum(m_velocity.begin(), m_velocity.size());
		printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~UpdateStates velocity1111:  %f  %f , %f \n", maxvel[0], maxvel[1], sqrt(maxvel[0] * maxvel[0] + maxvel[1] * maxvel[1]));

		extrapolate2D();
		advect2D(dt);

		//timer.start();
		particlesToGrids();
		//timer.stop();
		//printf("particlesToGrids:   %f \n", timer.getElapsedTime());


		old_velocity.assign(m_velocity);

		solve_pressure2D(m_pressure, m_velocity, dt);

		update_velocity2D(m_velocity, m_pressure, dt);

		DArray<Coord2D> delta_velocity(m_identifier.size());
		cuExecute(m_identifier.size(),
			ESS_SubtractVelocity,
			delta_velocity,
			m_velocity,
			old_velocity,
			m_identifier);

		//timer.start();
		gridsToParticles(delta_velocity);
		//timer.stop();
		//printf("gridsToParticles:   %f \n", timer.getElapsedTime());


		old_velocity.clear();
		old_points.clear();
		delta_velocity.clear();

		if (this->outVelocity()->isEmpty())
			this->outVelocity()->allocate();
		auto& outvel = this->outVelocity()->getData();
		outvel.resize(m_velocity.size());
		cuExecute(m_velocity.size(),
			ESS_2DTo3D,
			outvel,
			m_velocity);
		//this->outVelocity()->assign(m_velocity);
		if (this->outPressure()->isEmpty())
			this->outPressure()->allocate();
		this->outPressure()->assign(m_density);
	}

	template <typename Coord2D>
	__global__ void ESS_Extrapolate2D(
		DArray<Coord2D> velocity,
		DArray<CellType> identifier,
		DArrayList<int> neighbors)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= identifier.size()) return;

		if (identifier[tId] != CellType::Inside)
		{
			Coord2D sum = Coord2D(0.0, 0.0);
			int count = 0;
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < neighbors[4 * tId + i].size(); j++)
				{
					int id = neighbors[4 * tId + i][j];
					if (identifier[id] == CellType::Inside)
					{
						sum += velocity[id];
						++count;
					}
				}
			}

			__syncthreads();

			if (count > 0)
			{
				identifier[tId] = CellType::Inside;
				velocity[tId] = sum / count;
				//velocity[tId][2] = 0.0f;
			}
		}
	}

	template<typename TDataType>
	void EulerianSimSplitting2D<TDataType>::extrapolate2D()
	{
		//printf("extrapolate start!   \n");

		DArray<CellType> identifier_temp;
		identifier_temp.assign(m_identifier);

		for (int layers = 0; layers < 10; ++layers)
		{
			cuExecute(m_identifier.size(),
				ESS_Extrapolate2D,
				m_velocity,
				identifier_temp,
				m_neighbor);
		}

		identifier_temp.clear();
		//printf("extrapolate is ok!   \n");
	}

	template <typename Coord2D, typename Real>
	__global__ void ESS_RKPosition2D(
		DArray<Coord2D> position,
		DArray<AdaptiveGridNode2D> node,
		DArray<Coord2D> velocity,
		DArray<CellType> identifier,
		Real dt)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= identifier.size()) return;

		if (identifier[tId] != CellType::Inside)
			position[tId] = node[tId].m_position;
		else
			position[tId] = node[tId].m_position - dt * velocity[tId];
	}

	//compute the interpolation of face
	template <typename Coord2D>
	GPU_FUNC void ESS_Interpolation2D(
		Coord2D& fv,
		int index,
		DArray<AdaptiveGridNode2D>& node,
		DArrayList<int>& neighbors,
		DArray<Coord2D>& velocity,
		int nid)//0,1,2,3, are -x,+x,-y,+y
	{
		if (neighbors[4 * index + nid].size() <= 0)
		{
			fv = velocity[index];
			return;
		}

		Level l0 = node[index].m_level;
		Level l1 = node[neighbors[4 * index + nid][0]].m_level;
		if (l0 > l1)
		{
			int nnid = nid % 2;
			if (nnid == 0) nnid = 1;
			else nnid = -1;

			int nindex = neighbors[4 * index + nid][0];
			int nnindex0 = neighbors[4 * nindex + nid + nnid][0];
			int nnindex1 = neighbors[4 * nindex + nid + nnid][1];

			fv = velocity[nnindex0] / 3 + velocity[nnindex1] / 3 + velocity[nindex] / 3;
		}
		else if (l0 < l1)
		{
			int nindex0 = neighbors[4 * index + nid][0];
			int nindex1 = neighbors[4 * index + nid][1];

			fv = velocity[nindex0] / 3 + velocity[nindex1] / 3 + velocity[index] / 3;
		}
		else
		{
			int nindex = neighbors[4 * index + nid][0];

			fv = velocity[nindex] / 2 + velocity[index] / 2;
		}
	}

	template <typename Coord2D, typename Real>
	__global__ void ESS_InterpV2DEdge(
		DArray<Coord2D> vel,
		DArray<int> pindex,
		DArray<Coord2D> pos,
		DArray<AdaptiveGridNode2D> node,
		DArrayList<int> neighbors,
		DArray<Coord2D> velocity,
		DArray<CellType> identifier,
		Level level,
		Real dx)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= identifier.size()) return;

		if (identifier[tId] != CellType::Inside)
		{
			vel[tId] = Coord2D(0.0f, 0.0f);
			return;
		}

		int index = pindex[tId];
		Real pdx = dx * (1 << (level - node[index].m_level));
		Coord2D resid = (pos[tId] - node[index].m_position) / pdx;
		Real alpha = resid[0];
		Real beta = resid[1];

		Coord2D n0, n1, n2, n3;
		ESS_Interpolation2D(n0, index, node, neighbors, velocity, 0);
		ESS_Interpolation2D(n1, index, node, neighbors, velocity, 1);
		ESS_Interpolation2D(n2, index, node, neighbors, velocity, 2);
		ESS_Interpolation2D(n3, index, node, neighbors, velocity, 3);
		vel[tId] = velocity[index] + alpha * (n1 - n0) + beta * (n3 - n2);
		//vel[tId][2] = 0.0f;
	}

	//The semi-Lagrangian method
	template<typename TDataType>
	void EulerianSimSplitting2D<TDataType>::advect2D(Real dt)
	{
		DArray<Coord2D> position(m_identifier.size());
		cuExecute(m_identifier.size(),
			ESS_RKPosition2D,
			position,
			m_node,
			m_velocity,
			m_identifier,
			0.5*dt);

		DArray<int> pos_index(m_identifier.size());
		auto nodeSet = this->inAdaptiveVolume2D()->getDataPtr();
		nodeSet->accessRandom(pos_index, position);

		DArray<Coord2D> vel_temp(m_identifier.size());
		cuExecute(m_identifier.size(),
			ESS_InterpV2DEdge,
			vel_temp,
			pos_index,
			position,
			m_node,
			m_neighbor,
			m_velocity,
			m_identifier,
			this->inAdaptiveVolume2D()->getDataPtr()->adaptiveGridLevelMax2D(),
			this->inAdaptiveVolume2D()->getDataPtr()->adaptiveGridDx2D());
		cuExecute(m_identifier.size(),
			ESS_RKPosition2D,
			position,
			m_node,
			vel_temp,
			m_identifier,
			dt);

		nodeSet->accessRandom(pos_index, position);
		cuExecute(m_identifier.size(),
			ESS_InterpV2DEdge,
			vel_temp,
			pos_index,
			position,
			m_node,
			m_neighbor,
			m_velocity,
			m_identifier,
			this->inAdaptiveVolume2D()->getDataPtr()->adaptiveGridLevelMax2D(),
			this->inAdaptiveVolume2D()->getDataPtr()->adaptiveGridDx2D());

		m_velocity.assign(vel_temp);
		position.clear();
		pos_index.clear();
		vel_temp.clear();

		//printf("advect is ok!  %d  \n", m_identifier.size());
	}

	template <typename Coord, typename Real>
	__global__ void ESS_BodyForce2D(
		DArray<Coord> velocity,
		DArray<CellType> identifier,
		Real gravity,
		Real dt)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= identifier.size()) return;

		if (identifier[tId] == CellType::Inside)
			velocity[tId][1] -= gravity * dt;
	}

	template<typename TDataType>
	void EulerianSimSplitting2D<TDataType>::body_force2D(Real dt)
	{
		cuExecute(m_identifier.size(),
			ESS_BodyForce2D,
			m_velocity,
			m_identifier,
			gravity,
			dt);
		//printf("body force is ok!  %d  \n", m_identifier.size());
	}

	__global__ void ESS_CountMatrix2D(
		DArray<uint> count,
		DArray<AdaptiveGridNode2D> node,
		DArrayList<int> neighbors,
		DArray<CellType> identifier)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= identifier.size()) return;

		if (identifier[tId] != CellType::Inside)
		{
			count[tId] = 0;
			return;
		}

		auto CountPN2D = [&](int nid) -> uint {
			int nindex = neighbors[4 * tId + nid][0];

			if (identifier[nindex] != CellType::Inside)
				return 0;
			else
			{
				if (node[tId].m_level == node[nindex].m_level)
					return 1;
				else if (node[tId].m_level > node[nindex].m_level)
					return 1;
				else
					return 2;
			}
		};

		count[tId] = 1 + CountPN2D(0) + CountPN2D(1) + CountPN2D(2) + CountPN2D(3);
	}

	//compute the gradient of pressure
	template <typename Real>
	GPU_FUNC void ESS_CMLevel2D(
		Map<int, Real>& map,
		DArrayList<int>& neighbors,
		Level& level,
		Level& nlevel,
		int index,
		int nindex,
		int nnindex,
		Real term)//For nindex and nnindex: 0,1,2,3 are -x,+x,-y,+y
	{
		if (level < nlevel)
		{
			int n0 = neighbors[4 * index + nindex][0];
			int n1 = neighbors[4 * index + nindex][1];

			//map.plusInsert(Pair<int, Real>(n0, -(0.66)*term));
			//map.plusInsert(Pair<int, Real>(n1, -(0.66)*term));
			//map.plusInsert(Pair<int, Real>(index, (1.32)*term));
			map.plusInsert(Pair<int, Real>(n0, -0.66*term));
			map.plusInsert(Pair<int, Real>(n1, -0.66*term));
			map.plusInsert(Pair<int, Real>(index, 1.32*term));
		}
		else if (level > nlevel)
		{
			int n0 = neighbors[4 * index + nindex][0];
			int nn0 = neighbors[4 * n0 + nnindex][0];
			int nn1 = neighbors[4 * n0 + nnindex][1];

			//map.plusInsert(Pair<int, Real>(nn0, (0.33)*term));
			//map.plusInsert(Pair<int, Real>(nn1, (0.33)*term));
			//map.plusInsert(Pair<int, Real>(n0, -(0.66) * term));
			map.plusInsert(Pair<int, Real>(n0, -0.66*term));
			map.plusInsert(Pair<int, Real>(index, 0.66*term));
		}
		else
		{
			int n0 = neighbors[4 * index + nindex][0];

			map.plusInsert(Pair<int, Real>(n0, -term));
			map.plusInsert(Pair<int, Real>(index, term));
		}
	}

	template <typename Coord2D, typename Real>
	GPU_FUNC void ESS_ConstructPM2D(
		Map<int, Real>& map,
		Real& mb,
		DArray<AdaptiveGridNode2D>& node,
		DArrayList<int>& neighbors,
		DArray<CellType>& identifier,
		DArray<Real>& sdf,
		DArray<Coord2D>& velocity,
		Real& term,
		Real& dx,//dx is the grid space of grid[index]
		int index,
		int nid,
		Coord2D& inletv1)//0,1 are x,y
	{
		int nindex_minus = neighbors[4 * index + 2 * nid][0];
		int nindex_plus = neighbors[4 * index + 2 * nid + 1][0];

		Real dx_m = dx / 100.0f;
		Real coefficient = 1.0f;
		if (identifier[nindex_minus] == CellType::Inside && identifier[nindex_plus] == CellType::Inside)
		{
			ESS_CMLevel2D(map, neighbors, node[index].m_level, node[nindex_minus].m_level, index, 2 * nid, 2 * nid + 1, coefficient);
			ESS_CMLevel2D(map, neighbors, node[index].m_level, node[nindex_plus].m_level, index, 2 * nid + 1, 2 * nid, coefficient);

			Coord2D vel0, vel1;
			ESS_Interpolation2D(vel0, index, node, neighbors, velocity, 2 * nid);
			ESS_Interpolation2D(vel1, index, node, neighbors, velocity, 2 * nid + 1);
			mb += -term * dx_m*(vel1[nid] - vel0[nid]);
		}
		else if (identifier[nindex_minus] == CellType::Inside && identifier[nindex_plus] != CellType::Inside)
		{
			if (node[nindex_plus].m_level != node[index].m_level)
			{
				printf("Warning!!!!! The node on the boundary is located in different level!!! %d %d %d %f %f, %f;  %d %d %d %f %f, %f \n",
					index, node[index].m_level, identifier[index], node[index].m_position[0], node[index].m_position[1], sdf[index],
					nindex_plus, node[nindex_plus].m_level, identifier[nindex_plus], node[nindex_plus].m_position[0], node[nindex_plus].m_position[1], sdf[nindex_plus]);
				return;
			}

			if (identifier[nindex_plus] == CellType::Inlet1)
			{
				//Real C = 2 * dx / (-2 * sdf[index] + dx);
				//ESS_CMLevel2D(map, neighbors, node[index].m_level, node[nindex_minus].m_level, index, 2 * nid, 2 * nid + 1, C);
				//Coord vel0;
				//ESS_Interpolation2D(vel0, index, node, neighbors, velocity, 2 * nid);
				//mb += -C * term * dx_m*(inletv1[nid] - vel0[nid]);

				ESS_CMLevel2D(map, neighbors, node[index].m_level, node[nindex_minus].m_level, index, 2 * nid, 2 * nid + 1, coefficient);
				Coord2D vel0;
				ESS_Interpolation2D(vel0, index, node, neighbors, velocity, 2 * nid);
				mb += -term * dx_m*(inletv1[nid] - vel0[nid]);

				//printf("Plus is inlet: %d \n", index);
			}
			else if (identifier[nindex_plus] == CellType::Static)
			{
				//Real C = 2 * dx / (-2 * sdf[index] + dx);
				//ESS_CMLevel2D(map, neighbors, node[index].m_level, node[nindex_minus].m_level, index, 2 * nid, 2 * nid + 1, C);
				//Coord vel0;
				//ESS_Interpolation2D(vel0, index, node, neighbors, velocity, 2 * nid);
				//mb += -C * term * dx_m*(-vel0[nid]);

				ESS_CMLevel2D(map, neighbors, node[index].m_level, node[nindex_minus].m_level, index, 2 * nid, 2 * nid + 1, coefficient);
				Coord2D vel0;
				ESS_Interpolation2D(vel0, index, node, neighbors, velocity, 2 * nid);
				mb += -term * dx_m*(-vel0[nid]);

				//printf("Plus is static: %d \n", index);
			}
			else if (identifier[nindex_plus] == CellType::Outlet1 || identifier[nindex_plus] == CellType::Outlet2)
			{
				//Real Dx = dx;
				//if ((node[index].m_level) > (node[nindex_minus].m_level)) Dx = dx * 3.0f / 2.0f;
				//Real B1 = (dx + Dx) / (-sdf[index] + Dx);
				//Real B2 = (-2.0f*sdf[index] + Dx - dx) / (-sdf[index] + Dx);
				//ESS_CMLevel2D(map, neighbors, node[index].m_level, node[nindex_minus].m_level, index, 2 * nid, 2 * nid + 1, B2);
				//map.plusInsert(Pair<int, Real>(index, B1));

				//ESS_CMLevel2D(map, neighbors, node[index].m_level, node[nindex_minus].m_level, index, 2 * nid, 2 * nid + 1, coefficient);
				//Real A = -dx / sdf[index];
				//map.plusInsert(Pair<int, Real>(index, A));

				ESS_CMLevel2D(map, neighbors, node[index].m_level, node[nindex_minus].m_level, index, 2 * nid, 2 * nid + 1, coefficient);
				map.plusInsert(Pair<int, Real>(index, 1.0f));

				//printf("Plus is outlet: %d \n", index);
			}
		}
		else if (identifier[nindex_minus] != CellType::Inside && identifier[nindex_plus] == CellType::Inside)
		{
			if (node[nindex_minus].m_level != node[index].m_level)
			{
				printf("Warning!!!!! The node on the boundary is located in different level!!! %d %d %d %f %f,  %f;  %d %d %d %f %f,  %f \n",
					index, node[index].m_level, identifier[index], node[index].m_position[0], node[index].m_position[1], sdf[index],
					nindex_minus, node[nindex_minus].m_level, identifier[nindex_minus], node[nindex_minus].m_position[0], node[nindex_minus].m_position[1], sdf[nindex_minus]);
				return;
			}

			if (identifier[nindex_minus] == CellType::Inlet1)
			{
				//Real C = 2 * dx / (-2 * sdf[index] + dx);
				//ESS_CMLevel2D(map, neighbors, node[index].m_level, node[nindex_plus].m_level, index, 2 * nid + 1, 2 * nid, C);
				//Coord vel1;
				//ESS_Interpolation2D(vel1, index, node, neighbors, velocity, 2 * nid + 1);
				//mb += -C * term * dx_m*(vel1[nid] - inletv1[nid]);

				ESS_CMLevel2D(map, neighbors, node[index].m_level, node[nindex_plus].m_level, index, 2 * nid + 1, 2 * nid, coefficient);
				Coord2D vel1;
				ESS_Interpolation2D(vel1, index, node, neighbors, velocity, 2 * nid + 1);
				mb += -term * dx_m*(vel1[nid] - inletv1[nid]);

				//printf("Minus is inlet: %d \n", index);
			}
			else if (identifier[nindex_minus] == CellType::Static)
			{
				//Real C = 2 * dx / (-2 * sdf[index] + dx);
				//ESS_CMLevel2D(map, neighbors, node[index].m_level, node[nindex_plus].m_level, index, 2 * nid + 1, 2 * nid, C);
				//Coord vel1;
				//ESS_Interpolation2D(vel1, index, node, neighbors, velocity, 2 * nid + 1);
				//mb += -C * term * dx_m*(vel1[nid]);

				ESS_CMLevel2D(map, neighbors, node[index].m_level, node[nindex_plus].m_level, index, 2 * nid + 1, 2 * nid, coefficient);
				Coord2D vel1;
				ESS_Interpolation2D(vel1, index, node, neighbors, velocity, 2 * nid + 1);
				mb += -term * dx_m*(vel1[nid]);

				//printf("Minus is static: %d \n", index);
			}
			else if (identifier[nindex_minus] == CellType::Outlet1 || identifier[nindex_minus] == CellType::Outlet2)
			{
				//Real Dx = dx;
				//if ((node[index].m_level) > (node[nindex_plus].m_level)) Dx = dx * 3.0f / 2.0f;
				//Real B1 = (dx + Dx) / (-sdf[index] + Dx);
				//Real B2 = (-2.0f*sdf[index] + Dx - dx) / (-sdf[index] + Dx);
				//ESS_CMLevel2D(map, neighbors, node[index].m_level, node[nindex_plus].m_level, index, 2 * nid + 1, 2 * nid, B2);
				//map.plusInsert(Pair<int, Real>(index, B1));

				//ESS_CMLevel2D(map, neighbors, node[index].m_level, node[nindex_plus].m_level, index, 2 * nid + 1, 2 * nid, coefficient);
				//Real A = -dx / sdf[index];
				//map.plusInsert(Pair<int, Real>(index, A));

				ESS_CMLevel2D(map, neighbors, node[index].m_level, node[nindex_plus].m_level, index, 2 * nid + 1, 2 * nid, coefficient);
				map.plusInsert(Pair<int, Real>(index, 1.0f));
			}
		}
	}

	template <typename Coord2D, typename Real>
	__global__ void ESS_ConstructPressureMatrix2D(
		DArrayMap<Real> matrix,
		DArray<Real> vector,
		DArray<AdaptiveGridNode2D> node,
		DArrayList<int> neighbors,
		DArray<CellType> identifier,
		DArray<Real> sdf,
		DArray<Coord2D> velocity,
		DArray<Real> density,
		Level level,
		Real dx,
		Real dt,
		Real wdensity,
		Real sdensity,
		Coord2D inletv1)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= identifier.size()) return;

		if (identifier[tId] != CellType::Inside)
			return;

		Real pdx = dx * (1 << (level - node[tId].m_level));
		Real term = (density[tId] * sdensity + (1 - density[tId])*wdensity) / dt;

		ESS_ConstructPM2D(matrix[tId], vector[tId], node, neighbors, identifier, sdf, velocity, term, pdx, tId, (int)0, inletv1);
		ESS_ConstructPM2D(matrix[tId], vector[tId], node, neighbors, identifier, sdf, velocity, term, pdx, tId, (int)1, inletv1);
	}

	template<typename TDataType>
	void EulerianSimSplitting2D<TDataType>::solve_pressure2D(DArray<Real>& pressure, DArray<Coord2D>& velocity, Real dt)
	{
		//printf("solve pressure start  \n");
		int sys_num = m_identifier.size();
		SparseMatrix<Real> solver;
		auto& pmatrix = solver.getMatrix();
		auto& pvector = solver.getVector();
		pvector.resize(sys_num);
		pvector.reset();

		DArray<uint> matrix_count(sys_num);
		matrix_count.reset();
		cuExecute(sys_num,
			ESS_CountMatrix2D,
			matrix_count,
			m_node,
			m_neighbor,
			m_identifier);

		pmatrix.resize(matrix_count);
		pmatrix.reset();
		matrix_count.clear();
		cuExecute(sys_num,
			ESS_ConstructPressureMatrix2D,
			pmatrix,
			pvector,
			m_node,
			m_neighbor,
			m_identifier,
			m_sdf,
			velocity,
			m_density,
			this->inAdaptiveVolume2D()->getDataPtr()->adaptiveGridLevelMax2D(),
			this->inAdaptiveVolume2D()->getDataPtr()->adaptiveGridDx2D(),
			dt,
			water_density,
			this->varSandDensity()->getData(),
			inlet_velocity);

		int solver_iterations = m_identifier.size()*0.3;
		//if (solver_iterations < 2000) solver_iterations = 2000;
		solver.CG(solver_iterations, 0.001);
		pressure.assign(solver.getX());

		solver.clear();

		DArray<Real> pv(m_identifier.size());
		pv.reset();
		Function2Pt::multiply(pv, pressure, pressure);
		Reduction<Real> reduce;
		convergence_indicator = reduce.accumulate(pv.begin(), pv.size());
		convergence_indicator = sqrt(convergence_indicator);
		pv.clear();
		//printf("Solve pressure end: Convergence_indicator %f  \n", convergence_indicator);
	}

	//compute the gradient of pressure
	template <typename Real>
	GPU_FUNC void ESS_UV2D(
		Real& pressureg,
		DArrayList<int>& neighbors,
		DArray<Real>& pressure,
		Level& level,
		Level& nlevel,
		int index,
		int nindex,
		int nnindex)//For nindex and nnindex: 0,1,2,3 are -x,+x,-y,+y
	{
		//Real sign = pow(Real(-1.0f), nindex);
		Real sign = (nindex % 2 == 1) ? (-1.0f) : 1.0f;
		if (level < nlevel)
		{
			int n0 = neighbors[4 * index + nindex][0];
			int n1 = neighbors[4 * index + nindex][1];
			pressureg = sign * (pressure[index] - ((pressure[n0] + pressure[n1]) / 2.0f))*(1.32);
		}
		else if (level > nlevel)
		{
			int n0 = neighbors[4 * index + nindex][0];
			int nn0 = neighbors[4 * n0 + nnindex][0];
			int nn1 = neighbors[4 * n0 + nnindex][1];
			pressureg = sign * (((pressure[nn0] + pressure[nn1]) / 2.0f) - pressure[n0])*(0.66);
		}
		else
		{
			int n0 = neighbors[4 * index + nindex][0];
			pressureg = sign * (pressure[index] - pressure[n0]);
		}
	}

	template <typename Coord2D, typename Real>
	GPU_FUNC void ESS_UpdateV2D(
		Coord2D& velocity,
		DArray<AdaptiveGridNode2D>& node,
		DArrayList<int>& neighbors,
		DArray<CellType>& identifier,
		DArray<Real>& sdf,
		DArray<Coord2D>& velocity_old,
		DArray<Real>& pressure,
		Real& term,
		Real& dx,//dx is the grid space of grid[index]
		int index,
		int nid,
		Coord2D& inletv1)//0,1 are x,y
	{
		int nindex_minus = neighbors[4 * index + 2 * nid][0];
		int nindex_plus = neighbors[4 * index + 2 * nid + 1][0];

		Real dx_m = dx / 100.0f;
		if (identifier[nindex_minus] == CellType::Inside && identifier[nindex_plus] == CellType::Inside)
		{
			Real preg_minus, preg_plus;
			ESS_UV2D(preg_minus, neighbors, pressure, node[index].m_level, node[nindex_minus].m_level, index, 2 * nid, 2 * nid + 1);
			ESS_UV2D(preg_plus, neighbors, pressure, node[index].m_level, node[nindex_plus].m_level, index, 2 * nid + 1, 2 * nid);

			velocity[nid] -= (term / dx_m)*(preg_minus + preg_plus) / 2;
		}
		else if (identifier[nindex_minus] == CellType::Inside && identifier[nindex_plus] != CellType::Inside)
		{
			if (node[nindex_plus].m_level != node[index].m_level)
			{
				printf("Warning!!!!! The node on the boundary is located in different level!!! \n");
				return;
			}

			if (identifier[nindex_plus] == CellType::Inlet1)
			{
				Real preg_minus;
				ESS_UV2D(preg_minus, neighbors, pressure, node[index].m_level, node[nindex_minus].m_level, index, (2 * nid), (2 * nid + 1));

				Coord2D vel0;
				ESS_Interpolation2D(vel0, index, node, neighbors, velocity_old, 2 * nid);
				vel0[nid] -= (term / dx_m)*preg_minus;

				Real C = dx / (-2 * sdf[index] + dx);
				velocity[nid] = C * inletv1[nid] + (1 - C)*vel0[nid];
			}
			else if (identifier[nindex_plus] == CellType::Static)
			{
				Real preg_minus;
				ESS_UV2D(preg_minus, neighbors, pressure, node[index].m_level, node[nindex_minus].m_level, index, (2 * nid), (2 * nid + 1));

				Coord2D vel0;
				ESS_Interpolation2D(vel0, index, node, neighbors, velocity_old, 2 * nid);
				vel0[nid] -= (term / dx_m)*preg_minus;

				Real C = dx / (-2 * sdf[index] + dx);
				velocity[nid] = (1 - C)*vel0[nid];
			}
			else if (identifier[nindex_plus] == CellType::Outlet1 || identifier[nindex_plus] == CellType::Outlet2)
			{
				//Real Dx = dx;
				//if ((node[index].m_level) > (node[nindex_minus].m_level)) Dx = dx * 3.0f / 2.0f;
				//Real B1 = (dx + Dx) / (-sdf[index] + Dx);
				//Real B3 = (dx - (-sdf[index])) / (-sdf[index] + Dx);
				//Real preg_minus;
				//ESS_UV2D(preg_minus, neighbors, pressure, node[index].m_level, node[nindex_minus].m_level, index, (2 * nid), (2 * nid + 1));
				//Real preg_plus = B1 * (-pressure[index]) + B3 * preg_minus;
				//velocity[nid] -= (term / dx_m)*(preg_minus + preg_plus) / 2;

				//Real preg_minus;
				//ESS_UV2D(preg_minus, neighbors, pressure, node[index].m_level, node[nindex_minus].m_level, index, (2 * nid), (2 * nid + 1));
				//Real A = -dx / sdf[index];
				//Real preg_plus = A * (-pressure[index]);
				//velocity[nid] -= (term / dx_m)*(preg_minus + preg_plus) / 2;

				Real preg_minus;
				ESS_UV2D(preg_minus, neighbors, pressure, node[index].m_level, node[nindex_minus].m_level, index, (2 * nid), (2 * nid + 1));
				Real preg_plus = (-pressure[index]);
				velocity[nid] -= (term / dx_m)*(preg_minus + preg_plus) / 2;
			}
		}
		else if (identifier[nindex_minus] != CellType::Inside && identifier[nindex_plus] == CellType::Inside)
		{
			if (node[nindex_minus].m_level != node[index].m_level)
			{
				printf("Warning!!!!! The node on the boundary is located in different level!!! \n");
				return;
			}

			if (identifier[nindex_minus] == CellType::Inlet1)
			{
				Real preg_plus;
				ESS_UV2D(preg_plus, neighbors, pressure, node[index].m_level, node[nindex_plus].m_level, index, 2 * nid + 1, 2 * nid);

				Coord2D vel1;
				ESS_Interpolation2D(vel1, index, node, neighbors, velocity_old, 2 * nid + 1);
				vel1[nid] -= (term / dx_m)*preg_plus;

				Real C = dx / (-2 * sdf[index] + dx);
				velocity[nid] = C * inletv1[nid] + (1 - C)*vel1[nid];
			}
			else if (identifier[nindex_minus] == CellType::Static)
			{
				Real preg_plus;
				ESS_UV2D(preg_plus, neighbors, pressure, node[index].m_level, node[nindex_plus].m_level, index, 2 * nid + 1, 2 * nid);

				Coord2D vel1;
				ESS_Interpolation2D(vel1, index, node, neighbors, velocity_old, 2 * nid + 1);
				vel1[nid] -= (term / dx_m)*preg_plus;

				Real C = dx / (-2 * sdf[index] + dx);
				velocity[nid] = (1 - C)*vel1[nid];
			}
			else if (identifier[nindex_minus] == CellType::Outlet1 || identifier[nindex_minus] == CellType::Outlet2)
			{
				//Real Dx = dx;
				//if ((node[index].m_level) > (node[nindex_plus].m_level)) Dx = dx * 3.0f / 2.0f;
				//Real B1 = (dx + Dx) / (-sdf[index] + Dx);
				//Real B3 = (dx - (-sdf[index])) / (-sdf[index] + Dx);
				//Real preg_plus;
				//ESS_UV2D(preg_plus, neighbors, pressure, node[index].m_level, node[nindex_plus].m_level, index, 2 * nid + 1, 2 * nid);
				//Real preg_minus = B1 * (pressure[index]) + B3 * preg_plus;
				//velocity[nid] -= (term / dx_m)*(preg_minus + preg_plus) / 2;

				//Real preg_plus;
				//ESS_UV2D(preg_plus, neighbors, pressure, node[index].m_level, node[nindex_plus].m_level, index, 2 * nid + 1, 2 * nid);
				//Real A = -dx / sdf[index];
				//Real preg_minus = A * (pressure[index]);
				//velocity[nid] -= (term / dx_m)*(preg_minus + preg_plus) / 2;

				Real preg_plus;
				ESS_UV2D(preg_plus, neighbors, pressure, node[index].m_level, node[nindex_plus].m_level, index, 2 * nid + 1, 2 * nid);
				Real preg_minus = (pressure[index]);
				velocity[nid] -= (term / dx_m)*(preg_minus + preg_plus) / 2;
			}
		}
	}

	template <typename Coord2D, typename Real>
	__global__ void ESS_UpdateVelocity2D(
		DArray<Coord2D> velocity,
		DArray<AdaptiveGridNode2D> node,
		DArrayList<int> neighbors,
		DArray<CellType> identifier,
		DArray<Real> sdf,
		DArray<Coord2D> velocity_old,
		DArray<Real> pressure,
		DArray<Real> density,
		Level level,
		Real dx,
		Real dt,
		Real wdensity,
		Real sdensity,
		Coord2D inletv1)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= identifier.size()) return;

		if (identifier[tId] != CellType::Inside)
			return;

		Real pdx = dx * (1 << (level - node[tId].m_level));
		Real term = dt / (density[tId] * sdensity + (1 - density[tId])*wdensity);

		ESS_UpdateV2D(velocity[tId], node, neighbors, identifier, sdf, velocity_old, pressure, term, pdx, tId, (int)0, inletv1);
		ESS_UpdateV2D(velocity[tId], node, neighbors, identifier, sdf, velocity_old, pressure, term, pdx, tId, (int)1, inletv1);
	}

	template<typename TDataType>
	void EulerianSimSplitting2D<TDataType>::update_velocity2D(DArray<Coord2D>& velocity, DArray<Real>& pressure, Real dt)
	{
		//printf("update velocity start  \n");
		DArray<Coord2D> velocity_old;
		velocity_old.assign(velocity);

		cuExecute(m_identifier.size(),
			ESS_UpdateVelocity2D,
			velocity,
			m_node,
			m_neighbor,
			m_identifier,
			m_sdf,
			velocity_old,
			pressure,
			m_density,
			this->inAdaptiveVolume2D()->getDataPtr()->adaptiveGridLevelMax2D(),
			this->inAdaptiveVolume2D()->getDataPtr()->adaptiveGridDx2D(),
			dt,
			water_density,
			this->varSandDensity()->getData(),
			inlet_velocity);

		velocity_old.clear();
		//printf("update velocity end  \n");
	}

	DEFINE_CLASS(EulerianSimSplitting2D);
}