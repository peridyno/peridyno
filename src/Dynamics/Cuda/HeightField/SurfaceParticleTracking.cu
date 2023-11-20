#include "SurfaceParticleTracking.h"

#include "GLPointVisualModule.h"

#include "Algorithm/CudaRand.h"

namespace dyno
{
	IMPLEMENT_TCLASS(SurfaceParticleTracking, TDataType)

	template<typename TDataType>
	SurfaceParticleTracking<TDataType>::SurfaceParticleTracking()
		: Node()
	{
		auto ps = std::make_shared<PointSet<TDataType>>();
		this->statePointSet()->setDataPtr(ps);

		auto render = std::make_shared<GLPointVisualModule>();
		render->varPointSize()->setValue(0.25);
		render->varBaseColor()->setValue(Color(1, 1, 0.1));
		this->statePointSet()->connect(render->inPointSet());
		this->graphicsPipeline()->pushModule(render);
	}

	template<typename TDataType>
	SurfaceParticleTracking<TDataType>::~SurfaceParticleTracking()
	{
		mPosition.clear();
		mMask.clear();

		mPositionBuffer.clear();
		mMaskBuffer.clear();

		mMutex.clear();
	}

	template<typename TDataType>
	bool SurfaceParticleTracking<TDataType>::validateInputs()
	{
		return this->importGranularMedia()->getDerivedNode() != nullptr;
	}

	template<typename TDataType>
	void SurfaceParticleTracking<TDataType>::resetStates()
	{
		auto granular = this->importGranularMedia()->getDerivedNode();

		Real ps = this->varSpacing()->getValue();

		auto w = granular->varWidth()->getValue();
		auto h = granular->varHeight()->getValue();

		mNx = floor((w - 0.5f * ps) / ps) + 1;
		mNy = floor((h - 0.5f * ps) / ps) + 1;
		mNz = this->varLayer()->getValue();

		mPosition.resize(mNx, mNy, mNz);
		mMask.resize(mNx, mNy, mNz);

		mPositionBuffer.resize(mNx, mNy, mNz);
		mMaskBuffer.resize(mNx, mNy, mNz);
		mMutex.resize(mNx, mNy, mNz);

		generate();

		updatePointSet();
	}

	//Section 6.1
	template<typename Real>
	__global__ void SPT_GenerateParticles(
		DArray3D<Vector<Real, 3>> position, 
		DArray3D<bool> bExist, 
		int ci, 
		int cj, 
		int ck)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = position.nx();
		int ny = position.ny();
		int nz = position.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		if (bExist(i, j, k))
		{
			return;
		}

		if (!(i % 2 == ci && j % 2 == cj && k % 2 == ck))
			return;

		RandNumber gen(position.index(i, j, k));
		Real x, y, z;
		x = gen.Generate();
		y = gen.Generate();
		z = gen.Generate();

		position(i, j, k) = Vec3f(x, y, z);
		bExist(i, j, k) = true;
	}

	template<typename TDataType>
	void SurfaceParticleTracking<TDataType>::generate()
	{
		if (mNz == 1)
		{
			for (int ci = 0; ci < 2; ci++)
			{
				for (int cj = 0; cj < 2; cj++)
				{
					cuExecute3D(make_uint3(mNx, mNy, mNz),
						SPT_GenerateParticles,
						mPosition, 
						mMask, 
						ci, 
						cj, 
						0);
				}
			}
		}
		else
		{
			for (int ci = 0; ci < 2; ci++)
			{
				for (int ck = 0; ck < 2; ck++)
				{
					for (int cj = 0; cj < 2; cj++)
					{
						cuExecute3D(make_uint3(mNx, mNy, mNz),
							SPT_GenerateParticles,
							mPosition, 
							mMask, 
							ci, 
							cj, 
							ck);
					}
				}
			}
		}
	}

	template<typename Real>
	__device__ Vector<Real, 2> d_transfer_velocity(Vector<Real, 4> vel)
	{
		Real u = vel.x < EPSILON ? 0.0f : vel.y / vel.x;
		Real v = vel.x < EPSILON ? 0.0f : vel.z / vel.x;
		return Vector<Real, 2>(u, v);
	}

	//Section 6.2
	template<typename Real, typename Coord3D, typename Coord4D>
	__global__ void SPT_AdvectParticles(
		DArray3D<Coord3D> p_pos,
		DArray2D<Coord4D> g_vel,
		Real p_spacing,
		Real g_spacing,
		Real dt)
	{
		uint i = blockDim.x * blockIdx.x + threadIdx.x;
		uint j = blockIdx.y * blockDim.y + threadIdx.y;
		uint k = blockIdx.z * blockDim.z + threadIdx.z;

		int p_nx = p_pos.nx();
		int p_ny = p_pos.ny();
		int p_nz = p_pos.nz();

		if (i >= p_nx) return;
		if (j >= p_ny) return;
		if (k >= p_nz) return;

		int g_nx = g_vel.nx();
		int g_ny = g_vel.ny();

		int k0 = p_pos.index(i, j, k);

		Real w00, w10, w01, w11;
		int g_ix, g_iy, g_iz;

		Coord3D p_ijk = p_pos[k0];

		Real g_fx = (i + p_ijk.x) * p_spacing / g_spacing;
		Real g_fy = (j + p_ijk.y) * p_spacing / g_spacing;

		if (g_fx < 0.0f) g_fx = 0.0f;
		if (g_fx > g_nx - 1) g_fx = g_nx - 1.0f;
		if (g_fy < 0.0f) g_fy = 0.0f;
		if (g_fy > g_ny - 1) g_fy = g_ny - 1.0f;

		g_ix = floor(g_fx);		g_iy = floor(g_fy);
		g_fx -= g_ix;			g_fy -= g_iy;

		if (g_ix == g_nx - 1) { g_ix = g_nx - 2; g_fx = 1.0f; }
		if (g_iy == g_ny - 1) { g_iy = g_ny - 2; g_fy = 1.0f; }

		w00 = (1.0f - g_fx) * (1.0f - g_fy);
		w10 = g_fx * (1.0f - g_fy);
		w01 = (1.0f - g_fx) * g_fy;
		w11 = g_fx * g_fy;

		g_ix++;
		g_iy++;

		Vector<Real, 2> vel_ijk = w00 * d_transfer_velocity(g_vel(g_ix, g_iy)) + w10 * d_transfer_velocity(g_vel(g_ix + 1, g_iy)) + w01 * d_transfer_velocity(g_vel(g_ix, g_iy + 1)) + w11 * d_transfer_velocity(g_vel(g_ix + 1, g_iy + 1));

		p_ijk.x += vel_ijk.x * dt / p_spacing;
		p_ijk.y += vel_ijk.y * dt / p_spacing;

		p_pos(i, j, k) = p_ijk;
	}

	template<typename TDataType>
	void SurfaceParticleTracking<TDataType>::advect()
	{
		auto granular = this->importGranularMedia()->getDerivedNode();

		Real ps = this->varSpacing()->getValue();
		Real gs = granular->varSpacing()->getValue();

		Real dt = this->stateTimeStep()->getValue();

		auto& computeGrid = granular->stateGrid()->getData();

		cuExecute3D(make_uint3(mNx, mNy, mNz),
			SPT_AdvectParticles,
			mPosition, 
			computeGrid, 
			ps,
			gs, 
			dt);
	}

	template<typename Coord3D>
	__global__ void SPT_DepositPigments(
		DArray3D<Coord3D> position, 
		DArray3D<Coord3D> positionBuffer, 
		DArray3D<bool> mask, 
		DArray3D<bool> maskBuffer, 
		DArray3D<uint> mutex)
	{
		uint i = blockIdx.x * blockDim.x + threadIdx.x;
		uint j = blockIdx.y * blockDim.y + threadIdx.y;
		uint k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = position.nx();
		int ny = position.ny();
		int nz = position.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		if (!maskBuffer(i, j, k)) return;

		int ix, iy, iz;
		Real fx, fy, fz;

		int id = positionBuffer.index(i, j, k);
		Coord3D pos = positionBuffer[id];

		ix = floor(i + pos.x);
		iy = floor(j + pos.y);
		iz = floor(k + pos.z);

		fx = i + pos.x - ix;
		fy = j + pos.y - iy;
		fz = k + pos.z - iz;

		Coord3D p = Coord3D(fx, fy, fz);

		if (ix < 0) { return; }
		if (ix >= nx) { return; }
		if (iy < 0) { return; }
		if (iy >= ny) { return; }
		if (iz < 0) { return; }
		if (iz >= nz) { return; }

		int id_new = position.index(ix, iy, iz);

		while (atomicCAS(&(mutex[id_new]), 0, 1) == 0) break;
		position[id_new] = p;
		mask[id_new] = true;
		atomicExch(&(mutex[id_new]), 0);
	}

	template<typename TDataType>
	void SurfaceParticleTracking<TDataType>::deposit()
	{
		mPositionBuffer.assign(mPosition);
		mMaskBuffer.assign(mMask);

		mPosition.reset();
		mMask.reset();

		cuExecute3D(make_uint3(mNx, mNy, mNz),
			SPT_DepositPigments,
			mPosition,
			mPositionBuffer,
			mMask,
			mMaskBuffer,
			mMutex);
	}

	template<typename TDataType>
	void SurfaceParticleTracking<TDataType>::updateStates()
	{
		advect();
		deposit();
		generate();

		updatePointSet();
	}

	//Section 6.3
	template<typename Real, typename Coord3D, typename Coord4D>
	__global__ void SPT_UpdatePointSet(
		DArray<Coord3D> points,
		DArray2D<Coord4D> grid,
		DArray3D<Coord3D> point3d,
		Coord3D origin,
		Real s)
	{
		uint i = blockIdx.x * blockDim.x + threadIdx.x;
		uint j = blockIdx.y * blockDim.y + threadIdx.y;

		Real grid_spacing = 1.0;

		uint width = point3d.nx();
		uint height = point3d.ny();

		if (i >= point3d.nx() || j >= point3d.ny()) return;

		for (uint k = 0; k < point3d.nz(); k++)
		{
			Coord3D pos = point3d(i, j, k);
			Real grid_fx = (i + pos.x) * s / grid_spacing;
			Real grid_fy = (j + pos.y) * s / grid_spacing;

			if (grid_fx < 0.0f) grid_fx = 0.0f;
			if (grid_fx > width - 1) grid_fx = width - 1.0f;
			if (grid_fy < 0.0f) grid_fy = 0.0f;
			if (grid_fy > height - 1) grid_fy = height - 1.0f;

			int gridx = floor(grid_fx);		int gridy = floor(grid_fy);
			Real fx = grid_fx - gridx;		Real fy = grid_fy - gridy;

			if (gridx == width - 1) { gridx = width - 2; fx = 1.0f; }
			if (gridy == height - 1) { gridy = height - 2; fy = 1.0f; }

			Real w00 = (1.0f - fx) * (1.0f - fy);
			Real w10 = fx * (1.0f - fy);
			Real w01 = (1.0f - fx) * fy;
			Real w11 = fx * fy;

			Coord4D gp00 = grid(gridx + 1, gridy + 1);
			Coord4D gp10 = grid(gridx + 2, gridy + 1);
			Coord4D gp01 = grid(gridx + 1, gridy + 2);
			Coord4D gp11 = grid(gridx + 2, gridy + 2);

			Coord4D gp = w00 * gp00 + w10 * gp10 + w01 * gp01 + w11 * gp11;

			uint id = point3d.index(i, j, k);
			points[id] = origin + Coord3D((pos.x + i) * s, gp.x + gp.w - k * s, (pos.z + j) * s);
		}
	}

	template<typename TDataType>
	void SurfaceParticleTracking<TDataType>::updatePointSet()
	{
		auto pointSet = this->statePointSet()->getDataPtr();

		float ps = this->varSpacing()->getValue();

		auto granular = this->importGranularMedia()->getDerivedNode();

		auto& computeGrid = granular->stateGrid()->getData();

		auto& points = pointSet->getPoints();

		auto origin = granular->varOrigin()->getValue();

		uint num = mPosition.size();

		points.resize(num);

		cuExecute2D(make_uint2(mPosition.nx(), mPosition.ny()),
			SPT_UpdatePointSet,
			points,
			computeGrid,
			mPosition,
			origin,
			ps);
	}

	DEFINE_CLASS(SurfaceParticleTracking);
}