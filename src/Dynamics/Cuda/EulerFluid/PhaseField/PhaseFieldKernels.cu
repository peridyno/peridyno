#include "PhaseFieldKernels.h"

#include "Algorithm/Reduction.h"


namespace dyno{

	template<typename T>
	__global__ void K_CopyData(T dst, T src)
	{
		uint i = blockDim.x * blockIdx.x + threadIdx.x;
		uint j = blockIdx.y * blockDim.y + threadIdx.y;
		uint k = blockIdx.z * blockDim.z + threadIdx.z;

		if (i >= src.nx()) return;
		if (j >= src.ny()) return;
		if (k >= src.nz()) return;

		dst(i, j, k) = src(i, j, k);
	}

	__global__ void K_SetVelocityBoundary(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w)
	{
		uint i = blockDim.x * blockIdx.x + threadIdx.x;
		uint j = blockIdx.y * blockDim.y + threadIdx.y;
		uint k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = vel_v.nx();
		int ny = vel_w.ny();
		int nz = vel_u.nz();

		if (i >= vel_v.nx()) return;
		if (j >= vel_w.ny()) return;
		if (k >= vel_u.nz()) return;

		if (k == nz - 1) { vel_u(i, j, k) = vel_u(i, j, k - 1); vel_w(i, j, k) = vel_w(i, j, k - 1); return; }
	}

	template<typename Grid3f, typename Grid1f>
	__global__ void K_InterpolateVelocity(Grid3f vel, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w)
	{
		uint i = blockDim.x * blockIdx.x + threadIdx.x;
		uint j = blockIdx.y * blockDim.y + threadIdx.y;
		uint k = blockIdx.z * blockDim.z + threadIdx.z;

		if (i >= vel.nx()) return;
		if (j >= vel.ny()) return;
		if (k >= vel.nz()) return;

		float a = 1.0f;

		Vec3f vel_ijk;
		vel_ijk.x = 0.5f*(vel_u(i, j, k) + vel_u(i + 1, j, k));
		vel_ijk.y = 0.5f*(vel_v(i, j, k) + vel_v(i, j + 1, k));
		vel_ijk.z = 0.5f*(vel_w(i, j, k) + vel_w(i, j, k + 1));

// 			if (i == 10)
// 			{
// 				printf("%f ", 0.5f*(vel_w(i, j, k) + vel_w(i, j, k + 1)));
// 			}

		vel(i, j, k) = vel_ijk;
	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::InterpolateVelocity(Grid3f vel, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w)
	{
		dim3 gridDims, blockDims;
		uint3 fDims = make_uint3(vel.nx(), vel.ny(), vel.nz());
		computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

//			K_SetVelocityBoundary << < gridDims, blockDims >> >(vel_u, vel_v, vel_w);
		K_InterpolateVelocity << < gridDims, blockDims >> >(vel, vel_u, vel_v, vel_w);
	}

	template<typename Grid1f, typename Grid3f>
	__global__ void K_InterpolateVelocity(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid3f vel)
	{
		uint i = blockDim.x * blockIdx.x + threadIdx.x;
		uint j = blockIdx.y * blockDim.y + threadIdx.y;
		uint k = blockIdx.z * blockDim.z + threadIdx.z;

		if (i >= vel.nx()) return;
		if (j >= vel.ny()) return;
		if (k >= vel.nz()) return;

		auto v = vel(i, j, k);

		if (i < vel.nx() - 1) { auto v_i_plus = vel(i + 1, j, k); auto vx = 0.5f * (v.x + v_i_plus.x); vel_u(i + 1, j, k) = vx; }
		if (j < vel.ny() - 1) { auto v_j_plus = vel(i, j + 1, k); auto vy = 0.5f * (v.y + v_j_plus.y); vel_v(i, j + 1, k) = vy; }
		if (k < vel.nz() - 1) { auto v_k_plus = vel(i, j, k + 1); auto vz = 0.5f * (v.z + v_k_plus.z); vel_w(i, j, k + 1) = vz; }
	}

	template<typename Grid3f>
	__global__ void K_DampVelocity(Grid3f vel, float radius)
	{
		uint i = blockDim.x * blockIdx.x + threadIdx.x;
		uint j = blockIdx.y * blockDim.y + threadIdx.y;
		uint k = blockIdx.z * blockDim.z + threadIdx.z;

		if (i >= vel.nx() - 1) return;
		if (j >= vel.ny() - 1) return;
		if (k >= vel.nz() - 1) return;


		float w = 1.0f;

		float d = sqrt(((float)i - vel.nx() / 2.0f)*((float)i - vel.nx() / 2.0f) + ((float)j - vel.ny() / 2.0f)*((float)j - vel.ny() / 2.0f));
		if (d > 1.5f*radius)
		{
			w = 0.0f;
		}
		else if (d > 0.5f*radius)
		{
			w = 1.0f - (d - 0.5f*radius) / radius;
		}

		vel(i, j, k) = w*vel(i, j, k);
	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::InterpolateVelocity(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid3f vel)
	{
		uint3 gDIms = make_uint3(vel.nx(), vel.ny(), vel.nz());
		uint3 bDims = make_uint3(32, 8, 1);

		vel_u.reset();
		vel_v.reset();
		vel_w.reset();

		//LAUNCH_KERNEL(K_DampVelocity, gDIms, bDims, vel, radius);
		LAUNCH_KERNEL(K_InterpolateVelocity, gDIms, bDims, vel_u, vel_v, vel_w, vel);
	}

	template<typename Grid1f, typename Grid3f>
	__global__ void K_AdvectBackward(Grid1f dst, Grid1f src, Grid3f vel, float dt)
	{
		uint i = blockDim.x * blockIdx.x + threadIdx.x;
		uint j = blockIdx.y * blockDim.y + threadIdx.y;
		uint k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = dst.nx();
		int ny = dst.ny();
		int nz = dst.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		auto vel_ijk = vel(i, j, k);
			
		int ix, iy, iz;
		float fx, fy, fz;
		float w000, w100, w010, w001, w111, w011, w101, w110;

		fx = i - vel_ijk.x * dt;
		fy = j - vel_ijk.y * dt;
		fz = k - vel_ijk.z * dt;

		if (fx < 0.0) fx = 0.0;
		if (fx > nx - 1) fx = nx - 1.0;
		if (fy < 0.0) fy = 0.0f;
		if (fy > ny - 1) fy = ny - 1.0;
		if (fz < 0.0) fz = 0.0f;
		if (fz > nz - 1) fz = nz - 1.0;

		ix = uint(fx);      iy = uint(fy);		iz = uint(fz);
		fx -= ix;			fy -= iy;			fz -= iz;

		if (ix == nx - 1) { ix = nx - 2; fx = 1.0; }
		if (iy == ny - 1) { iy = ny - 2; fy = 1.0; }
		if (iz == nz - 1) { iz = nz - 2; fz = 1.0; }

		w000 = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
		w100 = fx * (1.0f - fy) * (1.0f - fz);
		w010 = (1.0f - fx) * fy * (1.0f - fz);
		w001 = (1.0f - fx) * (1.0f - fy) * fz;
		w111 = fx * fy * fz;
		w011 = (1.0f - fx) * fy * fz;
		w101 = fx * (1.0f - fy) * fz;
		w110 = fx * fy * (1.0f - fz);

		Vec3f vel_ijk_p = w000 * vel(ix, iy, iz) + w100 * vel(ix + 1, iy, iz) +
			w010 * vel(ix, iy + 1, iz) + w001 * vel(ix, iy, iz + 1) +
			w111 * vel(ix + 1, iy + 1, iz + 1) + w011 * vel(ix, iy + 1, iz + 1) +
			w101 * vel(ix + 1, iy, iz + 1) + w110 * vel(ix + 1, iy + 1, iz);

		vel_ijk = (vel_ijk_p + vel_ijk) * 0.5f;

		fx = i - vel_ijk.x*dt;
		fy = j - vel_ijk.y*dt;
		fz = k - vel_ijk.z*dt;

		if (fx < 0.0f) fx = 0.0f;
		if (fx > nx - 1) fx = nx - 1.0f;
		if (fy < 0.0f) fy = 0.0f;
		if (fy > ny - 1) fy = ny - 1.0f;
		if (fz < 0.0f) fz = 0.0f;
		if (fz > nz - 1) fz = nz - 1.0f;

		ix = (int)fx;		iy = (int)fy;		iz = (int)fz;
		fx -= ix;			fy -= iy;			fz -= iz;

		if (ix == nx - 1) { ix = nx - 2; fx = 1.0f; }
		if (iy == ny - 1) { iy = ny - 2; fy = 1.0f; }
		if (iz == nz - 1) { iz = nz - 2; fz = 1.0f; }

		w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
		w100 = fx*(1.0f - fy)*(1.0f - fz);
		w010 = (1.0f - fx)*fy*(1.0f - fz);
		w001 = (1.0f - fx)*(1.0f - fy)*fz;
		w111 = fx*fy*fz;
		w011 = (1.0f - fx)*fy*fz;
		w101 = fx*(1.0f - fy)*fz;
		w110 = fx*fy*(1.0f - fz);

		dst(i, j, k) = w000*src(ix, iy, iz) + w100 * src(ix + 1, iy, iz) + w010 * src(ix, iy + 1, iz) + w001 * src(ix, iy, iz + 1)
			+ w111*src(ix + 1, iy + 1, iz + 1) + w011*src(ix, iy + 1, iz + 1) + w101*src(ix + 1, iy, iz + 1) + w110*src(ix + 1, iy + 1, iz);
	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::AdvectBackward(Grid1f dst, Grid1f src, Grid3f vel, float dt)
	{
		dim3 gridDims, blockDims;
		uint3 fDims = make_uint3(src.nx(), src.ny(), src.nz());
		computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

		K_AdvectBackward <<< gridDims, blockDims >>>(dst, src, vel, dt);
	}

	template<typename Grid1f>
	__global__ void K_AdvectStaggeredBackward(Grid1f dst, Grid1f src, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, float dt, int axis)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = dst.nx();
		int ny = dst.ny();
		int nz = dst.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		float vx, vy, vz;
		float fx, fy, fz;
		int ix, iy, iz;
		float w000, w100, w010, w001, w111, w011, w101, w110;

		if (axis == 1)
		{
			int i_minus = glm::clamp(i - 1, 0, nx - 2);
			int i_plus = glm::clamp(i, 0, nx - 2);

			vx = vel_u(i, j, k);
			vy = 0.25f*(vel_v(i_minus, j, k) + vel_v(i_minus, j + 1, k) + vel_v(i_plus, j, k) + vel_v(i_plus, j + 1, k));
			vz = 0.25f*(vel_w(i_minus, j, k) + vel_w(i_minus, j, k + 1) + vel_w(i_plus, j, k) + vel_w(i_plus, j, k + 1));
		}
		if (axis == 2)
		{
			int j_minus = glm::clamp(j - 1, 0, ny - 2);
			int j_plus = glm::clamp(j, 0, ny - 2);

			vx = 0.25f*(vel_u(i, j_minus, k) + vel_u(i + 1, j_minus, k) + vel_u(i, j_plus, k) + vel_u(i + 1, j_plus, k));
			vy = vel_v(i, j, k);
			vz = 0.25f*(vel_w(i, j_minus, k) + vel_w(i, j_minus, k + 1) + vel_w(i, j_plus, k) + vel_w(i, j_plus, k + 1));
		}
		if (axis == 3)
		{
			int k_minus = glm::clamp(k - 1, 0, nz - 2);
			int k_plus = glm::clamp(k, 0, nz - 2);

			vx = 0.25f*(vel_u(i, j, k_minus) + vel_u(i + 1, j, k_minus) + vel_u(i, j, k_plus) + vel_u(i + 1, j, k_plus));
			vy = 0.25f*(vel_v(i, j, k_minus) + vel_v(i, j + 1, k_minus) + vel_v(i, j, k_plus) + vel_v(i, j + 1, k_plus));
			vz = vel_w(i, j, k);
		}

		fx = i - vx*dt;
		fy = j - vy*dt;
		fz = k - vz*dt;

		if (fx < 0.0f) fx = 0.0f;
		if (fx > nx - 1) fx = nx - 1.0f;
		if (fy < 0.0f) fy = 0.0f;
		if (fy > ny - 1) fy = ny - 1.0f;
		if (fz < 0.0f) fz = 0.0f;
		if (fz > nz - 1) fz = nz - 1.0f;

		ix = (int)fx;		iy = (int)fy;		iz = (int)fz;
		fx -= ix;			fy -= iy;			fz -= iz;

		if (ix == nx - 1) { ix = nx - 2; fx = 1.0f; }
		if (iy == ny - 1) { iy = ny - 2; fy = 1.0f; }
		if (iz == nz - 1) { iz = nz - 2; fz = 1.0f; }

		w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
		w100 = fx*(1.0f - fy)*(1.0f - fz);
		w010 = (1.0f - fx)*fy*(1.0f - fz);
		w001 = (1.0f - fx)*(1.0f - fy)*fz;
		w111 = fx*fy*fz;
		w011 = (1.0f - fx)*fy*fz;
		w101 = fx*(1.0f - fy)*fz;
		w110 = fx*fy*(1.0f - fz);

		int nxy = nx*ny;
		int k0 = ix + iy*nx + iz*nxy;

		dst(i, j, k) = w000*src[k0] + w100*src[k0 + 1] + w010*src[k0 + nx] + w001*src[k0 + nxy]
			+ w111*src[k0 + 1 + nx + nxy] + w011*src[k0 + nx + nxy] + w101*src[k0 + 1 + nxy] + w110*src[k0 + 1 + nx];
	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::AdvectStaggeredBackward(Grid1f dst, Grid1f src, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, float dt, int axis)
	{
		dim3 gridDims, blockDims;
		uint3 fDims = make_uint3(src.nx(), src.ny(), src.nz());
		computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

		K_AdvectStaggeredBackward << < gridDims, blockDims >> >(dst, src, vel_u, vel_v, vel_w, dt, axis);
	}


	template<typename Grid1f, typename Grid3f>
	__global__ void K_AdvectForward(Grid1f dst, Grid1f src, Grid3f vel, float dt)
	{
		uint i = blockDim.x * blockIdx.x + threadIdx.x;
		uint j = blockIdx.y * blockDim.y + threadIdx.y;
		uint k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = dst.nx();
		int ny = dst.ny();
		int nz = dst.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		float fx, fy, fz;
		int ix, iy, iz;
		float w000, w100, w010, w001, w111, w011, w101, w110;

		auto vel_ijk = vel(i, j, k);

		fx = i + vel_ijk.x * dt;
		fy = j + vel_ijk.y * dt;
		fz = k + vel_ijk.z * dt;

		if (fx < 0.0) fx = 0.0;
		if (fx > nx - 1) fx = nx - 1.0;
		if (fy < 0.0) fy = 0.0f;
		if (fy > ny - 1) fy = ny - 1.0;
		if (fz < 0.0) fz = 0.0f;
		if (fz > nz - 1) fz = nz - 1.0;

		ix = int(fx);      iy = int(fy);		iz = int(fz);
		fx -= ix;			fy -= iy;			fz -= iz;

		if (ix == nx - 1) { ix = nx - 2; fx = 1.0; }
		if (iy == ny - 1) { iy = ny - 2; fy = 1.0; }
		if (iz == nz - 1) { iz = nz - 2; fz = 1.0; }

		w000 = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
		w100 = fx * (1.0f - fy) * (1.0f - fz);
		w010 = (1.0f - fx) * fy * (1.0f - fz);
		w001 = (1.0f - fx) * (1.0f - fy) * fz;
		w111 = fx * fy * fz;
		w011 = (1.0f - fx) * fy * fz;
		w101 = fx * (1.0f - fy) * fz;
		w110 = fx * fy * (1.0f - fz);

		Vec3f vel_ijk_p = w000 * vel(ix, iy, iz) + w100 * vel(ix + 1, iy, iz) +
			w010 * vel(ix, iy + 1, iz) + w001 * vel(ix, iy, iz + 1) +
			w111 * vel(ix + 1, iy + 1, iz + 1) + w011 * vel(ix, iy + 1, iz + 1) +
			w101 * vel(ix + 1, iy, iz + 1) + w110 * vel(ix + 1, iy + 1, iz);

		vel_ijk = (vel_ijk_p + vel_ijk) * 0.5f;

		fx = i + vel_ijk.x*dt;
		fy = j + vel_ijk.y*dt;
		fz = k + vel_ijk.z*dt;

		if (fx < 0.0f) fx = 0.0f;
		if (fx > nx - 1) fx = nx - 1.0f;
		if (fy < 0.0f) fy = 0.0f;
		if (fy > ny - 1) fy = ny - 1.0f;
		if (fz < 0.0f) fz = 0.0f;
		if (fz > nz - 1) fz = nz - 1.0f;

		ix = (int)fx;		iy = (int)fy;		iz = (int)fz;
		fx -= ix;			fy -= iy;			fz -= iz;

		if (ix == nx - 1) { ix = nx - 2; fx = 1.0f; }
		if (iy == ny - 1) { iy = ny - 2; fy = 1.0f; }
		if (iz == nz - 1) { iz = nz - 2; fz = 1.0f; }

		float val = src(i, j, k);
		w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
		w100 = fx*(1.0f - fy)*(1.0f - fz);
		w010 = (1.0f - fx)*fy*(1.0f - fz);
		w001 = (1.0f - fx)*(1.0f - fy)*fz;
		w111 = fx*fy*fz;
		w011 = (1.0f - fx)*fy*fz;
		w101 = fx*(1.0f - fy)*fz;
		w110 = fx*fy*(1.0f - fz);

		atomicAdd(&dst(ix, iy, iz), val*w000);
		atomicAdd(&dst(ix + 1, iy, iz), val*w100);
		atomicAdd(&dst(ix, iy + 1, iz), val*w010);
		atomicAdd(&dst(ix, iy, iz + 1), val*w001);
		atomicAdd(&dst(ix + 1, iy + 1, iz + 1), val*w111);
		atomicAdd(&dst(ix, iy + 1, iz + 1), val*w011);
		atomicAdd(&dst(ix + 1, iy, iz + 1), val*w101);
		atomicAdd(&dst(ix + 1, iy + 1, iz), val*w110);

// 			d(ix, iy, iz) += val*w000;
// 			d(ix + 1, iy, iz) += val*w100;
// 			d(ix, iy + 1, iz) += val*w010;
// 			d(ix, iy, iz + 1) += val*w001;
// 			d(ix + 1, iy + 1, iz + 1) += val*w111;
// 			d(ix, iy + 1, iz + 1) += val*w011;
// 			d(ix + 1, iy, iz + 1) += val*w101;
// 			d(ix + 1, iy + 1, iz) += val*w110;
	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::AdvectForward(Grid1f dst, Grid1f src, Grid3f vel, float dt)
	{
		dst.reset();

		dim3 gridDims, blockDims;
		uint3 fDims = make_uint3(src.nx(), src.ny(), src.nz());
		computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

//			LAUNCH_KERNEL(K_ClampVelocity, fDims, make_uint3(32, 8, 1), vel);

		K_AdvectForward << < gridDims, blockDims >> >(dst, src, vel, dt);
	}

	template<typename Grid4f, typename Grid3f>
	__global__ void K_AdvectForward(Grid4f dst, Grid4f src, Grid3f vel, Grid1f weight, float dt)
	{
		uint i = blockDim.x * blockIdx.x + threadIdx.x;
		uint j = blockIdx.y * blockDim.y + threadIdx.y;
		uint k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = dst.nx();
		int ny = dst.ny();
		int nz = dst.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		float fx, fy, fz;
		int ix, iy, iz;
		float w000, w100, w010, w001, w111, w011, w101, w110;

		auto vel_ijk = vel(i, j, k);

		fx = i + vel_ijk.x*dt;
		fy = j + vel_ijk.y*dt;
		fz = k + vel_ijk.z*dt;

		if (fx < 0.0f) fx = 0.0f;
		if (fx > nx - 1) fx = nx - 1.0f;
		if (fy < 0.0f) fy = 0.0f;
		if (fy > ny - 1) fy = ny - 1.0f;
		if (fz < 0.0f) fz = 0.0f;
		if (fz > nz - 1) fz = nz - 1.0f;

		ix = (int)fx;		iy = (int)fy;		iz = (int)fz;
		fx -= ix;			fy -= iy;			fz -= iz;

		if (ix == nx - 1) { ix = nx - 2; fx = 1.0f; }
		if (iy == ny - 1) { iy = ny - 2; fy = 1.0f; }
		if (iz == nz - 1) { iz = nz - 2; fz = 1.0f; }

		auto val = src(i, j, k);
		w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
		w100 = fx*(1.0f - fy)*(1.0f - fz);
		w010 = (1.0f - fx)*fy*(1.0f - fz);
		w001 = (1.0f - fx)*(1.0f - fy)*fz;
		w111 = fx*fy*fz;
		w011 = (1.0f - fx)*fy*fz;
		w101 = fx*(1.0f - fy)*fz;
		w110 = fx*fy*(1.0f - fz);

		int nxy = nx*ny;
		int k0 = dst.index(ix, iy, iz);

		atomicAdd(&weight[k0], w000);
		atomicAdd(&weight[k0 + 1], w100);
		atomicAdd(&weight[k0 + nx], w010);
		atomicAdd(&weight[k0 + nxy], w001);
		atomicAdd(&weight[k0 + 1 + nx + nxy], w111);
		atomicAdd(&weight[k0 + nx + nxy], w011);
		atomicAdd(&weight[k0 + 1 + nxy], w101);
		atomicAdd(&weight[k0 + 1 + nx], w110);

		atomicAdd(&(dst[k0].x), w000*val.x);
		atomicAdd(&(dst[k0 + 1].x), w100*val.x);
		atomicAdd(&(dst[k0 + nx].x), w010*val.x);
		atomicAdd(&(dst[k0 + nxy].x), w001*val.x);
		atomicAdd(&(dst[k0 + 1 + nx + nxy].x), w111*val.x);
		atomicAdd(&(dst[k0 + nx + nxy].x), w011*val.x);
		atomicAdd(&(dst[k0 + 1 + nxy].x), w101*val.x);
		atomicAdd(&(dst[k0 + 1 + nx].x), w110*val.x);

		atomicAdd(&(dst[k0].y), w000*val.y);
		atomicAdd(&(dst[k0 + 1].y), w100*val.y);
		atomicAdd(&(dst[k0 + nx].y), w010*val.y);
		atomicAdd(&(dst[k0 + nxy].y), w001*val.y);
		atomicAdd(&(dst[k0 + 1 + nx + nxy].y), w111*val.y);
		atomicAdd(&(dst[k0 + nx + nxy].y), w011*val.y);
		atomicAdd(&(dst[k0 + 1 + nxy].y), w101*val.y);
		atomicAdd(&(dst[k0 + 1 + nx].y), w110*val.y);

		atomicAdd(&(dst[k0].z), w000*val.z);
		atomicAdd(&(dst[k0 + 1].z), w100*val.z);
		atomicAdd(&(dst[k0 + nx].z), w010*val.z);
		atomicAdd(&(dst[k0 + nxy].z), w001*val.z);
		atomicAdd(&(dst[k0 + 1 + nx + nxy].z), w111*val.z);
		atomicAdd(&(dst[k0 + nx + nxy].z), w011*val.z);
		atomicAdd(&(dst[k0 + 1 + nxy].z), w101*val.z);
		atomicAdd(&(dst[k0 + 1 + nx].z), w110*val.z);

		atomicAdd(&(dst[k0].w), w000*val.w);
		atomicAdd(&(dst[k0 + 1].w), w100*val.w);
		atomicAdd(&(dst[k0 + nx].w), w010*val.w);
		atomicAdd(&(dst[k0 + nxy].w), w001*val.w);
		atomicAdd(&(dst[k0 + 1 + nx + nxy].w), w111*val.w);
		atomicAdd(&(dst[k0 + nx + nxy].w), w011*val.w);
		atomicAdd(&(dst[k0 + 1 + nxy].w), w101*val.w);
		atomicAdd(&(dst[k0 + 1 + nx].w), w110*val.w);
	}

	template<typename Grid4f, typename Grid1f>
	__global__ void K_NormalizePigment(Grid4f dst, Grid1f weight)
	{
		uint i = blockDim.x * blockIdx.x + threadIdx.x;
		uint j = blockIdx.y * blockDim.y + threadIdx.y;
		uint k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = dst.nx();
		int ny = dst.ny();
		int nz = dst.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		int k0 = weight.index(i, j, k);
		float w = weight[k0];
		if (w > EPSILON)
		{
			dst[k0] /= w;
		}
		else
			dst[k0] = Vec4f(0.0f);
	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::AdvectForward(Grid4f dst, Grid4f src, Grid3f vel, Grid1f weight, float dt)
	{
		dst.reset();
		weight.reset();

		uint3 gDIms = make_uint3(vel.nx(), vel.ny(), vel.nz());
		uint3 bDims = make_uint3(32, 8, 1);

		LAUNCH_KERNEL(K_AdvectForward, gDIms, bDims, dst, src, vel, weight, dt);
		LAUNCH_KERNEL(K_NormalizePigment, gDIms, bDims, dst, weight);
	}

	template<typename Grid3f>
	__global__ void K_AdvectBackward(Grid3f dst, Grid3f src, Grid3f vel, float dt)
	{
		uint i = blockDim.x * blockIdx.x + threadIdx.x;
		uint j = blockIdx.y * blockDim.y + threadIdx.y;
		uint k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = dst.nx();
		int ny = dst.ny();
		int nz = dst.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		auto vel_ijk = vel(i, j, k);

		int ix, iy, iz;
		float fx, fy, fz;
		float w000, w100, w010, w001, w111, w011, w101, w110;

		fx = i - vel_ijk.x * dt;
		fy = j - vel_ijk.y * dt;
		fz = k - vel_ijk.z * dt;

		if (fx < 0.0) fx = 0.0;
		if (fx > nx - 1) fx = nx - 1.0;
		if (fy < 0.0) fy = 0.0f;
		if (fy > ny - 1) fy = ny - 1.0;
		if (fz < 0.0) fz = 0.0f;
		if (fz > nz - 1) fz = nz - 1.0;

		ix = uint(fx);      iy = uint(fy);		iz = uint(fz);
		fx -= ix;			fy -= iy;			fz -= iz;

		if (ix == nx - 1) { ix = nx - 2; fx = 1.0; }
		if (iy == ny - 1) { iy = ny - 2; fy = 1.0; }
		if (iz == nz - 1) { iz = nz - 2; fz = 1.0; }

		w000 = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
		w100 = fx * (1.0f - fy) * (1.0f - fz);
		w010 = (1.0f - fx) * fy * (1.0f - fz);
		w001 = (1.0f - fx) * (1.0f - fy) * fz;
		w111 = fx * fy * fz;
		w011 = (1.0f - fx) * fy * fz;
		w101 = fx * (1.0f - fy) * fz;
		w110 = fx * fy * (1.0f - fz);

		Vec3f vel_ijk_p = w000 * vel(ix, iy, iz) + w100 * vel(ix + 1, iy, iz) +
			w010 * vel(ix, iy + 1, iz) + w001 * vel(ix, iy, iz + 1) +
			w111 * vel(ix + 1, iy + 1, iz + 1) + w011 * vel(ix, iy + 1, iz + 1) +
			w101 * vel(ix + 1, iy, iz + 1) + w110 * vel(ix + 1, iy + 1, iz);

		vel_ijk = (vel_ijk_p + vel_ijk) * 0.5f;

		fx = i - vel_ijk.x*dt;
		fy = j - vel_ijk.y*dt;
		fz = k - vel_ijk.z*dt;

		if (fx < 0.0f) fx = 0.0f;
		if (fx > nx - 1) fx = nx - 1.0f;
		if (fy < 0.0f) fy = 0.0f;
		if (fy > ny - 1) fy = ny - 1.0f;
		if (fz < 0.0f) fz = 0.0f;
		if (fz > nz - 1) fz = nz - 1.0f;

		ix = (int)fx;		iy = (int)fy;		iz = (int)fz;
		fx -= ix;			fy -= iy;			fz -= iz;

		if (ix == nx - 1) { ix = nx - 2; fx = 1.0f; }
		if (iy == ny - 1) { iy = ny - 2; fy = 1.0f; }
		if (iz == nz - 1) { iz = nz - 2; fz = 1.0f; }

		w000 = (1.0f - fx)*(1.0f - fy)*(1.0f - fz);
		w100 = fx*(1.0f - fy)*(1.0f - fz);
		w010 = (1.0f - fx)*fy*(1.0f - fz);
		w001 = (1.0f - fx)*(1.0f - fy)*fz;
		w111 = fx*fy*fz;
		w011 = (1.0f - fx)*fy*fz;
		w101 = fx*(1.0f - fy)*fz;
		w110 = fx*fy*(1.0f - fz);

		dst(i, j, k) = w000 * src(ix, iy, iz) + w100 * src(ix + 1, iy, iz) + w010 * src(ix, iy + 1, iz) + w001 * src(ix, iy, iz + 1)
			+ w111 * src(ix + 1, iy + 1, iz + 1) + w011 * src(ix, iy + 1, iz + 1) + w101 * src(ix + 1, iy, iz + 1) + w110 * src(ix + 1, iy + 1, iz);
	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::AdvectBackward(Grid3f dst, Grid3f src, Grid3f vel, float dt)
	{
		dim3 gridDims, blockDims;
		uint3 fDims = make_uint3(src.nx(), src.ny(), src.nz());
		computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

		K_AdvectBackward << < gridDims, blockDims >> >(dst, src, vel, dt);
	}

	__device__ float K_SharpeningWeight(float dist)
	{
		float fx = dist - floor(dist);
		fx = 1.0f - 2.0f*abs(fx - 0.5f);

		if (fx < 0.01f)
		{
			fx = 0.0f;
		}

		return fx;
	}

	template<typename Grid1f, typename Grid3f>
	__global__ void K_Sharpening(
		Grid1f dst, 
		Grid3f dir, 
		Grid1f src, 
		Grid1f vel_u, 
		Grid1f vel_v, 
		Grid1f vel_w,
		Grid1f omega,
		float gamma,
		float h,
		float dt)
	{
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		const int j = blockIdx.y * blockDim.y + threadIdx.y;
		const int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = dst.nx();
		int ny = dst.ny();
		int nz = dst.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		int i_minus = glm::clamp(i - 1, 0, nx - 1);
		int i_plus = glm::clamp(i + 1, 0, nx - 1);

		int j_minus = glm::clamp(j - 1, 0, ny - 1);
		int j_plus = glm::clamp(j + 1, 0, ny - 1);

		int k_minus = glm::clamp(k - 1, 0, nz - 1);
		int k_plus = glm::clamp(k + 1, 0, nz - 1);

// 			float norm_x, norm_y, norm_z;
// 
// 			norm_x = src(i_plus, j, k) - src(i_minus, j, k);
// 			norm_y = src(i, j_plus, k) - src(i, j_minus, k);
// 			norm_z = src(i, j, k_plus) - src(i, j, k_minus);
// 
// 			float l = sqrt(norm_x*norm_x + norm_y*norm_y + norm_z*norm_z);
// 			if (l < EPSILON)
// 			{
// 				norm_x = 0.0f;
// 				norm_y = 0.0f;
// 				norm_z = 0.0f;
// 			}
// 			else
// 			{
// 				norm_x /= l;
// 				norm_y /= l;
// 				norm_z /= l;
// 			}
// 			
// 
// 			dir(i, j, k) = make_float3(norm_x, norm_y, norm_z);
// 			dst(i, j, k) = src(i, j, k);

//			__syncthreads();

		int k0, k1;
		Vec3f n0, n1;
		float c1, c0, dc;

		float ceo = 24.0f * gamma / h;
		k0 = dir.index(i, j, k);

		float weight = 0.0f;


		//-----------------------------------------------------------------------
		//i and i+1
			
		if (i < nx - 1)
		{
			k1 = dir.index(i_plus, j, k);
			//if (src[k0] < 1.0f && src[k1] < 1.0f)
			{
				n0 = dir[k0];
				n1 = dir[k1];

				c1 = src[k1] * (1.0f - src[k1])*n1.x / omega[k1];
				c0 = src[k0] * (1.0f - src[k0])*n0.x / omega[k0];

				dc = 0.5f*(c1 + c0)*ceo*dt;

				weight = K_SharpeningWeight(vel_u(i + 1, j, k)*dt / h);
				atomicAdd(&(dst[k0]), -weight*dc);
				atomicAdd(&(dst[k1]), weight*dc);

// 					weight = K_SharpeningWeight(vel_u(i+1, j, k));
// 
// 					dst[k0] -= weight*dc;
			}
				
		}


		//j and j+1
		if (j < ny - 1)
		{
			k1 = dir.index(i, j_plus, k);
			//if (src[k0] < 1.0f && src[k1] < 1.0f)
			{
				n0 = dir[k0];
				n1 = dir[k1];

				c1 = src[k1] * (1.0f - src[k1])*n1.y / omega[k1];
				c0 = src[k0] * (1.0f - src[k0])*n0.y / omega[k0];

				dc = 0.5f*(c1 + c0)*ceo*dt;

				weight = K_SharpeningWeight(vel_v(i, j + 1, k)*dt / h);
	 			atomicAdd(&(dst[k0]), -weight*dc);
				atomicAdd(&(dst[k1]), weight*dc);
// 					weight = K_SharpeningWeight(vel_v(i, j + 1, k));
// 
// 					dst[k0] -= weight*dc;
				//dst[k1] += dc;
			}
		}

		//k and k+1
		if (k < nz - 1)
		{
			k1 = dir.index(i, j, k_plus);
			//if (src[k0] < 1.0f && src[k1] < 1.0f)
			{
				n0 = dir[k0];
				n1 = dir[k1];

				c1 = src[k1] * (1.0f - src[k1])*n1.z / omega[k1];
				c0 = src[k0] * (1.0f - src[k0])*n0.z / omega[k0];

				dc = 0.5f*(c1 + c0)*ceo*dt;

				weight = K_SharpeningWeight(vel_w(i, j, k + 1)*dt / h);
				atomicAdd(&(dst[k0]), -weight*dc);
				atomicAdd(&(dst[k1]), weight*dc);
// 					weight = K_SharpeningWeight(vel_w(i, j, k + 1));
// 
// 					dst[k0] -= weight*dc;
				//dst[k1] += dc;
			}
 		}
	}

	template<typename Grid1f, typename Grid3f>
	__global__ void K_ComputeNormals(Grid1f dst, Grid3f dir, Grid1f src)
	{
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		const int j = blockIdx.y * blockDim.y + threadIdx.y;
		const int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = src.nx();
		int ny = src.ny();
		int nz = src.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		int i_minus = glm::clamp(i - 1, 0, nx - 1);
		int i_plus = glm::clamp(i + 1, 0, nx - 1);

		int j_minus = glm::clamp(j - 1, 0, ny - 1);
		int j_plus = glm::clamp(j + 1, 0, ny - 1);

		int k_minus = glm::clamp(k - 1, 0, nz - 1);
		int k_plus = glm::clamp(k + 1, 0, nz - 1);

		float norm_x, norm_y, norm_z;

		norm_x = src(i_plus, j, k) - src(i_minus, j, k);
		norm_y = src(i, j_plus, k) - src(i, j_minus, k);
		norm_z = src(i, j, k_plus) - src(i, j, k_minus);

		float l = sqrt(norm_x*norm_x + norm_y*norm_y + norm_z*norm_z);
		if (l < EPSILON)
		{
			norm_x = 0.0f;
			norm_y = 0.0f;
			norm_z = 0.0f;
		}
		else
		{
			norm_x /= l;
			norm_y /= l;
			norm_z /= l;
		}

		dir(i, j, k) = Vec3f(norm_x, norm_y, norm_z);
		dst(i, j, k) = src(i, j, k);
	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::Sharpening(Grid1f dst, Grid3f dir, Grid1f src, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid1f omega, float gamma,
		float h, float dt)
	{
		dst.reset();

		dim3 gridDims, blockDims;
		uint3 fDims = make_uint3(src.nx(), src.ny(), src.nz());
		computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);
		K_ComputeNormals << < gridDims, blockDims >> >(dst, dir, src);
		K_Sharpening << < gridDims, blockDims >> >(dst, dir, src, vel_u, vel_v, vel_w, omega, gamma, h, dt);
	}

	__device__ float K_MapDiffusion(float m, float mag_v)
	{
		float weight = 1.0f;
//			if (mag_v < 0.001f && m < 0.1f)
// 			{
// 				weight = 0.0f;
// 			}
		if (m > 1.0f || m < 0.0f)	return 100.0f*weight;
		else return 0.0f;
	}

	template<typename Grid1f, typename Grid3f>
	__global__ void K_JacobiStep(Grid1f dst, Grid1f src, Grid1f buf, Grid3f vel, float a, float c)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = dst.nx();
		int ny = dst.ny();
		int nz = dst.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		int k0 = src.index(i, j, k);
		int nxy = nx*ny;

// 			float c1 = 1.0f / c;
// 			float c2 = (1.0f - c1) / 6.0f;

		float mag_v = vel(i, j, k).norm();

		int i_minus = glm::clamp(i - 1, 0, nx - 1);
		int i_plus = glm::clamp(i + 1, 0, nx - 1);

		int j_minus = glm::clamp(j - 1, 0, ny - 1);
		int j_plus = glm::clamp(j + 1, 0, ny - 1);

		int k_minus = glm::clamp(k - 1, 0, nz - 1);
		int k_plus = glm::clamp(k + 1, 0, nz - 1);

		float m_ijk = src[k0];

		float ax0 = a*K_MapDiffusion(0.5f*(m_ijk + src(i_minus, j, k)), mag_v);
		float ax1 = a*K_MapDiffusion(0.5f*(m_ijk + src(i_plus, j, k)), mag_v);

		float ay0 = a*K_MapDiffusion(0.5f*(m_ijk + src(i, j_minus, k)), mag_v);
		float ay1 = a*K_MapDiffusion(0.5f*(m_ijk + src(i, j_plus, k)), mag_v);

		float az0 = a*K_MapDiffusion(0.5f*(m_ijk + src(i, j, k_minus)), mag_v);
		float az1 = a*K_MapDiffusion(0.5f*(m_ijk + src(i, j, k_plus)), mag_v);

		float c1 = 1.0f / (1.0f + ax0 + ax1 + ay0 + ay1 + az0 + az1);

		//dst[k0] = (c1*src[k0] + c2*(buf(i_plus, j, k) + buf(i_minus, j, k) + buf(i, j_plus, k) + buf(i, j_minus, k) + buf(i, j, k_plus) + buf(i, j, k_minus)));
		dst[k0] = (c1*src[k0] + c1*ax1*buf(i_plus, j, k) + c1*ax0*buf(i_minus, j, k) + c1*ay1*buf(i, j_plus, k) + c1*ay0*buf(i, j_minus, k) + c1*az1*buf(i, j, k_plus) + c1*az0*buf(i, j, k_minus));
	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::Jacobi(Grid1f dst, Grid1f src, Grid1f buf, Grid3f vel, float a, float c, int iteration)
	{
		dim3 gridDims, blockDims;
		uint3 fDims = make_uint3(src.nx(), src.ny(), src.nz());
		computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

		K_CopyData << < gridDims, blockDims >> >(dst, src);
		for (int i = 0; i < iteration; i++)
		{
			K_CopyData << < gridDims, blockDims >> >(buf, dst);
			K_JacobiStep << < gridDims, blockDims >> >(dst, src, buf, vel, a, c);
		}
	}

	__device__ float D_VisocityFromShearingRate(float rate)
	{
		if (rate > 0.2f)
		{
			return MU_0;
		}
		else if (rate < 0.01f)
		{
			return MU_INF;
		}
		else
		{
			return MU_INF + (MU_0 - MU_INF)*(rate - 0.01f) / 0.19f;
		}
	}

	template<typename Grid3f>
	__global__ void K_SetVelocityBoundary(Grid3f vel)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = vel.nx();
		int ny = vel.ny();
		int nz = vel.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		if (i == 0) { vel(i, j, k) = Vec3f(0.0f); return; }
		if (i == nx - 1) { vel(i, j, k) = Vec3f(0.0f); return; }

		if (j == 0) { vel(i, j, k) = Vec3f(0.0f); return; }
		if (j == ny - 1) { vel(i, j, k) = Vec3f(0.0f); return; }

		if (k == 0) { vel(i, j, k) = Vec3f(0.0f); return; }
		if (k == nz - 1) { vel(i, j, k) = vel(i, j, k-1); return; }
	}

	template<typename Grid3f, typename Grid1f>
	__global__ void K_Laplacian(Grid3f dst, Grid3f src, Grid3f buf, Grid1f mass, float a)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = dst.nx();
		int ny = dst.ny();
		int nz = dst.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		int k0 = src.index(i, j, k);
		int nxy = nx*ny;

		float m = mass[k0];

		m = m > 1.0f ? 1.0f : m;
		m = m < 0.0f ? 0.0f : m;
		float vis = (VIS1*m + VIS2*(1.0f - m));

//			if (k > nz*0.75f)
//			{
//				vis = 0.0f;
//			}

		float c0 = 1.0f + 6.0f*a*vis;

		float c1 = 1.0f / c0;
		float c2 = (1.0f - c1) / 6.0f;

		int i_minus = glm::clamp(i - 1, 0, nx - 1);
		int i_plus = glm::clamp(i + 1, 0, nx - 1);

		int j_minus = glm::clamp(j - 1, 0, ny - 1);
		int j_plus = glm::clamp(j + 1, 0, ny - 1);

		int k_minus = glm::clamp(k - 1, 0, nz - 1);
		int k_plus = glm::clamp(k + 1, 0, nz - 1);


		dst[k0] = (c1*src[k0] + c2*(buf(i_plus, j, k) + buf(i_minus, j, k) + buf(i, j_plus, k) + buf(i, j_minus, k) + buf(i, j, k_plus) + buf(i, j, k_minus)));
	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::ApplyViscosity(Grid3f dst, Grid3f src, Grid3f buf, Grid1f mass, float a, int iteration)
	{
		dim3 gridDims, blockDims;
		uint3 fDims = make_uint3(src.nx(), src.ny(), src.nz());
		computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

		K_CopyData << < gridDims, blockDims >> >(dst, src);
		K_SetVelocityBoundary << < gridDims, blockDims >> >(src);

		for (int i = 0; i < iteration; i++)
		{
			buf.assign(dst);
			K_SetVelocityBoundary << < gridDims, blockDims >> >(buf);
			K_Laplacian << < gridDims, blockDims >> >(dst, src, buf, mass, a);
		}
	}

	template<typename GridCoef, typename Grid1f>
	__global__ void K_PrepareForProjection(GridCoef coefMatrix, Grid1f RHS, Grid1f mass, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, float h, float dt)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = mass.nx();
		int ny = mass.ny();
		int nz = mass.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		float hh = h*h;

		float div_ijk = 0.0f;

		Coef A_ijk;

		A_ijk.a = 0.0f;
		A_ijk.x0 = 0.0f;
		A_ijk.x1 = 0.0f;
		A_ijk.y0 = 0.0f;
		A_ijk.y1 = 0.0f;
		A_ijk.z0 = 0.0f;
		A_ijk.z1 = 0.0f;

		float m_ijk = mass(i, j, k);

		if (i+1 < nx) {
			float c = 0.5f*(m_ijk + mass(i + 1, j, k));
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

			A_ijk.a += term;
			A_ijk.x1 += term;

// 				matrix.add_to_element(index, index, term);
// 				matrix.add_to_element(index, index + 1, -term);
		}
		div_ijk -= vel_u(i + 1, j, k) / h;
		//rhs[index] -= vel_u(i + 1, j, k) / h;

		//left neighbour
		if (i-1 >= 0) {
			float c = 0.5f*(m_ijk + mass(i - 1, j, k));
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

			A_ijk.a += term;
			A_ijk.x0 += term;
// 				matrix.add_to_element(index, index, term);
// 				matrix.add_to_element(index, index - 1, -term);
		}
		div_ijk += vel_u(i, j, k) / h;
		//rhs[index] += vel_u(i, j, k) / h;

		//top neighbour
		if (j+1 < ny) {
			float c = 0.5f*(m_ijk + mass(i, j + 1, k));
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

			A_ijk.a += term;
			A_ijk.y1 += term;
// 				matrix.add_to_element(index, index, term);
// 				matrix.add_to_element(index, index + ni, -term);
		}
		div_ijk -= vel_v(i, j + 1, k) / h;

		//bottom neighbour
		if (j-1 >= 0) {
			float c = 0.5f*(m_ijk + mass(i, j - 1, k));
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

			A_ijk.a += term;
			A_ijk.y0 += term;
// 				matrix.add_to_element(index, index, term);
// 				matrix.add_to_element(index, index - ni, -term);
		}
		div_ijk += vel_v(i, j, k) / h;


		//far neighbour

		if (k+1 < nz) {
			float c = 0.5f*(m_ijk + mass(i, j, k + 1));
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

			A_ijk.a += term;
			A_ijk.z1 += term;
// 				matrix.add_to_element(index, index, term);
// 				matrix.add_to_element(index, index + ni*nj, -term);
		}
		div_ijk -= vel_w(i, j, k + 1) / h;

		//near neighbour

		if (k-1 >= 0) {
			float c = 0.5f*(m_ijk + mass(i, j, k - 1));
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			float term = dt / hh / (RHO1*c + RHO2*(1.0f - c));

			A_ijk.a += term;
			A_ijk.z0 += term;
// 				matrix.add_to_element(index, index, term);
// 				matrix.add_to_element(index, index - ni*nj, -term);
		}
		div_ijk += vel_w(i, j, k) / h;

		if (m_ijk > 1.0)
		{
			div_ijk += 0.5f*pow((m_ijk - 1.0f), 1.0f) / dt;
		}
// 			else if (m_ijk > 0.0f)
// 			{
// 				div_ijk -= 0.001f;
// 			}

		coefMatrix(i, j, k) = A_ijk;
		RHS(i, j, k) = div_ijk;
	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::PrepareForProjection(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, GridCoef coefMatrix, Grid1f RHS, Grid1f mass, float h, float dt)
	{
		dim3 gridDims, blockDims;
		uint3 fDims = make_uint3(mass.nx(), mass.ny(), mass.nz());
		computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

		K_PrepareForProjection << < gridDims, blockDims >> >(coefMatrix, RHS, mass, vel_u, vel_v, vel_w, h, dt);
	}

	template<typename Grid1f, typename GridCoef>
	__global__ void K_Projection(Grid1f pressure, Grid1f buf, GridCoef coefMatrix, Grid1f RHS, int numIter)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = pressure.nx();
		int ny = pressure.ny();
		int nz = pressure.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;
			
		int k0 = coefMatrix.index(i, j, k);
		Coef A_ijk = coefMatrix[k0];
			
		float a = A_ijk.a;
		float x0 = A_ijk.x0;
		float x1 = A_ijk.x1;
		float y0 = A_ijk.y0;
		float y1 = A_ijk.y1;
		float z0 = A_ijk.z0;
		float z1 = A_ijk.z1;
//			pressure[k0] = 0.0f;
 		float p_ijk;
// 			for (int it = 0; it < 1; it++)
 		{
// 				buf[k0] = 0.0f;// pressure[k0];
// 				__syncthreads();

			p_ijk = RHS[k0];
			if (i > 0) p_ijk += x0*buf(i - 1, j, k);
			if (i < nx - 1) p_ijk += x1*buf(i + 1, j, k);
			if (j > 0) p_ijk += y0*buf(i, j - 1, k);
			if (j < ny - 1) p_ijk += y1*buf(i, j + 1, k);
			if (k > 0) p_ijk += z0*buf(i, j, k - 1);
			if (k < nz - 1) p_ijk += z1*buf(i, j, k + 1);

			pressure[k0] = p_ijk / a;
		}
	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::Projection(Grid1f pressure, Grid1f buf, GridCoef coefMatrix, Grid1f RHS, int numIter)
	{
		dim3 gridDims, blockDims;
		uint3 fDims = make_uint3(RHS.nx(), RHS.ny(), RHS.nz());
		computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

		//pressure.reset();
		for (int i = 0; i < numIter; i++)
		{
			//K_CopyData << < gridDims, blockDims >> >(buf, pressure);
			buf.assign(pressure);
			K_Projection << < gridDims, blockDims >> >(pressure, buf, coefMatrix, RHS, numIter);
		}
	}

	template<typename Grid1f>
	__global__ void K_UpdateVelocity(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid1f pressure, Grid1f mass, float h, float dt)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = mass.nx();
		int ny = mass.ny();
		int nz = mass.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		//u(i, j, k) -= (p_(i, j, k) - p_(i - 1, j, k)) / h / rho
		if (i > 0)
		{
			float c = 0.5f * (mass(i - 1, j, k) + mass(i, j, k));
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			vel_u(i, j, k) -= dt * (pressure(i, j, k) - pressure(i - 1, j, k)) / h / (c * RHO1 + (1.0f - c) * RHO2);
		}
			
		//v(i, j, k) -= (p_(i, j, k) - p_(i, j - 1, k)) / h / rho
		if (j > 0)
		{
			float c = 0.5f * (mass(i, j, k) + mass(i, j - 1, k));
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			vel_v(i, j, k) -= dt * (pressure(i, j, k) - pressure(i, j - 1, k)) / h / (c * RHO1 + (1.0f - c) * RHO2);
		}
			
		//w(i, j, k) -= (p_(i, j, k) - p_(i, j, k - 1)) / h / rho
		if (k > 0)
		{
			float c = 0.5f * (mass(i, j, k) + mass(i, j, k - 1));
			c = c > 1.0f ? 1.0f : c;
			c = c < 0.0f ? 0.0f : c;
			vel_w(i, j, k) -= dt * (pressure(i, j, k) - pressure(i, j, k - 1)) / h / (c * RHO1 + (1.0f - c) * RHO2);
		}
	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::UpdateVelocity(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid1f pressure, Grid1f mass, float h, float dt)
	{
		dim3 gridDims, blockDims;
		uint3 fDims = make_uint3(mass.nx(), mass.ny(), mass.nz());
		computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

		K_UpdateVelocity << < gridDims, blockDims >> >(vel_u, vel_v, vel_w, pressure, mass, h, dt);
	}

	template<typename Grid3f>
	__global__ void K_ApplyGravity(
		Grid3f v,
		Vec3f g,
		float dt)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		if (i >= v.nx()) return;
		if (j >= v.ny()) return;
		if (k >= v.nz()) return;

		v(i, j, k) += g * dt;
	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::ApplyGravity(Grid3f v, Vec3f g, float dt)
	{
		dim3 gridDims, blockDims;
		uint3 fDims = make_uint3(v.nx(), v.ny(), v.nz());
		computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

		K_ApplyGravity << < gridDims, blockDims >> > (v, g, dt);
	}

	template<typename Grid1f>
	__global__ void K_SetU(Grid1f vel_u)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = vel_u.nx();
		int ny = vel_u.ny();
		int nz = vel_u.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		//Impose free-slip boundary condition
		if (i == 0) { vel_u(i, j, k) = 0; }
		if (i == nx - 1) { vel_u(i, j, k) = 0; }
	}

	template<typename Grid1f>
	__global__ void K_SetV(Grid1f vel_v)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = vel_v.nx();
		int ny = vel_v.ny();
		int nz = vel_v.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		//Impose free-slip boundary condition
		if (j == 0) { vel_v(i, j, k) = 0; }
		if (j == ny - 1) { vel_v(i, j, k) = 0; }
	}

	template<typename Grid1f>
	__global__ void K_SetW(Grid1f vel_w)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int k = blockIdx.z * blockDim.z + threadIdx.z;

		int nx = vel_w.nx();
		int ny = vel_w.ny();
		int nz = vel_w.nz();

		if (i >= nx) return;
		if (j >= ny) return;
		if (k >= nz) return;

		//Impose free-slip boundary condition
		if (k == 0) { vel_w(i, j, k) = 0; }
		if (k == nz - 1) { vel_w(i, j, k) = 0; }

	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::SetU(Grid1f vel_u)
	{
		dim3 gridDims, blockDims;
		uint3 fDims = make_uint3(vel_u.nx(), vel_u.ny(), vel_u.nz());
		computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

		K_SetU << < gridDims, blockDims >> >(vel_u);
	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::SetV(Grid1f vel_v)
	{
		dim3 gridDims, blockDims;
		uint3 fDims = make_uint3(vel_v.nx(), vel_v.ny(), vel_v.nz());
		computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

		K_SetV << < gridDims, blockDims >> >(vel_v);
	}

	template<typename TDataType>
	void PhaseFieldKernels<TDataType>::SetW(Grid1f vel_w)
	{
		dim3 gridDims, blockDims;
		uint3 fDims = make_uint3(vel_w.nx(), vel_w.ny(), vel_w.nz());
		computeGridSize3D(fDims, make_uint3(32, 8, 1), gridDims, blockDims);

		K_SetW << < gridDims, blockDims >> >(vel_w);
	}

	DEFINE_CLASS(PhaseFieldKernels);
}