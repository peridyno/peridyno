#include "OceanPatch.h"

#include <iostream>
#include <fstream>
#include <string.h>

namespace dyno
{

	//Round a / b to nearest higher integer value
	int cuda_iDivUp(int a, int b)
	{
		return (a + (b - 1)) / b;
	}


	// complex math functions
	__device__
		float2 conjugate(float2 arg)
	{
		return make_float2(arg.x, -arg.y);
	}

	__device__
		float2 complex_exp(float arg)
	{
		return make_float2(cosf(arg), sinf(arg));
	}

	__device__
		float2 complex_add(float2 a, float2 b)
	{
		return make_float2(a.x + b.x, a.y + b.y);
	}

	__device__
		float2 complex_mult(float2 ab, float2 cd)
	{
		return make_float2(ab.x * cd.x - ab.y * cd.y, ab.x * cd.y + ab.y * cd.x);
	}

	// generate wave heightfield at time t based on initial heightfield and dispersion relationship
	__global__ void generateSpectrumKernel(float2 *h0,
		float2 *ht,
		unsigned int in_width,
		unsigned int out_width,
		unsigned int out_height,
		float t,
		float patchSize)
	{
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		unsigned int in_index = y * in_width + x;
		unsigned int in_mindex = (out_height - y)*in_width + (out_width - x); // mirrored
		unsigned int out_index = y * out_width + x;

		// calculate wave vector
		float2 k;
		k.x = (-(int)out_width / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
		k.y = (-(int)out_width / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

		// calculate dispersion w(k)
		float k_len = sqrtf(k.x*k.x + k.y*k.y);
		float w = sqrtf(9.81f * k_len);

		if ((x < out_width) && (y < out_height))
		{
			float2 h0_k = h0[in_index];
			float2 h0_mk = h0[in_mindex];

			// output frequency-space complex values
			ht[out_index] = complex_add(complex_mult(h0_k, complex_exp(w * t)), complex_mult(conjugate(h0_mk), complex_exp(-w * t)));
			//ht[out_index] = h0_k;
		}
	}

	// update height map values based on output of FFT
	__global__ void updateHeightmapKernel(float  *heightMap,
		float2 *ht,
		unsigned int width)
	{
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		unsigned int i = y * width + x;

		// cos(pi * (m1 + m2))
		float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;

		heightMap[i] = ht[i].x * sign_correction;
	}

	// update height map values based on output of FFT
	__global__ void updateHeightmapKernel_y(float  *heightMap,
		float2 *ht,
		unsigned int width)
	{
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		unsigned int i = y * width + x;

		// cos(pi * (m1 + m2))
		float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;

		heightMap[i] = ht[i].y * sign_correction;
	}

	// generate slope by partial differences in spatial domain
	__global__ void calculateSlopeKernel(float *h, float2 *slopeOut, unsigned int width, unsigned int height)
	{
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		unsigned int i = y * width + x;

		float2 slope = make_float2(0.0f, 0.0f);

		if ((x > 0) && (y > 0) && (x < width - 1) && (y < height - 1))
		{
			slope.x = h[i + 1] - h[i - 1];
			slope.y = h[i + width] - h[i - width];
		}

		slopeOut[i] = slope;
	}


	__global__ void generateDispalcementKernel(
		float2 *ht,
		float2 *Dxt,
		float2 *Dzt,
		unsigned int width,
		unsigned int height,
		float patchSize)
	{
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		unsigned int id = y * width + x;

		// calculate wave vector
		float kx = (-(int)width / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
		float ky = (-(int)height / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);
		float k_squared = kx * kx + ky * ky;
		if (k_squared == 0.0f)
		{
			k_squared = 1.0f;
		}
		kx = kx / sqrtf(k_squared);
		ky = ky / sqrtf(k_squared);

		float2 ht_ij = ht[id];
		float2 idoth = make_float2(-ht_ij.y, ht_ij.x);

		Dxt[id] = kx * idoth;
		Dzt[id] = ky * idoth;
	}

	// 
	OceanPatch::OceanPatch(int size, float patchSize, int windType, std::string name)
		: Node(name)
	{
		std::ifstream input("windparam.txt", std::ios::in);
		for (int i = 0; i <= 12; i++)
		{
			WindParam param;
			int dummy;
			input >> dummy;
			input >> param.windSpeed;
			input >> param.A;
			input >> param.choppiness;
			input >> param.global;
			m_params.push_back(param);
		}


		m_size = size;

		m_spectrumW = size + 1;
		m_spectrumH = size + 4;

		m_windType = windType;
		m_realPatchSize = patchSize;
		m_windSpeed = m_params[m_windType].windSpeed;
		A = m_params[m_windType].A;
		m_maxChoppiness = m_params[m_windType].choppiness;
		m_choppiness = m_params[m_windType].choppiness;
		m_globalShift = m_params[m_windType].global;

		m_h0 = NULL;
		m_ht = NULL;

		initialize();
	}

	OceanPatch::OceanPatch(int size, float wind_dir, float windSpeed, float A_p, float max_choppiness, float global) {
		m_size = size;
		m_spectrumW = size + 1;
		m_spectrumH = size + 4;
		m_realPatchSize = m_size;
		windDir = wind_dir;
		m_windSpeed = windSpeed;
		A = A_p;
		m_maxChoppiness = max_choppiness;
		m_choppiness = 1.0f;
		m_globalShift = global;
		m_h0 = NULL;
		m_ht = NULL;
		initialize();
	}

	OceanPatch::~OceanPatch()
	{
		cudaFree(m_h0);
		cudaFree(m_ht);
		cudaFree(m_Dxt);
		cudaFree(m_Dzt);
		cudaFree(m_displacement);
		cudaFree(m_gradient);
		//glDeleteTextures(1, &m_displacement_texture);
		//glDeleteTextures(1, &m_gradient_texture);
		//cudaCheck(cudaGraphicsUnregisterResource(m_cuda_displacement_texture));
		//cudaCheck(cudaGraphicsUnregisterResource(m_cuda_gradient_texture));
	}

	bool OceanPatch::initialize()
	{
		cufftPlan2d(&fftPlan, m_size, m_size, CUFFT_C2C);

		int spectrumSize = m_spectrumW * m_spectrumH * sizeof(float2);
		cuSafeCall(cudaMalloc((void **)&m_h0, spectrumSize));
		//synchronCheck;
		float2* host_h0 = (float2 *)malloc(spectrumSize);
		generateH0(host_h0);

		cuSafeCall(cudaMemcpy(m_h0, host_h0, spectrumSize, cudaMemcpyHostToDevice));

		int outputSize = m_size * m_size * sizeof(float2);
		cudaMalloc((void **)&m_ht, outputSize);
		cudaMalloc((void **)&m_Dxt, outputSize);
		cudaMalloc((void **)&m_Dzt, outputSize);
		cudaMalloc((void **)&m_displacement, m_size*m_size * sizeof(float4));
		cuSafeCall(cudaMalloc((void **)&m_gradient, m_size*m_size * sizeof(float4)));

		//gl_utility::createTexture(m_size, m_size, GL_RGBA32F, m_displacement_texture, GL_REPEAT, GL_LINEAR, GL_LINEAR, GL_RGBA, GL_FLOAT);
		//cudaCheck(cudaGraphicsGLRegisterImage(&m_cuda_displacement_texture, m_displacement_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
		//gl_utility::createTexture(m_size, m_size, GL_RGBA32F, m_gradient_texture, GL_REPEAT, GL_LINEAR, GL_LINEAR, GL_RGBA, GL_FLOAT);
		//cudaCheck(cudaGraphicsGLRegisterImage(&m_cuda_gradient_texture, m_gradient_texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
		return true;
	}

	__global__ void O_UpdateDisplacement(
		float4* displacement,
		float2* Dh,
		float2* Dx,
		float2* Dz,
		int patchSize)
	{
		unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
		if (i < patchSize && j < patchSize)
		{
			int id = i + j * patchSize;

			float sign_correction = ((i + j) & 0x01) ? -1.0f : 1.0f;
			float h_ij = sign_correction * Dh[id].x;
			float x_ij = sign_correction * Dx[id].x;
			float z_ij = sign_correction * Dz[id].x;

			displacement[id] = make_float4(x_ij, h_ij, z_ij, 0);
		}
	}

	__global__ void O_UpdateGradient(float4* displacement, float4* gradiant, int patchSize)
	{
		unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
		if (i >= patchSize || j >= patchSize)return;
		int i_minus_one = (i - 1 + patchSize) % patchSize;
		int i_plus_one = (i + 1) % patchSize;
		int j_minus_one = (j - 1 + patchSize) % patchSize;
		int j_plus_one = (j + 1) % patchSize;

		float4 Dx = (displacement[i_plus_one + j * patchSize] - displacement[i_minus_one + j * patchSize]) / 2;
		float4 Dz = (displacement[i + j_plus_one * patchSize] - displacement[i + j_minus_one * patchSize]) / 2;
		float la = 0.8f;
		float Jxx = 1 + Dx.x * la;
		float Jyy = 1 + Dz.z * la;
		float Jyx = Dx.z * la;
		float Jxy = Dz.x * la;
		float J = Jxx * Jyy - Jyx * Jxy;
		float breakArea = fminf(fmaxf((0.7f - J)*0.5f, 0.0f), 1.0f);
		float lastZ = gradiant[j*patchSize + i].z;
		float currentZ = (lastZ + breakArea)*0.8f;
		gradiant[j*patchSize + i] = make_float4(Dx.y, Dz.y, currentZ, J);
		//gradiant[i + j*patchSize] = make_float4(Dx.x, Dx.z, Dz.x, Dz.z);

	}

	void OceanPatch::animate(float t)
	{
		t = m_fft_flow_speed * t;
		dim3 block(8, 8, 1);
		dim3 grid(cuda_iDivUp(m_size, block.x), cuda_iDivUp(m_size, block.y), 1);
		generateSpectrumKernel << <grid, block >> > (m_h0, m_ht, m_spectrumW, m_size, m_size, t, m_realPatchSize);
		cuSynchronize();
		generateDispalcementKernel << <grid, block >> > (m_ht, m_Dxt, m_Dzt, m_size, m_size, m_realPatchSize);
		cuSynchronize();

		cufftExecC2C(fftPlan, m_ht, m_ht, CUFFT_INVERSE);
		cufftExecC2C(fftPlan, m_Dxt, m_Dxt, CUFFT_INVERSE);
		cufftExecC2C(fftPlan, m_Dzt, m_Dzt, CUFFT_INVERSE);

		int x = (m_size + 16 - 1) / 16;
		int y = (m_size + 16 - 1) / 16;
		dim3 threadsPerBlock(16, 16);
		dim3 blocksPerGrid(x, y);
		O_UpdateDisplacement << <blocksPerGrid, threadsPerBlock >> > (m_displacement, m_ht, m_Dxt, m_Dzt, m_size);
		cuSynchronize();
		O_UpdateGradient << <blocksPerGrid, threadsPerBlock >> > (m_displacement, m_gradient, m_size);
		cuSynchronize();
	}

	float OceanPatch::getMaxChoppiness()
	{
		return m_maxChoppiness;
	}

	float OceanPatch::getChoppiness()
	{
		return m_choppiness;
	}

	void OceanPatch::generateH0(float2* h0)
	{
		for (unsigned int y = 0; y <= m_size; y++)
		{
			for (unsigned int x = 0; x <= m_size; x++)
			{
				float kx = (-(int)m_size / 2.0f + x) * (2.0f * CUDART_PI_F / m_realPatchSize);
				float ky = (-(int)m_size / 2.0f + y) * (2.0f * CUDART_PI_F / m_realPatchSize);

				float P = sqrtf(phillips(kx, ky, windDir, m_windSpeed, A, dirDepend));

				if (kx == 0.0f && ky == 0.0f)
				{
					P = 0.0f;
				}

				//float Er = urand()*2.0f-1.0f;
				//float Ei = urand()*2.0f-1.0f;
				float Er = gauss();
				float Ei = gauss();

				float h0_re = Er * P * CUDART_SQRT_HALF_F;
				float h0_im = Ei * P * CUDART_SQRT_HALF_F;

				int i = y * m_spectrumW + x;
				h0[i].x = h0_re;
				h0[i].y = h0_im;
			}
		}
	}

	float OceanPatch::phillips(float Kx, float Ky, float Vdir, float V, float A, float dir_depend)
	{
		float k_squared = Kx * Kx + Ky * Ky;

		if (k_squared == 0.0f)
		{
			return 0.0f;
		}

		// largest possible wave from constant wind of velocity v
		float L = V * V / g;

		float k_x = Kx / sqrtf(k_squared);
		float k_y = Ky / sqrtf(k_squared);
		float w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

		float phillips = A * expf(-1.0f / (k_squared * L * L)) / (k_squared * k_squared) * w_dot_k * w_dot_k;

		// filter out waves moving opposite to wind
		if (w_dot_k < 0.0f)
		{
			phillips *= dir_depend;
		}

		// damp out waves with very small length w << l
		//float w = L / 10000;
		//phillips *= expf(-k_squared * w * w);

		return phillips;
	}


	float OceanPatch::gauss()
	{
		float u1 = rand() / (float)RAND_MAX;
		float u2 = rand() / (float)RAND_MAX;

		if (u1 < 1e-6f)
		{
			u1 = 1e-6f;
		}

		return sqrtf(-2 * logf(u1)) * cosf(2 * CUDART_PI_F * u2);
	}

}