#include "OceanPatch.h"

#include <iostream>
#include <fstream>
#include <string.h>

#include "Topology/HeightField.h"

namespace dyno {

    //Round a / b to nearest higher integer value
    int cuda_iDivUp(int a, int b)
    {
        return (a + (b - 1)) / b;
    }

    // complex math functions
    __device__ Vec2f conjugate(Vec2f arg)
    {
        return Vec2f(arg.x, -arg.y);
    }

    __device__  Vec2f complex_exp(float arg)
    {
        return Vec2f(cosf(arg), sinf(arg));
    }

    __device__  Vec2f complex_add(Vec2f a, Vec2f b)
    {
        return Vec2f(a.x + b.x, a.y + b.y);
    }

    __device__ Vec2f complex_mult(Vec2f ab, Vec2f cd)
    {
        return Vec2f(ab.x * cd.x - ab.y * cd.y, ab.x * cd.y + ab.y * cd.x);
    }

    // generate wave heightfield at time t based on initial heightfield and dispersion relationship
    template <typename Coord>
    __global__ void generateSpectrumKernel(DArray2D<Coord> h0,
                                           DArray2D<Coord> ht,
                                           unsigned int    in_width,
                                           unsigned int    out_width,
                                           unsigned int    out_height,
                                           float           t,
                                           float           patchSize)
    {
        unsigned int x         = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y         = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int in_index  = y * in_width + x;
        unsigned int in_mindex = (out_height - y) * in_width + (out_width - x);  // mirrored
        unsigned int out_index = y * out_width + x;

        // calculate wave vector
        Coord k;
        k.x = (-( int )out_width / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
        k.y = (-( int )out_width / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

        // calculate dispersion w(k)
        float k_len = sqrtf(k.x * k.x + k.y * k.y);
        float w     = sqrtf(9.81f * k_len);

        if ((x < out_width) && (y < out_height))
        {
            Coord h0_k  = h0[in_index];
            Coord h0_mk = h0[in_mindex];

            // output frequency-space complex values
            ht[out_index] = complex_add(complex_mult(h0_k, complex_exp(w * t)), complex_mult(conjugate(h0_mk), complex_exp(-w * t)));
        }
    }

    // update height map values based on output of FFT
    template <typename Coord>
    __global__ void updateHeightmapKernel(float*       heightMap,
                                          Coord*       ht,
                                          unsigned int width)
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int i = y * width + x;

        // cos(pi * (m1 + m2))
        float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;

        heightMap[i] = ht(x, y).x * sign_correction;
    }

    // update height map values based on output of FFT
    template <typename Coord>
    __global__ void updateHeightmapKernel_y(float*       heightMap,
                                            Coord*       ht,
                                            unsigned int width)
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int i = y * width + x;

        // cos(pi * (m1 + m2))
        float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;

        heightMap[i] = ht(x, y).y * sign_correction;
    }

    // generate slope by partial differences in spatial domain
    template <typename Coord>
    __global__ void calculateSlopeKernel(float*       h, 
                                         Coord*      slopeOut,
                                         unsigned int width, 
                                         unsigned int height)
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int i = y * width + x;

        Coord slope = Coord(0.0f, 0.0f);

        if ((x > 0) && (y > 0) && (x < width - 1) && (y < height - 1))
        {
            slope.x = h[i + 1] - h[i - 1];
            slope.y = h[i + width] - h[i - width];
        }

        slopeOut(x, y) = slope;
    }

    template <typename Coord>
    __global__ void generateDispalcementKernel(
        DArray2D<Coord>      ht,
        DArray2D<Coord>      Dxt,
        DArray2D<Coord>      Dzt,
        unsigned int width,
        unsigned int height,
        float        patchSize)
    {
        unsigned int x  = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y  = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int id = y * width + x;

        // calculate wave vector
        float kx        = (-( int )width / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
        float ky        = (-( int )height / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);
        float k_squared = kx * kx + ky * ky;
        if (k_squared == 0.0f)
        {
            k_squared = 1.0f;
        }
        kx = kx / sqrtf(k_squared);
        ky = ky / sqrtf(k_squared);

        Coord ht_ij = ht(x, y);
        Coord idoth = Coord(-ht_ij.y, ht_ij.x);

        Dxt(x, y) = kx * idoth;
        Dzt(x, y) = ky * idoth;
    }

    template<typename TDataType>
    OceanPatch<TDataType>::OceanPatch(std::string name)
        : Node(name)
    {
    }

    template<typename TDataType>
    OceanPatch<TDataType>::OceanPatch(int size, float patchSize, int windType, std::string name)
        : Node(name)
    {
	    auto heights = std::make_shared<HeightField<TDataType>>();
	    this->stateTopology()->setDataPtr(heights);

        std::ifstream input("../../data/windparam.txt", std::ios::in);
        for (int i = 0; i <= 12; i++)
        {
            WindParam param;
            int       dummy;
            input >> dummy;
            input >> param.windSpeed;
            input >> param.A;
            input >> param.choppiness;
            input >> param.global;
            m_params.push_back(param);
        }

        mResolution = size;

        mSpectrumWidth = size + 1;
        mSpectrumHeight = size + 4;

        m_realPatchSize = patchSize;
    }

    template<typename TDataType>
    OceanPatch<TDataType>::OceanPatch(int size, float wind_dir, float windSpeed, float A_p, float max_choppiness, float global)
    {
	    auto heights = std::make_shared<HeightField<TDataType>>();
	    this->stateTopology()->setDataPtr(heights);

        mResolution          = size;
        mSpectrumWidth     = size + 1;
        mSpectrumHeight     = size + 4;
        m_realPatchSize = mResolution;
        windDir         = wind_dir;
        m_windSpeed     = windSpeed;
        A               = A_p;
        m_maxChoppiness = max_choppiness;
        mChoppiness    = 1.0f;
        m_globalShift   = global;
    }

    template<typename TDataType>
    OceanPatch<TDataType>::~OceanPatch()
    {     
        m_h0.clear();
        m_ht.clear();
        m_Dxt.clear();  
        m_Dzt.clear();     
        m_displacement.clear();
        m_gradient.clear();
    }

    template<typename TDataType>
    void OceanPatch<TDataType>::resetWindType()
    {
        m_windSpeed = m_params[var_my_windTypes.getValue()].windSpeed;
        A = m_params[var_my_windTypes.getValue()].A;
        m_maxChoppiness = m_params[var_my_windTypes.getValue()].choppiness;
        mChoppiness = m_params[var_my_windTypes.getValue()].choppiness;
        m_globalShift = m_params[var_my_windTypes.getValue()].global;
    }

    template<typename TDataType>
    void OceanPatch<TDataType>::resetStates()
    {
        resetWindType();

        cufftPlan2d(&fftPlan, mResolution, mResolution, CUFFT_C2C);

        int spectrumSize = mSpectrumWidth * mSpectrumHeight * sizeof(Vec2f);
        m_h0.resize(mSpectrumWidth, mSpectrumHeight);

        Vec2f* host_h0 = ( Vec2f* )malloc(spectrumSize);
        generateH0(host_h0);

        cuSafeCall(cudaMemcpy(m_h0.begin(), host_h0, spectrumSize, cudaMemcpyHostToDevice));

        m_ht.resize(mResolution, mResolution);
        m_Dxt.resize(mResolution, mResolution);
        m_Dzt.resize(mResolution, mResolution);
        m_displacement.resize(mResolution, mResolution);
        m_gradient.resize(mResolution, mResolution);

	    auto topo = TypeInfo::cast<HeightField<TDataType>>(this->stateTopology()->getDataPtr());
	    Real h = m_realPatchSize / mResolution;
	    topo->setExtents(mResolution, mResolution);
	    topo->setGridSpacing(h);
	    topo->setOrigin(Vec3f(-0.5*h*topo->width(), 0, -0.5*h*topo->height()));
    }

    float t = 0.0f;
    template<typename TDataType>
    void OceanPatch<TDataType>::updateStates()
    {
	    t += 0.016f;
	    this->animate(t);
    }

    template<typename Coord>
    __global__ void O_UpdateDisplacement(
        DArray2D<Vec4f> displacement,
        DArray2D<Coord> Dh,
        DArray2D<Coord> Dx,
        DArray2D<Coord> Dz,
        int             patchSize)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < patchSize && j < patchSize)
        {
            int id = i + j * patchSize;

            float sign_correction = ((i + j) & 0x01) ? -1.0f : 1.0f;
            float h_ij            = sign_correction * Dh[id].x;
            float x_ij            = sign_correction * Dx[id].x;
            float z_ij            = sign_correction * Dz[id].x;

            displacement(i, j) = Vec4f(x_ij, h_ij, z_ij, 0);
        }
    }

    template<typename TDataType>
    void OceanPatch<TDataType>::animate(float t)
    {
        t = m_fft_flow_speed * t;

		cuExecute2D(make_uint2(mResolution, mResolution),
			generateSpectrumKernel,
			m_h0, 
			m_ht, 
			mSpectrumWidth, 
			mResolution, 
			mResolution, 
			t, 
			m_realPatchSize);
        
		cuExecute2D(make_uint2(mResolution, mResolution),
			generateDispalcementKernel,
			m_ht, 
			m_Dxt, 
			m_Dzt, 
			mResolution, 
			mResolution, 
			m_realPatchSize);

         //generateSpectrumKernel<<<grid, block>>>(m_h0, m_ht, mSpectrumWidth, mResolution, mResolution, t, m_realPatchSize);
        // generateDispalcementKernel<<<grid, block>>>(m_ht, m_Dxt, m_Dzt, mResolution, mResolution, m_realPatchSize);

        cufftExecC2C(fftPlan, (float2*)m_ht.begin(), (float2*)m_ht.begin(), CUFFT_INVERSE);
        cufftExecC2C(fftPlan, (float2*)m_Dxt.begin(), (float2*)m_Dxt.begin(), CUFFT_INVERSE);
        cufftExecC2C(fftPlan, (float2*)m_Dzt.begin(), (float2*)m_Dzt.begin(), CUFFT_INVERSE);
        
		cuExecute2D(make_uint2(mResolution, mResolution),
			O_UpdateDisplacement,
			m_displacement,
			m_ht,
			m_Dxt,
			m_Dzt,
			mResolution);
            
        //O_UpdateDisplacement<<<blocksPerGrid, threadsPerBlock>>>(m_displacement, m_ht, m_Dxt, m_Dzt, mResolution);
    }

    template <typename Coord>
    __global__ void O_UpdateTopology(
        DArray2D<Coord> displacement,
        DArray2D<Vec4f> dis,
        float choppiness)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < displacement.nx() && j < displacement.ny())
        {
            int id = displacement.index(i, j);

            Vec4f Dij = dis[id];

            Coord v;
            v.x = choppiness * Dij.x;
            v.y = Dij.y;
            v.z = choppiness * Dij.z;

            displacement(i, j) = v;
        }
    }

    template<typename TDataType>
    void OceanPatch<TDataType>::updateTopology()
    {
        auto topo = TypeInfo::cast<HeightField<TDataType>>(this->stateTopology()->getDataPtr());

        auto& shifts = topo->getDisplacement();

        cuExecute2D(make_uint2(shifts.nx(), shifts.ny()),
            O_UpdateTopology,
            shifts,
            m_displacement,
            mChoppiness);
    }

    template<typename TDataType>
    float OceanPatch<TDataType>::getMaxChoppiness()
    {
        return m_maxChoppiness;
    }

    template<typename TDataType>
    float OceanPatch<TDataType>::getChoppiness()
    {
        return mChoppiness;
    }

    template<typename TDataType>
    void OceanPatch<TDataType>::generateH0(Coord* h0)
    {
        for (unsigned int y = 0; y <= mResolution; y++)
        {
            for (unsigned int x = 0; x <= mResolution; x++)
            {
                float kx = (-( int )mResolution / 2.0f + x) * (2.0f * CUDART_PI_F / m_realPatchSize);
                float ky = (-( int )mResolution / 2.0f + y) * (2.0f * CUDART_PI_F / m_realPatchSize);

                float P = sqrtf(phillips(kx, ky, windDir, m_windSpeed, A, dirDepend));

                if (kx == 0.0f && ky == 0.0f)
                {
                    P = 0.0f;
                }

                float Er = gauss();
                float Ei = gauss();

                float h0_re = Er * P * CUDART_SQRT_HALF_F;
                float h0_im = Ei * P * CUDART_SQRT_HALF_F;

                int i   = y * mSpectrumWidth + x;
                h0[i].x = h0_re;
                h0[i].y = h0_im;
            }
        }
    }

    template<typename TDataType>
    float OceanPatch<TDataType>::phillips(float Kx, float Ky, float Vdir, float V, float A, float dir_depend)
    {
        float k_squared = Kx * Kx + Ky * Ky;

        if (k_squared == 0.0f)
        {
            return 0.0f;
        }

        // largest possible wave from constant wind of velocity v
        float L = V * V / g;

        float k_x     = Kx / sqrtf(k_squared);
        float k_y     = Ky / sqrtf(k_squared);
        float w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

        float phillips = A * expf(-1.0f / (k_squared * L * L)) / (k_squared * k_squared) * w_dot_k * w_dot_k;

        // filter out waves moving opposite to wind
        if (w_dot_k < 0.0f)
        {
            phillips *= dir_depend;
        }

        return phillips;
    }

    template<typename TDataType>
    float OceanPatch<TDataType>::gauss()
    {
        float u1 = rand() / ( float )RAND_MAX;
        float u2 = rand() / ( float )RAND_MAX;

        if (u1 < 1e-6f)
        {
            u1 = 1e-6f;
        }

        return sqrtf(-2 * logf(u1)) * cosf(2 * CUDART_PI_F * u2);
    }

    DEFINE_CLASS(OceanPatch);
}  