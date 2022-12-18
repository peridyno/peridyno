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

    template<typename Real>
    __device__  Complex<Real> complex_exp(Real arg)
    {
        return Complex<Real>(cosf(arg), sinf(arg));
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
    template <typename Real, typename Complex>
    __global__ void generateSpectrumKernel(DArray2D<Complex> h0,
        DArray2D<Complex> ht,
        unsigned int    in_width,
        unsigned int    out_width,
        unsigned int    out_height,
        Real           t,
        Real           patchSize)
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int in_index = y * in_width + x;
        unsigned int in_mindex = (out_height - y) * in_width + (out_width - x);  // mirrored
        unsigned int out_index = y * out_width + x;

        // calculate wave vector
        Complex k((-(int)out_width / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize), (-(int)out_width / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize));

        // calculate dispersion w(k)
        Real k_len = k.normSquared();
        Real w = sqrtf(9.81f * k_len);

        if ((x < out_width) && (y < out_height))
        {
            Complex h0_k = h0[in_index];
            Complex h0_mk = h0[in_mindex];

            // output frequency-space complex values
            ht[out_index] = h0_k * complex_exp(w * t) + h0_mk.conjugate() * complex_exp(-w * t);
        }
    }

    template <typename Real, typename Complex>
    __global__ void generateDispalcementKernel(
        DArray2D<Complex>      ht,
        DArray2D<Complex>      Dxt,
        DArray2D<Complex>      Dzt,
        unsigned int width,
        unsigned int height,
        Real        patchSize)
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int id = y * width + x;

        // calculate wave vector
        Real kx = (-(int)width / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
        Real ky = (-(int)height / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);
        Real k_squared = kx * kx + ky * ky;
        if (k_squared == 0.0f)
        {
            k_squared = 1.0f;
        }
        kx = kx / sqrtf(k_squared);
        ky = ky / sqrtf(k_squared);

        Complex ht_ij = ht(x, y);
        Complex idoth = Complex(-ht_ij.imagPart(), ht_ij.realPart());

        Dxt(x, y) = kx * idoth;
        Dzt(x, y) = ky * idoth;
    }

    template<typename TDataType>
    OceanPatch<TDataType>::OceanPatch()
        : Node()
    {
        auto heights = std::make_shared<HeightField<TDataType>>();
        this->stateTopology()->setDataPtr(heights);

        std::ifstream input(getAssetPath() + "windparam.txt", std::ios::in);
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

 //       mResolution = size;

        mSpectrumWidth = this->varResolution()->getData() + 1;
        mSpectrumHeight = this->varResolution()->getData() + 4;


//        mRealPatchSize = patchSize;

//        this->varWindType()->setValue(windType);

    }

    template<typename TDataType>
    OceanPatch<TDataType>::~OceanPatch()
    {
        mH0.clear();
        mHt.clear();
        mDxt.clear();
        mDzt.clear();
        //m_gradient.clear();
    }

    template<typename TDataType>
    void OceanPatch<TDataType>::resetWindType()
    {
        int windType = this->varWindType()->getData();
        mWindSpeed = m_params[windType].windSpeed;
        A = m_params[windType].A;
        m_maxChoppiness = m_params[windType].choppiness;
        mChoppiness = m_params[windType].choppiness;
        m_globalShift = m_params[windType].global;
    }

    template<typename TDataType>
    void OceanPatch<TDataType>::resetStates()
    {
		resetWindType();

        uint res = this->varResolution()->getData();

		cufftPlan2d(&fftPlan, res, res, CUFFT_C2C);

		int spectrumSize = mSpectrumWidth * mSpectrumHeight * sizeof(Complex);
		mH0.resize(mSpectrumWidth, mSpectrumHeight);

        Complex* host_h0 = (Complex*)malloc(spectrumSize);
		generateH0(host_h0);

		cuSafeCall(cudaMemcpy(mH0.begin(), host_h0, spectrumSize, cudaMemcpyHostToDevice));

		mHt.resize(res, res);
		mDxt.resize(res, res);
		mDzt.resize(res, res);
		this->stateDisplacement()->resize(res, res);
		//m_gradient.resize(mResolution, mResolution);

		auto topo = TypeInfo::cast<HeightField<TDataType>>(this->stateTopology()->getDataPtr());
		Real h = this->varPatchSize()->getData() / res;
		topo->setExtents(res, res);
		topo->setGridSpacing(h);
		topo->setOrigin(Vec3f(-0.5 * h * topo->width(), 0, -0.5 * h * topo->height()));
    }

    float t = 0.0f;
    template<typename TDataType>
    void OceanPatch<TDataType>::updateStates()
    {
        t += 0.016f;
        this->animate(t);
    }

	template<typename TDataType>
	void OceanPatch<TDataType>::postUpdateStates()
	{
		auto topo = TypeInfo::cast<HeightField<TDataType>>(this->stateTopology()->getDataPtr());

		auto& shifts = topo->getDisplacement();

		uint2 extent;
		extent.x = shifts.nx();
		extent.y = shifts.ny();
		cuExecute2D(extent,
			O_UpdateTopology,
			shifts,
			this->stateDisplacement()->getData(),
			mChoppiness);
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
            float h_ij = sign_correction * Dh[id].realPart();
            float x_ij = sign_correction * Dx[id].realPart();
            float z_ij = sign_correction * Dz[id].realPart();

            displacement(i, j) = Vec4f(x_ij, h_ij, z_ij, 0);
        }
    }

    template<typename TDataType>
    void OceanPatch<TDataType>::animate(float t)
    {
        t = m_fft_flow_speed * t;

        uint res = this->varResolution()->getData();

        cuExecute2D(make_uint2(res, res),
            generateSpectrumKernel,
            mH0,
            mHt,
            mSpectrumWidth,
            res,
            res,
            t,
            this->varPatchSize()->getData());

        cuExecute2D(make_uint2(res, res),
            generateDispalcementKernel,
            mHt,
            mDxt,
            mDzt,
            res,
            res,
            this->varPatchSize()->getData());

        cufftExecC2C(fftPlan, (float2*)mHt.begin(), (float2*)mHt.begin(), CUFFT_INVERSE);
        cufftExecC2C(fftPlan, (float2*)mDxt.begin(), (float2*)mDxt.begin(), CUFFT_INVERSE);
        cufftExecC2C(fftPlan, (float2*)mDzt.begin(), (float2*)mDzt.begin(), CUFFT_INVERSE);


        cuExecute2D(make_uint2(res, res),
            O_UpdateDisplacement,
            this->stateDisplacement()->getData(),
            mHt,
            mDxt,
            mDzt,
            res);
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
            v[0] = choppiness * Dij[0];
            v[1] = Dij[1];
            v[2] = choppiness * Dij[2];

            displacement(i, j) = v;
        }

    }

    template<typename TDataType>
    void OceanPatch<TDataType>::generateH0(Complex* h0)
    {
        Real windDir = this->varWindDirection()->getData();

        uint res = this->varResolution()->getData();
        for (unsigned int y = 0; y <= res; y++)
        {
            for (unsigned int x = 0; x <= res; x++)
            {
                float kx = (-(int)res / 2.0f + x) * (2.0f * CUDART_PI_F / this->varPatchSize()->getData());
                float ky = (-(int)res / 2.0f + y) * (2.0f * CUDART_PI_F / this->varPatchSize()->getData());

                float P = sqrtf(phillips(kx, ky, windDir, mWindSpeed, A, dirDepend));

                if (kx == 0.0f && ky == 0.0f)
                {
                    P = 0.0f;
                }

                float Er = gauss();
                float Ei = gauss();

                float h0_re = Er * P * CUDART_SQRT_HALF_F;
                float h0_im = Ei * P * CUDART_SQRT_HALF_F;

                int i = y * mSpectrumWidth + x;
//                 h0[i].x = h0_re;
//                 h0[i].y = h0_im;
				h0[i] = Complex(h0_re, h0_im);
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
        Real L = V * V / g;

        Real k_x = Kx / sqrtf(k_squared);
        Real k_y = Ky / sqrtf(k_squared);
        Real w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

        Real phillips = A * expf(-1.0f / (k_squared * L * L)) / (k_squared * k_squared) * w_dot_k * w_dot_k;

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
        Real u1 = rand() / (Real)RAND_MAX;
        Real u2 = rand() / (Real)RAND_MAX;

        if (u1 < 1e-6f)
        {
            u1 = 1e-6f;
        }

        return sqrtf(-2 * logf(u1)) * cosf(2 * CUDART_PI_F * u2);
    }

    DEFINE_CLASS(OceanPatch);
}