/**
 * Copyright 2021~2024 Yue Chang, Shusen Liu
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "Module/ComputeModule.h"

#include "Kernel.h"

namespace dyno
{

#define cuIntegralAdh(size, type, scale, Func,...){					\
		uint pDims = cudaGridSize((uint)size, BLOCK_SIZE);				\
		if (type == ParticleApproximation<TDataType>::KT_Smooth)											\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return SmoothKernel<Real>::integral(r, h, s);	\
			};																\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, ParticleApproximation<TDataType>::mScalingFactor);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Spiky)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return SpikyKernel<Real>::integral(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, ParticleApproximation<TDataType>::mScalingFactor);	\
		}																\
		cuSynchronize();												\
	}

#define cuIntegral(size, type, scale, Func,...){					\
		uint pDims = cudaGridSize((uint)size, BLOCK_SIZE);				\
		if (type == ParticleApproximation<TDataType>::KT_Smooth)		\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return SmoothKernel<Real>::integral(r, h, s);	\
			};																\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, ParticleApproximation<TDataType>::mScalingFactor);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Spiky)	\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return SpikyKernel<Real>::integral(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, ParticleApproximation<TDataType>::mScalingFactor);	\
		}																\
		cuSynchronize();												\
	}

#define cuZerothOrder(size, type, scale, Func,...){					\
		uint pDims = cudaGridSize((uint)size, BLOCK_SIZE);				\
		if (type == ParticleApproximation<TDataType>::KT_Smooth)											\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return SmoothKernel<Real>::weight(r, h, s);	\
			};																\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Spiky)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return SpikyKernel<Real>::weight(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Cubic)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return CubicKernel<Real>::weight(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Constant)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return ConstantKernel<Real>::weight(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Quartic)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return QuarticKernel<Real>::weight(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Corrected)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return CorrectedKernel<Real>::weight(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_CorrectedQuatic)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return CorrectedQuaticKernel<Real>::weight(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_WendlandC2)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return WendlandC2Kernel<Real>::weight(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_CorrectedMPSKernel)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return CorrectedMPSKernel<Real>::weight(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		cuSynchronize();												\
	}

#define cuFirstOrder(size, type, scale, Func,...){					\
		uint pDims = cudaGridSize((uint)size, BLOCK_SIZE);				\
		if (type == ParticleApproximation<TDataType>::KT_Smooth)											\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return SmoothKernel<Real>::gradient(r, h, s);	\
			};																\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Spiky)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return SpikyKernel<Real>::gradient(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Cubic)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return CubicKernel<Real>::gradient(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Constant)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return ConstantKernel<Real>::gradient(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Quartic)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return QuarticKernel<Real>::gradient(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Corrected)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return CorrectedKernel<Real>::gradient(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_CorrectedQuatic)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return CorrectedQuaticKernel<Real>::gradient(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_WendlandC2)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return WendlandC2Kernel<Real>::gradient(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_CorrectedMPSKernel)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return CorrectedMPSKernel<Real>::gradient(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		cuSynchronize();												\
	}

#define cuSecondOrder(size, type, scale, Func,...){					\
		uint pDims = cudaGridSize((uint)size, BLOCK_SIZE);				\
		if (type == ParticleApproximation<TDataType>::KT_Smooth)											\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return ConstantKernel<Real>::weightRR(r, h, s);	\
			};																\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Spiky)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return ConstantKernel<Real>::weightRR(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Cubic)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return ConstantKernel<Real>::weightRR(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Constant)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return ConstantKernel<Real>::weightRR(r, h, s);				\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Quartic)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return ConstantKernel<Real>::weightRR(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_Corrected)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return CorrectedKernel<Real>::weightRR(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_CorrectedQuatic)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return CorrectedQuaticKernel<Real>::weightRR(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_WendlandC2)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return ConstantKernel<Real>::weightRR(r, h, s);						\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		else if (type == ParticleApproximation<TDataType>::KT_CorrectedMPSKernel)										\
		{																\
			auto lambdaFunc = [=] __device__(Real r, Real h, Real s) -> Real {		\
				return CorrectedMPSKernel<Real>::weightRR(r, h, s);					\
			};															\
			Func << <pDims, BLOCK_SIZE >> > (__VA_ARGS__, lambdaFunc, scale);	\
		}																\
		cuSynchronize();												\
	}




	template<typename TDataType>
	class ParticleApproximation : public ComputeModule
	{
		DECLARE_TCLASS(ParticleApproximation, TDataType)
	public:
		typedef typename TDataType::Real Real;

		ParticleApproximation();
		virtual ~ParticleApproximation();

		DECLARE_ENUM(EKernelType,
			KT_Smooth = 0,
			KT_Spiky = 1,
			KT_Cubic = 2,
			KT_Constant = 3,
			KT_Quartic = 4,
			KT_Corrected = 5,
			KT_CorrectedQuatic = 6,
			KT_WendlandC2 = 7,
			KT_CorrectedMPSKernel = 8);

		void compute() override {};

	public:
		DEF_VAR_IN(Real, SmoothingLength, "Smoothing Length");
		DEF_VAR_IN(Real, SamplingDistance, "Particle sampling distance");

		DEF_ENUM(EKernelType, KernelType, EKernelType::KT_Spiky, "Rendering mode");

	protected:
		Real mScalingFactor = Real(1);

	private:
		void calculateScalingFactor();
	};
}