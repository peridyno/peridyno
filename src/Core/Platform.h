#pragma once
#define PERIDYNO_VERSION 0.1.0
#define PERIDYNO_VERSION_MAJOR 0
#define PERIDYNO_VERSION_MINOR 1
#define PERIDYNO_VERSION_PATCH 0


#if ((defined _WIN32) || (defined(__MINGW32__) || defined(__CYGWIN__))) && defined(_DLL)
#if !defined(PERIDYNO_DLL) && !defined(PERIDYNO_STATIC)
#define PERIDYNO_DLL
#endif
#endif

#if ((defined _WIN32) || (defined(__MINGW32__) || defined(__CYGWIN__))) && defined(PERIDYNO_DLL)
#define PERIDYNO_EXPORT __declspec(dllexport)
#define PERIDYNO_IMPORT __declspec(dllimport)
#else
#define PERIDYNO_EXPORT
#define PERIDYNO_IMPORT
#endif

#if defined(PERIDYNO_API_COMPILE)
#define PERIDYNOApi PERIDYNO_EXPORT
#else
#define PERIDYNOAPI PERIDYNO_IMPORT
#endif

#define PERIDYNO_COMPILER_CUDA

#if(defined(PERIDYNO_COMPILER_CUDA))
#include <cuda_runtime.h>
#	define DYN_FUNC __device__ __host__ 
#	define GPU_FUNC __device__ 
#	define CPU_FUNC __host__ 
#else
#	define DYN_FUNC
#	define GPU_FUNC 
#	define CPU_FUNC 
#endif

enum DeviceType
{
	CPU,
	GPU,
	UNDEFINED
};

#define PRECISION_FLOAT

#include "Typedef.h"

//#define SIMULATION2D
