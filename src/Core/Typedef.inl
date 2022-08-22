#include <list>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <algorithm>

#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <iostream>

#ifdef PRECISION_FLOAT
typedef float Real;
#else
typedef double Real;
#endif

namespace dyno {

	using uint = unsigned int;

	using uint64 = unsigned long long;
	using int64 = signed long long;

#define INVALID -1
#define M_PI 3.14159265358979323846
#define M_E 2.71828182845904523536

	constexpr Real EPSILON = std::numeric_limits<Real>::epsilon();
	constexpr Real REAL_MAX = (std::numeric_limits<Real>::max)();
	constexpr Real REAL_MIN = (std::numeric_limits<Real>::min)();
	constexpr uint BLOCK_SIZE = 64;

	static uint iDivUp(uint a, uint b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	// compute grid and thread block size for a given number of elements
	static uint cudaGridSize(uint totalSize, uint blockSize)
	{
		int dim = iDivUp(totalSize, blockSize);
		return dim == 0 ? 1 : dim;
	}

	static uint3 cudaGridSize2D(uint2 totalSize, uint blockSize)
	{
		uint3 gridDims;
		gridDims.x = iDivUp(totalSize.x, blockSize);
		gridDims.y = iDivUp(totalSize.y, blockSize);

		gridDims.x = gridDims.x == 0 ? 1 : gridDims.x;
		gridDims.y = gridDims.y == 0 ? 1 : gridDims.y;
		gridDims.z = 1;

		return gridDims;
	}

	static uint3 cudaGridSize3D(uint3 totalSize, uint blockSize)
	{
		uint3 gridDims;
		gridDims.x = iDivUp(totalSize.x, blockSize);
		gridDims.y = iDivUp(totalSize.y, blockSize);
		gridDims.z = iDivUp(totalSize.z, blockSize);

		gridDims.x = gridDims.x == 0 ? 1 : gridDims.x;
		gridDims.y = gridDims.y == 0 ? 1 : gridDims.y;
		gridDims.z = gridDims.z == 0 ? 1 : gridDims.z;

		return gridDims;
	}

	static uint3 cudaGridSize3D(uint3 totalSize, uint3 blockSize)
	{
		uint3 gridDims;
		gridDims.x = iDivUp(totalSize.x, blockSize.x);
		gridDims.y = iDivUp(totalSize.y, blockSize.y);
		gridDims.z = iDivUp(totalSize.z, blockSize.z);

		gridDims.x = gridDims.x == 0 ? 1 : gridDims.x;
		gridDims.y = gridDims.y == 0 ? 1 : gridDims.y;
		gridDims.z = gridDims.z == 0 ? 1 : gridDims.z;

		return gridDims;
	}

	/** check whether cuda thinks there was an error and fail with msg, if this is the case
	* @ingroup tools
	*/
	static inline void checkCudaError(const char *msg) {
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err) {
			printf( "CUDA error: %d : %s at %s:%d \n", err, cudaGetErrorString(err), __FILE__, __LINE__);
 			throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
		}
	}

	// use this macro to make sure no error occurs when cuda functions are called
#ifdef NDEBUG
#define cuSafeCall(X)  X
#else
#define cuSafeCall(X) X; dyno::checkCudaError(#X);
#endif

/**
 * @brief Macro to check cuda errors
 *
 */
#ifdef NDEBUG
#define cuSynchronize() {}
#else
#define cuSynchronize()	{						\
		char str[200];							\
		cudaDeviceSynchronize();				\
		cudaError_t err = cudaGetLastError();	\
		if (err != cudaSuccess)					\
		{										\
			sprintf_s(str, "CUDA error: %d : %s at %s:%d \n", err, cudaGetErrorString(err), __FILE__, __LINE__);		\
			throw std::runtime_error(std::string(str));																\
		}																											\
	}
#endif

 /**
  * @brief Macro definition for execuation of cuda kernels, note that at lease one block will be executed.
  *
  * size: indicate how many threads are required in total.
  * Func: kernel function
  */
#define cuExecute(size, Func, ...){						\
		uint pDims = cudaGridSize((uint)size, BLOCK_SIZE);	\
		Func << <pDims, BLOCK_SIZE >> > (				\
		__VA_ARGS__);									\
		cuSynchronize();								\
	}

#define cuExecute2D(size, Func, ...){						\
		uint3 pDims = cudaGridSize2D(size, 8);				\
		dim3 threadsPerBlock(8, 8, 1);		\
		Func << <pDims, threadsPerBlock >> > (				\
		__VA_ARGS__);										\
		cuSynchronize();									\
	}

#define cuExecute3D(size, Func, ...){						\
		dim3 pDims = cudaGridSize3D(size, 8);		\
		dim3 threadsPerBlock(8, 8, 8);		\
		Func << <pDims, threadsPerBlock >> > (				\
		__VA_ARGS__);										\
		cuSynchronize();									\
	}


	class Bool
	{
	public:
		DYN_FUNC Bool(bool v = false) { val = v ? 1 : 0; }

		DYN_FUNC inline bool operator! () const {
			return 1 - val ? true : false;
		}

		DYN_FUNC inline bool operator== (bool v) const {
			uint tmpV = v ? 1 : 0;
			return val == tmpV;
		}

		DYN_FUNC inline bool operator== (const Bool& v) const {
			return val == v.val;
		}

		DYN_FUNC inline Bool& operator= (const bool v) {
			val = v ? 1 : 0;
			return *this;
		}

		DYN_FUNC inline Bool& operator= (const Bool& v) {
			val = v.val;
			return *this;
		}

		DYN_FUNC inline Bool& operator&= (const bool v) {
			val &= (v ? 1 : 0);
			return *this;
		}

		DYN_FUNC inline Bool& operator|= (const bool v) {
			val |= (v ? 1 : 0);
			return *this;
		}

		DYN_FUNC inline Bool operator& (const bool v) const {
			Bool ret;
			ret.val = val & (v ? 1 : 0);
			return ret;
		}

		DYN_FUNC inline Bool operator| (const bool v) const {
			Bool ret;
			ret.val = val | (v ? 1 : 0);
			return ret;
		}

		DYN_FUNC inline Bool& operator&= (const Bool& v) {
			val &= v.val;
			return *this;
		}

		DYN_FUNC inline Bool& operator|= (const Bool& v) {
			val |= v.val;
			return *this;
		}

		DYN_FUNC inline Bool operator& (const Bool& v) const {
			Bool ret;
			ret.val = val & v.val;
			return ret;
		}

		DYN_FUNC inline Bool operator| (const Bool& v) const {
			Bool ret;
			ret.val = val | v.val;
			return ret;
		}

		DYN_FUNC inline bool operator&& (const Bool& v) const {
			return val & v.val;
		}

		DYN_FUNC inline bool operator|| (const Bool& v) const {
			return val | v.val;
		}

		DYN_FUNC inline bool operator&& (const bool& v) const {
			uint tmpV = v ? 1 : 0;
			return val & tmpV;
		}

		DYN_FUNC inline bool operator|| (const bool& v) const {
			uint tmpV = v ? 1 : 0;
			return val | tmpV;
		}

		DYN_FUNC inline operator bool() const {
			return val == 1;
		}


	private:
		uint val = 0;
	};
}// end of namespace dyno


namespace TypeInfo
{
	template<class T, class ... Args>
	std::shared_ptr<T> New(Args&& ... args) { std::shared_ptr<T> p(new T(std::forward<Args>(args)...)); return p; }

	template<class TA, class TB>
	inline TA* cast(TB* b)
	{
		TA* ptr = dynamic_cast<TA*>(b);
		return ptr;
	}

	template<class TA, class TB>
	inline std::shared_ptr<TA> cast(std::shared_ptr<TB> b)
	{
		std::shared_ptr<TA> ptr = std::dynamic_pointer_cast<TA>(b);
		return ptr;
	}
}


