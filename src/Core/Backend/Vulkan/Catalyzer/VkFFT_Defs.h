#ifndef VKFFT_DEFS_H
#define VKFFT_DEFS_H
#include <vulkan/vulkan.h>
typedef struct {
#if(VKFFT_BACKEND==0)
	VkInstance instance;//a connection between the application and the Vulkan library 
	VkPhysicalDevice physicalDevice;//a handle for the graphics card used in the application
	VkPhysicalDeviceProperties physicalDeviceProperties;//bastic device properties
	VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;//bastic memory properties of the device
	VkDevice device;//a logical device, interacting with physical device
	VkDebugUtilsMessengerEXT debugMessenger;//extension for debugging
	uint64_t queueFamilyIndex;//if multiple queues are available, specify the used one
	VkQueue queue;//a place, where all operations are submitted
	VkCommandPool commandPool;//an opaque objects that command buffer memory is allocated from
	VkFence fence;//a vkGPU->fence used to synchronize dispatches
	std::vector<const char*> enabledDeviceExtensions;
	uint64_t enableValidationLayers;
#elif(VKFFT_BACKEND==1)
	CUdevice device;
	CUcontext context;
#elif(VKFFT_BACKEND==2)
	hipDevice_t device;
	hipCtx_t context;
#elif(VKFFT_BACKEND==3)
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue commandQueue;
#endif
	uint64_t device_id;//an id of a device, reported by Vulkan device list
} VkGPU;//an example structure containing Vulkan primitives

typedef struct {
	//WHDCN layout

	//required parameters:
	uint64_t FFTdim; //FFT dimensionality (1, 2 or 3)
	uint64_t size[3]; // WHD -system dimensions

#if(VKFFT_BACKEND==0)
	VkPhysicalDevice* physicalDevice;//pointer to Vulkan physical device, obtained from vkEnumeratePhysicalDevices
	VkDevice* device;//pointer to Vulkan device, created with vkCreateDevice
	VkQueue* queue;//pointer to Vulkan queue, created with vkGetDeviceQueue
	VkCommandPool* commandPool;//pointer to Vulkan command pool, created with vkCreateCommandPool
	VkFence* fence;//pointer to Vulkan fence, created with vkCreateFence
	uint64_t isCompilerInitialized;//specify if glslang compiler has been intialized before (0 - off, 1 - on). Default 0
#elif(VKFFT_BACKEND==1)
	CUdevice* device;//pointer to CUDA device, obtained from cuDeviceGet
	//CUcontext* context;//pointer to CUDA context, obtained from cuDeviceGet
	cudaStream_t* stream;//pointer to streams (can be more than 1), where to execute the kernels
	uint64_t num_streams;//try to submit CUDA kernels in multiple streams for asynchronous execution. Default 1
#elif(VKFFT_BACKEND==2)
	hipDevice_t* device;//pointer to HIP device, obtained from hipDeviceGet
	//hipCtx_t* context;//pointer to HIP context, obtained from hipDeviceGet
	hipStream_t* stream;//pointer to streams (can be more than 1), where to execute the kernels
	uint64_t num_streams;//try to submit HIP kernels in multiple streams for asynchronous execution. Default 1
#elif(VKFFT_BACKEND==3)
	cl_platform_id* platform;
	cl_device_id* device;
	cl_context* context;
#endif

	//data parameters:
	uint64_t userTempBuffer; //buffer allocated by app automatically if needed to reorder Four step algorithm. Setting to non zero value enables manual user allocation (0 - off, 1 - on)

	uint64_t bufferNum;//multiple buffer sequence storage is Vulkan only. Default 1
	uint64_t tempBufferNum;//multiple buffer sequence storage is Vulkan only. Default 1, buffer allocated by app automatically if needed to reorder Four step algorithm. Setting to non zero value enables manual user allocation
	uint64_t inputBufferNum;//multiple buffer sequence storage is Vulkan only. Default 1, if isInputFormatted is enabled
	uint64_t outputBufferNum;//multiple buffer sequence storage is Vulkan only. Default 1, if isOutputFormatted is enabled
	uint64_t kernelNum;//multiple buffer sequence storage is Vulkan only. Default 1, if performConvolution is enabled

	//sizes are obligatory in Vulkan backend, optional in others
	uint64_t* bufferSize;//array of buffers sizes in bytes
	uint64_t* tempBufferSize;//array of temp buffers sizes in bytes. Default set to bufferSize sum, buffer allocated by app automatically if needed to reorder Four step algorithm. Setting to non zero value enables manual user allocation
	uint64_t* inputBufferSize;//array of input buffers sizes in bytes, if isInputFormatted is enabled
	uint64_t* outputBufferSize;//array of output buffers sizes in bytes, if isOutputFormatted is enabled
	uint64_t* kernelSize;//array of kernel buffers sizes in bytes, if performConvolution is enabled

#if(VKFFT_BACKEND==0)
	VkBuffer* buffer;//pointer to array of buffers (or one buffer) used for computations
	VkBuffer* tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same sum size or bigger as buffer (can be split in multiple). Default 0. Setting to non zero value enables manual user allocation
	VkBuffer* inputBuffer;//pointer to array of input buffers (or one buffer) used to read data from if isInputFormatted is enabled
	VkBuffer* outputBuffer;//pointer to array of output buffers (or one buffer) used for write data to if isOutputFormatted is enabled
	VkBuffer* kernel;//pointer to array of kernel buffers (or one buffer) used for read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==1)
	void** buffer;//pointer to device buffer used for computations
	void** tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
	void** inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
	void** outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
	void** kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==2)
	void** buffer;//pointer to device buffer used for computations
	void** tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
	void** inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
	void** outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
	void** kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==3)
	cl_mem* buffer;//pointer to device buffer used for computations
	cl_mem* tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
	cl_mem* inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
	cl_mem* outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
	cl_mem* kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#endif
	uint64_t bufferOffset;//specify if VkFFT has to offset the first element position inside the buffer. In bytes. Default 0 
	uint64_t tempBufferOffset;//specify if VkFFT has to offset the first element position inside the temp buffer. In bytes. Default 0 
	uint64_t inputBufferOffset;//specify if VkFFT has to offset the first element position inside the input buffer. In bytes. Default 0 
	uint64_t outputBufferOffset;//specify if VkFFT has to offset the first element position inside the output buffer. In bytes. Default 0
	uint64_t kernelOffset;//specify if VkFFT has to offset the first element position inside the kernel. In bytes. Default 0

	//optional: (default 0 if not stated otherwise)
	uint64_t coalescedMemory;//in bytes, for Nvidia and AMD is equal to 32, Intel is equal 64, scaled for half precision. Gonna work regardles, but if specified by user correctly, the performance will be higher.
	uint64_t aimThreads;//aim at this many threads per block. Default 128
	uint64_t numSharedBanks;//how many banks shared memory has. Default 32
	uint64_t inverseReturnToInputBuffer;//return data to the input buffer in inverse transform (0 - off, 1 - on). isInputFormatted must be enabled
	uint64_t numberBatches;// N - used to perform multiple batches of initial data. Default 1
	uint64_t useUint64;//use 64-bit addressing mode in generated kernels
	uint64_t omitDimension[3];//disable FFT for this dimension (0 - FFT enabled, 1 - FFT disabled). Default 0. Doesn't work for R2C dimension 0 for now. Doesn't work with convolutions.
	uint64_t fixMaxRadixBluestein;//controls the padding of sequences in Bluestein convolution. If specified, padded sequence will be made of up to fixMaxRadixBluestein primes. Default: 2 for CUDA and Vulkan/OpenCL/HIP up to 1048576 combined dimension FFT system, 7 for Vulkan/OpenCL/HIP past after. Min = 2, Max = 13.
	uint64_t performBandwidthBoost;//try to reduce coalsesced number by a factor of X to get bigger sequence in one upload for strided axes. Default: -1 for DCT, 2 for Bluestein's algorithm (or -1 if DCT), 0 otherwise 

	uint64_t doublePrecision; //perform calculations in double precision (0 - off, 1 - on).
	uint64_t halfPrecision; //perform calculations in half precision (0 - off, 1 - on)
	uint64_t halfPrecisionMemoryOnly; //use half precision only as input/output buffer. Input/Output have to be allocated as half, buffer/tempBuffer have to be allocated as float (out of place mode only). Specify isInputFormatted and isOutputFormatted to use (0 - off, 1 - on)
	uint64_t doublePrecisionFloatMemory; //use FP64 precision for all calculations, while all memory storage is done in FP32.

	uint64_t performR2C; //perform R2C/C2R decomposition (0 - off, 1 - on)
	uint64_t performDCT; //perform DCT transformation (X - DCT type, 1-4)
	uint64_t disableMergeSequencesR2C; //disable merging of two real sequences to reduce calculations (0 - off, 1 - on)
	uint64_t normalize; //normalize inverse transform (0 - off, 1 - on)
	uint64_t disableReorderFourStep; // disables unshuffling of Four step algorithm. Requires tempbuffer allocation (0 - off, 1 - on)
	uint64_t useLUT; //switches from calculating sincos to using precomputed LUT tables (0 - off, 1 - on). Configured by initialization routine
	uint64_t makeForwardPlanOnly; //generate code only for forward FFT (0 - off, 1 - on)
	uint64_t makeInversePlanOnly; //generate code only for inverse FFT (0 - off, 1 - on)

	uint64_t bufferStride[3];//buffer strides - default set to x - x*y - x*y*z values
	uint64_t isInputFormatted; //specify if input buffer is padded - 0 - padded, 1 - not padded. For example if it is not padded for R2C if out-of-place mode is selected (only if numberBatches==1 and numberKernels==1)
	uint64_t isOutputFormatted; //specify if output buffer is padded - 0 - padded, 1 - not padded. For example if it is not padded for R2C if out-of-place mode is selected (only if numberBatches==1 and numberKernels==1)
	uint64_t inputBufferStride[3];//input buffer strides. Used if isInputFormatted is enabled. Default set to bufferStride values
	uint64_t outputBufferStride[3];//output buffer strides. Used if isInputFormatted is enabled. Default set to bufferStride values

	uint64_t considerAllAxesStrided;//will create plan for nonstrided axis similar as a strided axis - used with disableReorderFourStep to get the same layout for Bluestein kernel (0 - off, 1 - on)
	uint64_t keepShaderCode;//will keep shader code and print all executed shaders during the plan execution in order (0 - off, 1 - on)
	uint64_t printMemoryLayout;//will print order of buffers used in shaders (0 - off, 1 - on)

	//optional zero padding control parameters: (default 0 if not stated otherwise)
	uint64_t performZeropadding[3]; // don't read some data/perform computations if some input sequences are zeropadded for each axis (0 - off, 1 - on)
	uint64_t fft_zeropad_left[3];//specify start boundary of zero block in the system for each axis
	uint64_t fft_zeropad_right[3];//specify end boundary of zero block in the system for each axis
	uint64_t frequencyZeroPadding; //set to 1 if zeropadding of frequency domain, default 0 - spatial zeropadding

	//optional convolution control parameters: (default 0 if not stated otherwise)
	uint64_t performConvolution; //perform convolution in this application (0 - off, 1 - on). Disables reorderFourStep parameter
	uint64_t conjugateConvolution;//0 off, 1 - conjugation of the sequence FFT is currently done on, 2 - conjugation of the convolution kernel
	uint64_t crossPowerSpectrumNormalization;//normalize the FFT x kernel multiplication in frequency domain
	uint64_t coordinateFeatures; // C - coordinate, or dimension of features vector. In matrix convolution - size of vector
	uint64_t matrixConvolution; //if equal to 2 perform 2x2, if equal to 3 perform 3x3 matrix-vector convolution. Overrides coordinateFeatures
	uint64_t symmetricKernel; //specify if kernel in 2x2 or 3x3 matrix convolution is symmetric
	uint64_t numberKernels;// N - only used in convolution step - specify how many kernels were initialized before. Expands one input to multiple (batched) output
	uint64_t kernelConvolution;// specify if this application is used to create kernel for convolution, so it has the same properties. performConvolution has to be set to 0 for kernel creation

	//register overutilization (experimental): (default 0 if not stated otherwise)
	uint64_t registerBoost; //specify if register file size is bigger than shared memory and can be used to extend it X times (on Nvidia 256KB register file can be used instead of 32KB of shared memory, set this constant to 4 to emulate 128KB of shared memory). Default 1
	uint64_t registerBoostNonPow2; //specify if register overutilization should be used on non power of 2 sequences (0 - off, 1 - on)
	uint64_t registerBoost4Step; //specify if register file overutilization should be used in big sequences (>2^14), same definition as registerBoost. Default 1

	//not used techniques:
	uint64_t swapTo3Stage4Step; //specify at which power of 2 to switch from 2 upload to 3 upload 4-step FFT, in case if making max sequence size lower than coalesced sequence helps to combat TLB misses. Default 0 - disabled. Must be at least 17
	uint64_t devicePageSize;//in KB, the size of a page on the GPU. Setting to 0 disables local buffer split in pages
	uint64_t localPageSize;//in KB, the size to split page into if sequence spans multiple devicePageSize pages

	//automatically filled based on device info (still can be reconfigured by user):
	uint64_t maxComputeWorkGroupCount[3]; // maxComputeWorkGroupCount from VkPhysicalDeviceLimits
	uint64_t maxComputeWorkGroupSize[3]; // maxComputeWorkGroupCount from VkPhysicalDeviceLimits
	uint64_t maxThreadsNum; //max number of threads from VkPhysicalDeviceLimits
	uint64_t sharedMemorySizeStatic; //available for static allocation shared memory size, in bytes
	uint64_t sharedMemorySize; //available for allocation shared memory size, in bytes
	uint64_t sharedMemorySizePow2; //power of 2 which is less or equal to sharedMemorySize, in bytes
	uint64_t warpSize; //number of threads per warp/wavefront.
	uint64_t halfThreads;//Intel fix
	uint64_t allocateTempBuffer; //buffer allocated by app automatically if needed to reorder Four step algorithm. Parameter to check if it has been allocated
	uint64_t reorderFourStep; // unshuffle Four step algorithm. Requires tempbuffer allocation (0 - off, 1 - on). Default 1.
	int64_t maxCodeLength; //specify how big can be buffer used for code generation (in char). Default 1000000 chars.
	int64_t maxTempLength; //specify how big can be buffer used for intermediate string sprintfs be (in char). Default 5000 chars. If code segfaults for some reason - try increasing this number.
#if(VKFFT_BACKEND==0)
	VkDeviceMemory tempBufferDeviceMemory;//Filled at app creation
	VkCommandBuffer* commandBuffer;//Filled at app execution
	VkMemoryBarrier* memory_barrier;//Filled at app creation
#elif(VKFFT_BACKEND==1)
	cudaEvent_t* stream_event;//Filled at app creation
	uint64_t streamCounter;//Filled at app creation
	uint64_t streamID;//Filled at app creation
#elif(VKFFT_BACKEND==2)
	hipEvent_t* stream_event;//Filled at app creation
	uint64_t streamCounter;//Filled at app creation
	uint64_t streamID;//Filled at app creation
#elif(VKFFT_BACKEND==3)
	cl_command_queue* commandQueue;
#endif
} VkFFTConfiguration;//parameters specified at plan creation

typedef struct {
#if(VKFFT_BACKEND==0)
	VkCommandBuffer* commandBuffer;//commandBuffer to which FFT is appended

	VkBuffer* buffer;//pointer to array of buffers (or one buffer) used for computations
	VkBuffer* tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same sum size or bigger as buffer (can be split in multiple). Default 0. Setting to non zero value enables manual user allocation
	VkBuffer* inputBuffer;//pointer to array of input buffers (or one buffer) used to read data from if isInputFormatted is enabled
	VkBuffer* outputBuffer;//pointer to array of output buffers (or one buffer) used for write data to if isOutputFormatted is enabled
	VkBuffer* kernel;//pointer to array of kernel buffers (or one buffer) used for read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==1)
	void** buffer;//pointer to device buffer used for computations
	void** tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
	void** inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
	void** outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
	void** kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==2)
	void** buffer;//pointer to device buffer used for computations
	void** tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
	void** inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
	void** outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
	void** kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#elif(VKFFT_BACKEND==3)
	cl_command_queue* commandQueue;//commandBuffer to which FFT is appended

	cl_mem* buffer;//pointer to device buffer used for computations
	cl_mem* tempBuffer;//needed if reorderFourStep is enabled to transpose the array. Same size as buffer. Default 0. Setting to non zero value enables manual user allocation
	cl_mem* inputBuffer;//pointer to device buffer used to read data from if isInputFormatted is enabled
	cl_mem* outputBuffer;//pointer to device buffer used to read data from if isOutputFormatted is enabled
	cl_mem* kernel;//pointer to device buffer used to read kernel data from if performConvolution is enabled
#endif
} VkFFTLaunchParams;//parameters specified at plan execution
typedef enum VkFFTResult {
	VKFFT_SUCCESS = 0,
	VKFFT_ERROR_MALLOC_FAILED = 1,
	VKFFT_ERROR_INSUFFICIENT_CODE_BUFFER = 2,
	VKFFT_ERROR_INSUFFICIENT_TEMP_BUFFER = 3,
	VKFFT_ERROR_PLAN_NOT_INITIALIZED = 4,
	VKFFT_ERROR_NULL_TEMP_PASSED = 5,
	VKFFT_ERROR_INVALID_PHYSICAL_DEVICE = 1001,
	VKFFT_ERROR_INVALID_DEVICE = 1002,
	VKFFT_ERROR_INVALID_QUEUE = 1003,
	VKFFT_ERROR_INVALID_COMMAND_POOL = 1004,
	VKFFT_ERROR_INVALID_FENCE = 1005,
	VKFFT_ERROR_ONLY_FORWARD_FFT_INITIALIZED = 1006,
	VKFFT_ERROR_ONLY_INVERSE_FFT_INITIALIZED = 1007,
	VKFFT_ERROR_INVALID_CONTEXT = 1008,
	VKFFT_ERROR_INVALID_PLATFORM = 1009,
	VKFFT_ERROR_EMPTY_FFTdim = 2001,
	VKFFT_ERROR_EMPTY_size = 2002,
	VKFFT_ERROR_EMPTY_bufferSize = 2003,
	VKFFT_ERROR_EMPTY_buffer = 2004,
	VKFFT_ERROR_EMPTY_tempBufferSize = 2005,
	VKFFT_ERROR_EMPTY_tempBuffer = 2006,
	VKFFT_ERROR_EMPTY_inputBufferSize = 2007,
	VKFFT_ERROR_EMPTY_inputBuffer = 2008,
	VKFFT_ERROR_EMPTY_outputBufferSize = 2009,
	VKFFT_ERROR_EMPTY_outputBuffer = 2010,
	VKFFT_ERROR_EMPTY_kernelSize = 2011,
	VKFFT_ERROR_EMPTY_kernel = 2012,
	VKFFT_ERROR_UNSUPPORTED_RADIX = 3001,
	VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH = 3002,
	VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH_R2C = 3003,
	VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH_DCT = 3004,
	VKFFT_ERROR_UNSUPPORTED_FFT_OMIT = 3005,
	VKFFT_ERROR_FAILED_TO_ALLOCATE = 4001,
	VKFFT_ERROR_FAILED_TO_MAP_MEMORY = 4002,
	VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS = 4003,
	VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER = 4004,
	VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER = 4005,
	VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE = 4006,
	VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES = 4007,
	VKFFT_ERROR_FAILED_TO_RESET_FENCES = 4008,
	VKFFT_ERROR_FAILED_TO_CREATE_DESCRIPTOR_POOL = 4009,
	VKFFT_ERROR_FAILED_TO_CREATE_DESCRIPTOR_SET_LAYOUT = 4010,
	VKFFT_ERROR_FAILED_TO_ALLOCATE_DESCRIPTOR_SETS = 4011,
	VKFFT_ERROR_FAILED_TO_CREATE_PIPELINE_LAYOUT = 4012,
	VKFFT_ERROR_FAILED_SHADER_PREPROCESS = 4013,
	VKFFT_ERROR_FAILED_SHADER_PARSE = 4014,
	VKFFT_ERROR_FAILED_SHADER_LINK = 4015,
	VKFFT_ERROR_FAILED_SPIRV_GENERATE = 4016,
	VKFFT_ERROR_FAILED_TO_CREATE_SHADER_MODULE = 4017,
	VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE = 4018,
	VKFFT_ERROR_FAILED_TO_SETUP_DEBUG_MESSENGER = 4019,
	VKFFT_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE = 4020,
	VKFFT_ERROR_FAILED_TO_CREATE_DEVICE = 4021,
	VKFFT_ERROR_FAILED_TO_CREATE_FENCE = 4022,
	VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_POOL = 4023,
	VKFFT_ERROR_FAILED_TO_CREATE_BUFFER = 4024,
	VKFFT_ERROR_FAILED_TO_ALLOCATE_MEMORY = 4025,
	VKFFT_ERROR_FAILED_TO_BIND_BUFFER_MEMORY = 4026,
	VKFFT_ERROR_FAILED_TO_FIND_MEMORY = 4027,
	VKFFT_ERROR_FAILED_TO_SYNCHRONIZE = 4028,
	VKFFT_ERROR_FAILED_TO_COPY = 4029,
	VKFFT_ERROR_FAILED_TO_CREATE_PROGRAM = 4030,
	VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM = 4031,
	VKFFT_ERROR_FAILED_TO_GET_CODE_SIZE = 4032,
	VKFFT_ERROR_FAILED_TO_GET_CODE = 4033,
	VKFFT_ERROR_FAILED_TO_DESTROY_PROGRAM = 4034,
	VKFFT_ERROR_FAILED_TO_LOAD_MODULE = 4035,
	VKFFT_ERROR_FAILED_TO_GET_FUNCTION = 4036,
	VKFFT_ERROR_FAILED_TO_SET_DYNAMIC_SHARED_MEMORY = 4037,
	VKFFT_ERROR_FAILED_TO_MODULE_GET_GLOBAL = 4038,
	VKFFT_ERROR_FAILED_TO_LAUNCH_KERNEL = 4039,
	VKFFT_ERROR_FAILED_TO_EVENT_RECORD = 4040,
	VKFFT_ERROR_FAILED_TO_ADD_NAME_EXPRESSION = 4041,
	VKFFT_ERROR_FAILED_TO_INITIALIZE = 4042,
	VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID = 4043,
	VKFFT_ERROR_FAILED_TO_GET_DEVICE = 4044,
	VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT = 4045,
	VKFFT_ERROR_FAILED_TO_CREATE_PIPELINE = 4046,
	VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG = 4047,
	VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE = 4048,
	VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE = 4049,
	VKFFT_ERROR_FAILED_TO_ENUMERATE_DEVICES = 4050,
	VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE = 4051,
	VKFFT_ERROR_FAILED_TO_CREATE_EVENT = 4052
} VkFFTResult;
typedef struct {
	uint64_t size[3];
	uint64_t localSize[3];
	uint64_t sourceFFTSize;
	uint64_t fftDim;
	uint64_t inverse;
	uint64_t actualInverse;
	uint64_t inverseBluestein;
	uint64_t zeropad[2];
	uint64_t zeropadBluestein[2];
	uint64_t axis_id;
	uint64_t axis_upload_id;
	uint64_t numAxisUploads;
	uint64_t registers_per_thread;
	uint64_t registers_per_thread_per_radix[14];
	uint64_t min_registers_per_thread;
	uint64_t readToRegisters;
	uint64_t writeFromRegisters;
	uint64_t LUT;
	uint64_t useBluesteinFFT;
	uint64_t reverseBluesteinMultiUpload;
	uint64_t BluesteinConvolutionStep;
	uint64_t BluesteinPreMultiplication;
	uint64_t BluesteinPostMultiplication;
	uint64_t startDCT3LUT;
	uint64_t startDCT4LUT;
	uint64_t performR2C;
	uint64_t performR2CmultiUpload;
	uint64_t performDCT;
	uint64_t performBandwidthBoost;
	uint64_t frequencyZeropadding;
	uint64_t performZeropaddingFull[3]; // don't do read/write if full sequence is omitted
	uint64_t performZeropaddingInput[3]; // don't read if input is zeropadded (0 - off, 1 - on)
	uint64_t performZeropaddingOutput[3]; // don't write if output is zeropadded (0 - off, 1 - on)
	uint64_t fft_zeropad_left_full[3];
	uint64_t fft_zeropad_left_read[3];
	uint64_t fft_zeropad_left_write[3];
	uint64_t fft_zeropad_right_full[3];
	uint64_t fft_zeropad_right_read[3];
	uint64_t fft_zeropad_right_write[3];
	uint64_t fft_zeropad_Bluestein_left_read[3];
	uint64_t fft_zeropad_Bluestein_left_write[3];
	uint64_t fft_zeropad_Bluestein_right_read[3];
	uint64_t fft_zeropad_Bluestein_right_write[3];
	uint64_t inputStride[5];
	uint64_t outputStride[5];
	uint64_t fft_dim_full;
	uint64_t stageStartSize;
	uint64_t firstStageStartSize;
	uint64_t fft_dim_x;
	uint64_t dispatchZactualFFTSize;
	uint64_t numStages;
	uint64_t stageRadix[20];
	uint64_t inputOffset;
	uint64_t kernelOffset;
	uint64_t outputOffset;
	uint64_t reorderFourStep;
	uint64_t performWorkGroupShift[3];
	uint64_t inputBufferBlockNum;
	uint64_t inputBufferBlockSize;
	uint64_t outputBufferBlockNum;
	uint64_t outputBufferBlockSize;
	uint64_t kernelBlockNum;
	uint64_t kernelBlockSize;
	uint64_t numCoordinates;
	uint64_t matrixConvolution; //if equal to 2 perform 2x2, if equal to 3 perform 3x3 matrix-vector convolution. Overrides coordinateFeatures
	uint64_t numBatches;
	uint64_t numKernels;
	uint64_t conjugateConvolution;
	uint64_t crossPowerSpectrumNormalization;
	uint64_t usedSharedMemory;
	uint64_t sharedMemSize;
	uint64_t sharedMemSizePow2;
	uint64_t normalize;
	uint64_t complexSize;
	uint64_t inputNumberByteSize;
	uint64_t outputNumberByteSize;
	uint64_t kernelNumberByteSize;
	uint64_t maxStageSumLUT;
	uint64_t unroll;
	uint64_t convolutionStep;
	uint64_t symmetricKernel;
	uint64_t supportAxis;
	uint64_t cacheShuffle;
	uint64_t registerBoost;
	uint64_t warpSize;
	uint64_t numSharedBanks;
	uint64_t resolveBankConflictFirstStages;
	uint64_t sharedStrideBankConflictFirstStages;
	uint64_t sharedStrideReadWriteConflict;
	uint64_t maxSharedStride;
	uint64_t axisSwapped;
	uint64_t mergeSequencesR2C;

	uint64_t numBuffersBound[6];
	uint64_t convolutionBindingID;
	uint64_t LUTBindingID;
	uint64_t BluesteinConvolutionBindingID;
	uint64_t BluesteinMultiplicationBindingID;

	uint64_t performBufferSetUpdate;
	uint64_t useUint64;
	char** regIDs;
	char* disableThreadsStart;
	char* disableThreadsEnd;
	char sdataID[50];
	char inoutID[50];
	char combinedID[50];
	char gl_LocalInvocationID_x[50];
	char gl_LocalInvocationID_y[50];
	char gl_LocalInvocationID_z[50];
	char gl_GlobalInvocationID_x[200];
	char gl_GlobalInvocationID_y[200];
	char gl_GlobalInvocationID_z[200];
	char tshuffle[50];
	char sharedStride[50];
	char gl_WorkGroupSize_x[50];
	char gl_WorkGroupSize_y[50];
	char gl_WorkGroupSize_z[50];
	char gl_WorkGroupID_x[50];
	char gl_WorkGroupID_y[50];
	char gl_WorkGroupID_z[50];
	char tempReg[50];
	char stageInvocationID[50];
	char blockInvocationID[50];
	char temp[50];
	char w[50];
	char iw[50];
	char locID[13][40];
	char* code0;
	char* output;
	char* tempStr;
	int64_t tempLen;
	int64_t currentLen;
	int64_t maxCodeLength;
	int64_t maxTempLength;
} VkFFTSpecializationConstantsLayout;
typedef struct {
	uint32_t workGroupShift[3];
} VkFFTPushConstantsLayoutUint32;
typedef struct {
	uint64_t workGroupShift[3];
} VkFFTPushConstantsLayoutUint64;
typedef struct {
	uint64_t numBindings;
	uint64_t axisBlock[4];
	uint64_t groupedBatch;
	VkFFTSpecializationConstantsLayout specializationConstants;
	VkFFTPushConstantsLayoutUint32 pushConstantsUint32;
	VkFFTPushConstantsLayoutUint64 pushConstants;
	uint64_t updatePushConstants;
#if(VKFFT_BACKEND==0)
	VkBuffer* inputBuffer;
	VkBuffer* outputBuffer;
	VkDescriptorPool descriptorPool;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkDeviceMemory bufferLUTDeviceMemory;
	VkBuffer bufferLUT;
	VkDeviceMemory* bufferBluesteinDeviceMemory;
	VkDeviceMemory* bufferBluesteinFFTDeviceMemory;
	VkBuffer* bufferBluestein;
	VkBuffer* bufferBluesteinFFT;
#elif(VKFFT_BACKEND==1)
	void** inputBuffer;
	void** outputBuffer;
	CUmodule VkFFTModule;
	CUfunction VkFFTKernel;
	void* bufferLUT;
	CUdeviceptr consts_addr;
	void** bufferBluestein;
	void** bufferBluesteinFFT;
#elif(VKFFT_BACKEND==2)
	void** inputBuffer;
	void** outputBuffer;
	hipModule_t VkFFTModule;
	hipFunction_t VkFFTKernel;
	void* bufferLUT;
	hipDeviceptr_t consts_addr;
	void** bufferBluestein;
	void** bufferBluesteinFFT;
#elif(VKFFT_BACKEND==3)
	cl_mem* inputBuffer;
	cl_mem* outputBuffer;
	cl_program  program;
	cl_kernel kernel;
	cl_mem bufferLUT;
	cl_mem* bufferBluestein;
	cl_mem* bufferBluesteinFFT;
#endif
	uint64_t bufferLUTSize;
	uint64_t referenceLUT;
} VkFFTAxis;

typedef struct {
	uint64_t actualFFTSizePerAxis[3][3];
	uint64_t numAxisUploads[3];
	uint64_t axisSplit[3][4];
	VkFFTAxis axes[3][4];

	uint64_t multiUploadR2C;
	uint64_t actualPerformR2CPerAxis[3]; // automatically specified, shows if R2C is actually performed or inside FFT or as a separate step
	VkFFTAxis R2Cdecomposition;
	VkFFTAxis inverseBluesteinAxes[3][4];
} VkFFTPlan;
typedef struct {
	VkFFTConfiguration configuration;
	VkFFTPlan* localFFTPlan;
	VkFFTPlan* localFFTPlan_inverse; //additional inverse plan

	uint64_t actualNumBatches;
	uint64_t firstAxis;
	uint64_t lastAxis;
	//Bluestein buffers reused among plans
	uint64_t useBluesteinFFT[3];
#if(VKFFT_BACKEND==0)
	VkDeviceMemory bufferBluesteinDeviceMemory[3];
	VkDeviceMemory bufferBluesteinFFTDeviceMemory[3];
	VkDeviceMemory bufferBluesteinIFFTDeviceMemory[3];
	VkBuffer bufferBluestein[3];
	VkBuffer bufferBluesteinFFT[3];
	VkBuffer bufferBluesteinIFFT[3];
#elif(VKFFT_BACKEND==1)
	void* bufferBluestein[3];
	void* bufferBluesteinFFT[3];
	void* bufferBluesteinIFFT[3];
#elif(VKFFT_BACKEND==2)
	void* bufferBluestein[3];
	void* bufferBluesteinFFT[3];
	void* bufferBluesteinIFFT[3];
#elif(VKFFT_BACKEND==3)
	cl_mem bufferBluestein[3];
	cl_mem bufferBluesteinFFT[3];
	cl_mem bufferBluesteinIFFT[3];
#endif
	uint64_t bufferBluesteinSize[3];
} VkFFTApplication;

#endif