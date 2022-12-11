#include "VkFFT.h"
#include "VkSystem.h"
#include "VkContext.h"

#include "VkFFT_Utils.h"

namespace dyno
{
	VkFFT::VkFFT()
	{
	}

	VkFFT::~VkFFT()
	{
		deleteVkFFT(&app);
	}

	bool VkFFT::createContext()
	{
		VkFFTResult resFFT = VKFFT_SUCCESS;

		vkGPU.enableValidationLayers = 0;
		vkGPU.device_id = 0;

		VkResult res = VK_SUCCESS;
		//create instance - a connection between the application and the Vulkan library 
		vkGPU.instance = VkSystem::instance()->instanceHandle();
		// 		res = createInstance(vkGPU, sample_id);
		// 		if (res != 0) {
		// 			//printf("Instance creation failed, error code: %" PRIu64 "\n", res);
		// 			return VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE;
		// 		}
				//set up the debugging messenger 
		res = setupDebugMessenger(&vkGPU);
		if (res != 0) {
			//printf("Debug messenger creation failed, error code: %" PRIu64 "\n", res);
			return false;
		}
		//check if there are GPUs that support Vulkan and select one
		vkGPU.physicalDevice = VkSystem::instance()->getPhysicalDevice();
		// 		res = findPhysicalDevice(vkGPU);
		// 		if (res != 0) {
		// 			//printf("Physical device not found, error code: %" PRIu64 "\n", res);
		// 			return VKFFT_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE;
		// 		}
				//create logical device representation
		res = getComputeQueueFamilyIndex(&vkGPU);
		vkGPU.device = VkSystem::instance()->currentContext()->deviceHandle();
		vkGetDeviceQueue(vkGPU.device, (uint32_t)vkGPU.queueFamilyIndex, 0, &vkGPU.queue);
		// 		res = createDevice(vkGPU, sample_id);
		// 		if (res != 0) {
		// 			//printf("Device creation failed, error code: %" PRIu64 "\n", res);
		// 			return VKFFT_ERROR_FAILED_TO_CREATE_DEVICE;
		// 		}
				//create fence for synchronization 
		res = createFence(&vkGPU);
		if (res != 0) {
			//printf("Fence creation failed, error code: %" PRIu64 "\n", res);
			return false;
		}
		//create a place, command buffer memory is allocated from
		res = createCommandPool(&vkGPU);
		if (res != 0) {
			//printf("Fence creation failed, error code: %" PRIu64 "\n", res);
			return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_POOL;
		}
		vkGetPhysicalDeviceProperties(vkGPU.physicalDevice, &vkGPU.physicalDeviceProperties);
		vkGetPhysicalDeviceMemoryProperties(vkGPU.physicalDevice, &vkGPU.physicalDeviceMemoryProperties);

		glslang_initialize_process();//compiler can be initialized before VkFFT

		return true;
	}

	bool VkFFT::createPipeline(VkDeviceArray2D<dyno::Vec2f>& array2d)
	{
		bool isCompilerInitialized = true;

		VkFFTResult resFFT = VKFFT_SUCCESS;
		VkResult res = VK_SUCCESS;

		const int num_benchmark_samples = 2;
		const int num_runs = 3;
		uint64_t benchmark_dimensions[num_benchmark_samples][4] = { {256, 256, 1, 2}, {64, 64, 1, 2} };
		double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples


		//FFT + iFFT sample code.
		//Setting up FFT configuration for forward and inverse FFT.
		configuration.FFTdim = 2; //FFT dimension, 1D, 2D or 3D (default 1).
		configuration.size[0] = array2d.nx(); //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
		configuration.size[1] = array2d.ny();
		configuration.size[2] = 1;

		//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
		configuration.device = &vkGPU.device;
		configuration.queue = &vkGPU.queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
		configuration.fence = &vkGPU.fence;
		configuration.commandPool = &vkGPU.commandPool;
		configuration.physicalDevice = &vkGPU.physicalDevice;
		configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization

		//Allocate buffer for the input data.
		VkBuffer buffer = array2d.bufferHandle();
		configuration.buffer = &buffer;

		uint64_t bufferSize = array2d.bufferSize();
		configuration.bufferSize = &bufferSize;
		if (resFFT != VKFFT_SUCCESS) return false;
		//free(buffer_input);

		//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
		resFFT = initializeVkFFT(&app, configuration);
		if (resFFT != VKFFT_SUCCESS) return false;

		return true;
	}

	bool VkFFT::update(VkFFT_Type type)
	{
		VkFFTResult resFFT = VKFFT_SUCCESS;
		//Submit FFT+iFFT.
		int tag = type == VkFFT_INVERSE ? -1 : 1;
		VkFFTLaunchParams launchParams = {};
		resFFT = performVulkanFFT(&vkGPU, &app, &launchParams, tag, 1);
		if (resFFT != VKFFT_SUCCESS) return false;


		return true;
	}

	VkFFT* VkFFT::createInstance(VkDeviceArray2D<dyno::Vec2f>& array2d)
	{
		bool resFFT = VKFFT_SUCCESS;
		VkFFT* fft = new VkFFT;
		resFFT = fft->createContext();
		if (resFFT != true)
		{
			delete fft;
			return nullptr;
		}
			
		resFFT = fft->createPipeline(array2d);
		if (resFFT != true)
		{
			delete fft;
			return nullptr;
		}

		return fft;
	}

}