#include "VkSystem.h"
#include "VkProgram.h"

#include "Array/Array.h"

using namespace dyno;

/**
 * This example demonstrates how to use the wrapped api for vulkan to ease the programming 
 */

int main(int, char**)
{
	VkSystem::instance()->setAssetPath(getAssetPath());
	VkSystem::instance()->initialize(false);

	//Initialize all buffers
	uint num = 100;

	DArray<float> dA(num);
	DArray<float> dB(num);
	DArray<float> dC(num);

	CArray<float> hA(num);
	CArray<float> hB(num);
	CArray<float> hC(num);

	for (int i = 0; i < num; i++)
	{
		hA[i] = float(i);
		hB[i] = float(i);
	}

	dA.assign(hA);
	dB.assign(hB);

	//Declare a kernel
	auto kernel = std::make_shared<VkProgram>(
		BUFFER(float),		//Array A
		BUFFER(float),		//Array B
		BUFFER(float),		//Array C
		CONSTANT(uint));
	kernel->load(getAssetPath() + "shaders/glsl/tutorials/VecAdd.comp.spv");

	//Execuate the kernel
	VkConstant<uint> N(num);
	kernel->flush(
		vkDispatchSize(num, 128),
		dA.handle(),
		dB.handle(),
		dC.handle(),
		&N);

	//Copy results back to the host and print out
	hC.assign(dC);
	for (int i = 0; i < num; i++)
	{
		printf("%f \n", hC[i]);
	}

	return 0;
}
