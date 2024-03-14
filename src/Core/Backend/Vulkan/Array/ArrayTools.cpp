#include "ArrayTools.h"

namespace dyno
{
    VkListAllocator::VkListAllocator():mKernel(std::make_shared<VkProgram>(CONSTANT(Push))) {
		mKernel->load(VkSystem::instance()->getAssetPath() / "shaders/glsl/topology/ListAllocate.comp.spv");
    }
    VkListAllocator::~VkListAllocator() {
    }

} // namespace dyno