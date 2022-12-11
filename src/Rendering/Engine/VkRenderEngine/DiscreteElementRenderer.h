#pragma once 
#include "VkGraphicsPipeline.h"
#include "VkProgram.h"
#include "Topology/DiscreteElements.h"

using namespace dyno;

namespace dyno
{
	class DiscreteElementRenderer : public VkGraphicsPipeline
	{
	public:
		DiscreteElementRenderer();

		~DiscreteElementRenderer() override;

		DEF_INSTANCE_IN(DiscreteElements, Topology, "");

	protected:
		bool initializeImpl() override;
		void updateGraphicsContext() override;

	private:
		void initBoxes(VkDeviceArray<px::Box>& boxex);
		void initSpheres(VkDeviceArray<px::Sphere>& spheres);
		void initCapsules(VkDeviceArray<px::Capsule>& capsules);

		std::shared_ptr<VkProgram> setupFacets;
		VkConstant<uint32_t> mBoxNumber;

		VkUniform<ElementOffset> mOffset;
	};
}

