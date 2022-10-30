#pragma once
#include "VkDeviceArray.h"
#include "Primitive/Primitive3D.h"
#include "Module/TopologyModule.h"

namespace px
{
	struct ElementOffset
	{
		uint32_t box_bound;
		uint32_t capsule_bound;
		uint32_t sphere_bound;
	};

	class DiscreteElements : public dyno::TopologyModule
	{
	public:
		DiscreteElements();
		~DiscreteElements() override;

		void setBoxes(std::vector<Box> boxes);
		void setSpheres(std::vector<Sphere> spheres);
		void setCapsules(std::vector<Capsule> capsules);

		VkDeviceArray<Box>& getBoxes() { return mBoxes; }
		VkDeviceArray<Sphere>& getSpheres() { return mSpheres; }
		VkDeviceArray<Capsule>& getCapsules() { return mCapsules; }

		ElementOffset getElementOffset();

		uint32_t getTotalElementSize();

	private:
		VkDeviceArray<Box> mBoxes;
		VkDeviceArray<Capsule> mCapsules;
		VkDeviceArray<Sphere> mSpheres;
	};
}

