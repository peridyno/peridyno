#pragma once
#include "VkDeviceArray.h"
#include "Primitive/Primitive3D.h"
#include "Module/TopologyModule.h"

namespace dyno
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

		void setBoxes(std::vector<px::Box> boxes);
		void setSpheres(std::vector<px::Sphere> spheres);
		void setCapsules(std::vector<px::Capsule> capsules);

		VkDeviceArray<px::Box>& getBoxes() { return mBoxes; }
		VkDeviceArray<px::Sphere>& getSpheres() { return mSpheres; }
		VkDeviceArray<px::Capsule>& getCapsules() { return mCapsules; }

		ElementOffset getElementOffset();

		uint32_t getTotalElementSize();

	private:
		VkDeviceArray<px::Box> mBoxes;
		VkDeviceArray<px::Capsule> mCapsules;
		VkDeviceArray<px::Sphere> mSpheres;
	};
}

