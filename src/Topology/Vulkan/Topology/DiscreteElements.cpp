#include "DiscreteElements.h"
#include "VkTransfer.h"

namespace dyno
{
	DiscreteElements::DiscreteElements()
		: TopologyModule()
	{
	}

	DiscreteElements::~DiscreteElements()
	{
	}

	void DiscreteElements::setBoxes(std::vector<px::Box> boxes)
	{
		if (boxes.size() == 0)
			return;

		mBoxes.resize(boxes.size());
		vkTransfer(mBoxes, boxes);
	}

	void DiscreteElements::setSpheres(std::vector<px::Sphere> spheres)
	{
		if (spheres.size() == 0)
			return;

		mSpheres.resize(spheres.size());
		vkTransfer(mSpheres, spheres);
	}

	void DiscreteElements::setCapsules(std::vector<px::Capsule> capsules)
	{
		if (capsules.size() == 0)
			return;

		mCapsules.resize(capsules.size());
		vkTransfer(mCapsules, capsules);
	}

	ElementOffset DiscreteElements::getElementOffset()
	{
		ElementOffset offset;
		offset.box_bound = mBoxes.size();
		offset.capsule_bound = offset.box_bound + mCapsules.size();
		offset.sphere_bound = offset.capsule_bound + mSpheres.size();

		return offset;
	}

	uint32_t DiscreteElements::getTotalElementSize()
	{
		return mBoxes.size() + mSpheres.size() + mCapsules.size();
	}

}