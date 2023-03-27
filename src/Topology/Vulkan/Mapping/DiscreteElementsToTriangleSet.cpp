#include "DiscreteElementsToTriangleSet.h"

#include "VkTransfer.h"

namespace dyno
{
	IMPLEMENT_CLASS(DiscreteElementsToTriangleSet);

	DiscreteElementsToTriangleSet::DiscreteElementsToTriangleSet()
		: dyno::TopologyMapping()
	{
		this->addKernel(
			"SetupFacets",
			std::make_shared<VkProgram>(
				BUFFER(Vec3f),			//out: vertices
				BUFFER(TopologyModule::Triangle),			//out: indices
				BUFFER(px::Box),			//in: boxes
				BUFFER(px::Capsule),		//in: capsules
				BUFFER(px::Sphere),			//in: spheres
				UNIFORM(ElementOffset),	//in: offset
				CONSTANT(uint32_t))			//in: number of boxes
		);
		kernel("SetupFacets")->load(getAssetPath() + "shaders/glsl/topology/SetupFacets.comp.spv");
	}

	bool DiscreteElementsToTriangleSet::apply()
	{
		auto elements = this->inDiscreteElements()->getDataPtr();

		if (this->outTriangleSet()->isEmpty()) {
			this->outTriangleSet()->allocate();
		}

		auto outTopo = this->outTriangleSet()->getDataPtr();

		auto& boxes = elements->getBoxes();
		auto& spheres = elements->getSpheres();
		auto& capsules = elements->getCapsules();
		
		uint32_t totalSize = boxes.size() * 36 + capsules.size() * 48 + spheres.size() * 24;

		auto& vertices = outTopo->mPoints;
		auto& indices = outTopo->mTriangleIndex;

		vertices.resize(totalSize);
		indices.resize(totalSize / 3);

		uint32_t eleSize = elements->getTotalElementSize();

		VkUniform<ElementOffset> offset;
		offset.setValue(elements->getElementOffset());

		VkConstant<uint32_t> constUint;
		constUint.setValue(eleSize);

		kernel("SetupFacets")->flush(
			vkDispatchSize(eleSize, 64),
			vertices.handle(),
			indices.handle(),
			&elements->getBoxes(),
			&elements->getCapsules(),
			&elements->getSpheres(),
			&offset,
			&constUint);

		outTopo->update();

		return true;
	}
}