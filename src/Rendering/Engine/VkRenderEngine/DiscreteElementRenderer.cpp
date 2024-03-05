#include "DiscreteElementRenderer.h"

#include "VkTransfer.h"
#include "Node.h"
#include "Topology/Shape.h"

namespace dyno
{
	DiscreteElementRenderer::DiscreteElementRenderer()
		: VkGraphicsPipeline()
	{
	};

	DiscreteElementRenderer::~DiscreteElementRenderer()
	{
	};

	bool DiscreteElementRenderer::initializeImpl()
	{
		Node* pNode = getParentNode();
		assert(pNode != nullptr);

		auto eleSet = std::dynamic_pointer_cast<DiscreteElements>(this->inTopology()->getDataPtr());
		if (eleSet == nullptr)
			return false;

		auto& boxes = eleSet->getBoxes();
		auto& spheres = eleSet->getSpheres();
		auto& capsules = eleSet->getCapsules();

		initBoxes(boxes);
		initSpheres(spheres);
		initCapsules(capsules);

		return true;
	}

	void DiscreteElementRenderer::updateGraphicsContext()
	{
		auto eleSet = std::dynamic_pointer_cast<DiscreteElements>(this->inTopology()->getDataPtr());
		assert(eleSet != nullptr);

		auto& boxes = eleSet->getBoxes();
		if (boxes.size() > 0) {
			vkTransfer(mCubeInstanceData, boxes);
		}
		
		auto& spheres = eleSet->getSpheres();
		if (spheres.size() > 0) {
			vkTransfer(mSphereInstanceData, spheres);
		}

		auto& capsules = eleSet->getCapsules();
		if (capsules.size() > 0) {
			vkTransfer(mCapsuleInstanceData, capsules);
		}
	}

	void DiscreteElementRenderer::initBoxes(VkDeviceArray<px::Box>& boxes)
	{
		int vertSize = sizeof(CUBE_VERTICES) / sizeof(Vertex);
		int indexSize = sizeof(CUBE_INDICES) / sizeof(uint32_t);
		std::vector<Vertex> vertices;
		vertices.resize(vertSize);
		memcpy(vertices.data(), CUBE_VERTICES, sizeof(CUBE_VERTICES));
		std::vector<uint32_t> indices;
		indices.resize(indexSize);
		memcpy(indices.data(), CUBE_INDICES, sizeof(CUBE_INDICES));

		mCubeVertex.resize(vertSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		vkTransfer(mCubeVertex, vertices);
		mCubeIndex.resize(indexSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		vkTransfer(mCubeIndex, indices);
		mCubeInstanceData.resize(boxes.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	}

	void DiscreteElementRenderer::initSpheres(VkDeviceArray<px::Sphere>& spheres)
	{
		int vertSize = sizeof(SPHERE_VERTICES) / sizeof(Vertex);
		int indexSize = sizeof(SPHERE_INDICES) / sizeof(uint32_t);
		std::vector<Vertex> vertices;
		vertices.resize(vertSize);
		memcpy(vertices.data(), SPHERE_VERTICES, sizeof(SPHERE_VERTICES));
		std::vector<uint32_t> indices;
		indices.resize(indexSize);
		memcpy(indices.data(), SPHERE_INDICES, sizeof(SPHERE_INDICES));

		mSphereVertex.resize(vertSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		vkTransfer(mSphereVertex, vertices);
		mSphereIndex.resize(indexSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		vkTransfer(mSphereIndex, indices);
		mSphereInstanceData.resize(spheres.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	}

	void DiscreteElementRenderer::initCapsules(VkDeviceArray<px::Capsule>& capsules)
	{
		int vertSize = sizeof(SPHERE_VERTICES) / sizeof(Vertex);
		int indexSize = sizeof(SPHERE_INDICES) / sizeof(uint32_t);
		std::vector<Vertex> vertices;
		vertices.resize(vertSize);
		memcpy(vertices.data(), SPHERE_VERTICES, sizeof(SPHERE_VERTICES));
		std::vector<uint32_t> indices;
		indices.resize(indexSize);
		memcpy(indices.data(), SPHERE_INDICES, sizeof(SPHERE_INDICES));

		mCapsuleVertex.resize(vertSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		vkTransfer(mCapsuleVertex, vertices);
		mCapsuleIndex.resize(indexSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		vkTransfer(mCapsuleIndex, indices);
		mCapsuleInstanceData.resize(capsules.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	}
}