#include "UniformGridRenderer.h"

#include "VkTransfer.h"
#include "Node.h"
#include "Shape.h"

namespace dyno
{
	UniformGridRenderer::UniformGridRenderer()
		: VkGraphicsPipeline()
	{
	};

	UniformGridRenderer::~UniformGridRenderer()
	{
	};

	std::vector<px::Box> hBox;
	bool UniformGridRenderer::initializeImpl()
	{
		auto uGrid = std::dynamic_pointer_cast<UniformGrid3D>(this->inTopology()->getDataPtr());
		if (uGrid == nullptr)
			return false;

		dyno::Vec3f uo = uGrid->orgin();

		px::Box b;
		b.halfLength = dyno::Vec3f(0.5*uGrid->spacing());
		for (int k = 0; k < uGrid->nz(); k++)
		{
			for (int j = 0; j < uGrid->ny(); j++)
			{
				for (int i = 0; i < uGrid->nx(); i++)
				{
					b.center = dyno::Vec3f(uo.x, uo.y, uo.z) + uGrid->spacing() * dyno::Vec3f(i, j, k);
					hBox.push_back(b);
				}
			}
		}

		mBoxes.resize(hBox.size());
		vkTransfer(mBoxes, hBox);
//		mBoxes.resize(uGrid->totalGridSize());
// 		auto& boxes = eleSet->getBoxes();
// 		auto& spheres = eleSet->getSpheres();
// 		auto& capsules = eleSet->getCapsules();
// 
 		initBoxes(mBoxes);

		// Initialize mechanical states
		this->addKernel(
			"SetupBox",
			std::make_shared<VkProgram>(
				BUFFER(px::Box),			//boxes for rendering
				BUFFER3D(float),		//density
				UNIFORM(GridInfo),		//grid info
				CONSTANT(uint32_t))			//number of total grid cells
		);
		kernel("SetupBox")->load(getAssetPath() + "shaders/glsl/phasefield/SetupBox.comp.spv");

		return true;
	}

	void UniformGridRenderer::updateGraphicsContext()
	{
		auto uGrid = std::dynamic_pointer_cast<UniformGrid3D>(this->inTopology()->getDataPtr());
		assert(uGrid != nullptr);

		VkUniform<GridInfo> gridInfo;
		VkConstant<uint32_t> gridSize;
		gridInfo.setValue(uGrid->getGridInfo());
		gridSize.setValue(uGrid->totalGridSize());
		kernel("SetupBox")->flush(
			vkDispatchSize3D(uGrid->nx(), uGrid->ny(), uGrid->nz(), 8),
			&mBoxes,
			mDensity,
			&gridInfo,
			&gridSize);

// 		auto& boxes = eleSet->getBoxes();
		if (mBoxes.size() > 0) {
			vkTransfer(mCubeInstanceData, mBoxes);
		}
	}

	void UniformGridRenderer::initBoxes(VkDeviceArray<px::Box>& boxes)
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
}