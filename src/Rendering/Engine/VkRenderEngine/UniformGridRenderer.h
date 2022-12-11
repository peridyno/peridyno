#pragma once 
#include "VkGraphicsPipeline.h"
#include "VkProgram.h"
#include "VkDeviceArray3D.h"
#include "Topology/UniformGrid.h"

using namespace dyno;

namespace dyno
{
	class UniformGridRenderer : public VkGraphicsPipeline
	{
	public:
		UniformGridRenderer();

		~UniformGridRenderer() override;

		DEF_INSTANCE_IN(UniformGrid3D, Topology, "");

	protected:
		bool initializeImpl() override;
		void updateGraphicsContext() override;

	public:
		VkDeviceArray3D<float>* mDensity = nullptr;

	private:
		void initBoxes(VkDeviceArray<px::Box>& boxex);

		VkDeviceArray<px::Box> mBoxes;
	};
}

