#pragma once
#include "VkUniform.h"
#include "VkDeviceArray.h"
#include "VkHostArray.h"
#include "VkConstant.h"
#include "Array/ArrayList.h"

#include "Node.h"
#include "Topology/TriangleSet.h"

using namespace dyno;

namespace dyno
{
	class VkProgram;

	/*!
	*	\class	Cloth
	*	\brief	Cloth dynamics.
	*
	*	This class implements a simple cloth.
	*
	*/
	class Cloth : public Node
	{
	public:
		Cloth(std::string name = "default");
		virtual ~Cloth();

		// Setup and fill the compute shader storage buffers containing the particles
		void loadObjFile(std::string filename);

		DEF_INSTANCE_STATE(TriangleSet3f, Topology, "");

	protected:
		void updateStates() override;
		void resetStates() override;

	private:
		void setupKernels();
	};
}