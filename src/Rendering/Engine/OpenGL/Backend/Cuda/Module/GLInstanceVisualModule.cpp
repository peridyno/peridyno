#include "GLInstanceVisualModule.h"

namespace dyno {

	IMPLEMENT_CLASS(GLInstanceVisualModule)

	GLInstanceVisualModule::GLInstanceVisualModule()
	{
		this->setName("instance_renderer");
		this->inInstanceTransform()->tagOptional(false);
	}

	std::string GLInstanceVisualModule::caption()
	{
		return "Instance Visual Module";
	}

}