#include "GhostParticles.h"

#include "GLPointVisualModule.h"

namespace dyno
{
	//IMPLEMENT_TCLASS(GhostParticles, TDataType)

	template<typename TDataType>
	GhostParticles<TDataType>::GhostParticles()
		: ParticleSystem<TDataType>()
	{
		auto ghostRender = std::make_shared<GLPointVisualModule>();
		ghostRender->setColor(Color(1, 0.5, 0));
		ghostRender->setColorMapMode(GLPointVisualModule::PER_OBJECT_SHADER);

		this->statePointSet()->connect(ghostRender->inPointSet());

		this->graphicsPipeline()->pushModule(ghostRender);
	}

	template<typename TDataType>
	GhostParticles<TDataType>::~GhostParticles()
	{
	}

	template<typename TDataType>
	void GhostParticles<TDataType>::resetStates()
	{
		ParticleSystem<TDataType>::resetStates();
	}

	DEFINE_CLASS(GhostParticles);
}