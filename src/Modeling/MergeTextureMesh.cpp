#include "MergeTextureMesh.h"
#include "GLPhotorealisticRender.h"

namespace dyno 
{
	template<typename TDataType>
	MergeTextureMesh<TDataType>::MergeTextureMesh()
	{
		this->stateTextureMesh()->setDataPtr(std::make_shared<TextureMesh>());
		
		auto render = std::make_shared<GLPhotorealisticRender>();
		this->stateTextureMesh()->connect(render->inTextureMesh());
		this->stateTextureMesh()->promoteOuput();

		this->graphicsPipeline()->pushModule(render);
	}

	template<typename TDataType>
	MergeTextureMesh<TDataType>::~MergeTextureMesh()
	{
	}

	template<typename TDataType>
	void MergeTextureMesh<TDataType>::resetStates()
	{
		auto texMesh01 = this->inFirst()->constDataPtr();
		auto texMesh02 = this->inSecond()->constDataPtr();

		auto out = this->stateTextureMesh()->getDataPtr();
		
		out->merge(texMesh01,texMesh02);
	}

	DEFINE_CLASS(MergeTextureMesh);
}