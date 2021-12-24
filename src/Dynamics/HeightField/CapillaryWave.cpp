#include "CapillaryWave.h"

#include "Topology/HeightField.h"
#include "CapillaryWaveModule.h"
namespace dyno
{
	IMPLEMENT_CLASS_1(CapillaryWave, TDataType)

	template<typename TDataType>
	CapillaryWave<TDataType>::CapillaryWave(std::string name)
		: Node()
	{
		auto capillaryWaveModule = std::make_shared<CapillaryWaveModule<TDataType>>();
		this->statePosition()->connect(capillaryWaveModule->statePosition());
		this->animationPipeline()->pushModule(capillaryWaveModule);
	
		auto heights = std::make_shared<HeightField<TDataType>>();
		this->currentTopology()->setDataPtr(heights);
	}

	template<typename TDataType>
	CapillaryWave<TDataType>::~CapillaryWave()
	{
		
	}


	template<typename TDataType>
	void CapillaryWave<TDataType>::updateTopology()
	{

		auto heights = TypeInfo::cast<HeightField<TDataType>>(this->currentTopology()->getDataPtr());

		heights->getHeights().assign(this->statePosition()->getData());
	}


	template<typename TDataType>
	void CapillaryWave<TDataType>::resetStates()
	{
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::updateStates()
	{

	}

	DEFINE_CLASS(CapillaryWave);
}