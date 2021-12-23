#include "CapillaryWave.h"

#include "Topology/HeightField.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(CapillaryWave, TDataType)

	template<typename TDataType>
	CapillaryWave<TDataType>::CapillaryWave(std::string name)
		: Node()
	{
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