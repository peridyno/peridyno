#pragma once
#include "OBJexporter.h"


namespace dyno
{
	IMPLEMENT_TCLASS(OBJExporter, TDataType)
	template <typename TDataType> class TriangleSet;

	template<typename TDataType>
	OBJExporter<TDataType>::OBJExporter() 
	{	
		ExportModule = std::make_shared<TriangleMeshWriter<TDataType>>();
		this->animationPipeline()->pushModule(ExportModule);
		
		this->stateFrameNumber()->connect(ExportModule->inFrameNumber());
		this->inTriangleSet()->connect(ExportModule->inTopology());

		ExportModule->varOutputPath()->setValue(this->varOutputPath()->getData());
		ExportModule->varOutputType()->setValue(this->varOutputType()->getData());

		ExportModule->varStart()->setValue(this->varStartFrame()->getData());
		ExportModule->varEnd()->setValue(this->varEndFrame()->getData());
		
		
	}

	template<typename TDataType>
	void OBJExporter<TDataType>::setFrameStep()
	{
		unsigned s = this->varFrameStep()->getData();
		ExportModule->varFrameStep()->setValue(s);
	}

	template<typename TDataType>
	void OBJExporter<TDataType>::preUpdateStates()
	{
		this->setFrameStep();
		ExportModule->varOutputPath()->setValue(this->varOutputPath()->getData());
		ExportModule->varOutputType()->setValue(this->varOutputType()->getData());

		ExportModule->varStart()->setValue(this->varStartFrame()->getData());
		ExportModule->varEnd()->setValue(this->varEndFrame()->getData());
	}

	//reset
	template<typename TDataType>
	void OBJExporter<TDataType>::resetStates() 
	{



	}


	DEFINE_CLASS(OBJExporter)
}