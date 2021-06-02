#pragma once

#include <Framework/ModuleVisual.h>

class vtkActor;
namespace dyno
{
	class VtkVisualModule : public VisualModule
	{	
	public:
		VtkVisualModule();
		virtual vtkActor* createActor() = 0;		
	};
};