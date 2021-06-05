#include "VtkVisualModule.h"

#include <vtkActor.h>
#include <vtkMapper.h>
#include <vtkProperty.h>

using namespace dyno;

VtkVisualModule::VtkVisualModule()
{
	this->setName("GLVisualModule");
}

VtkVisualModule::~VtkVisualModule()
{
	if (m_actor != NULL)
	{
		m_actor->Delete();
		m_actor = NULL;
	}

	if (m_mapper != NULL)
	{
		m_mapper->Delete();
		m_mapper = NULL;
	}
}

void VtkVisualModule::setColor(float r, float g, float b, float a )
{
	m_actor->GetProperty()->SetColor(r, g, b);
}

vtkActor* VtkVisualModule::getActor()
{
	return m_actor;
}