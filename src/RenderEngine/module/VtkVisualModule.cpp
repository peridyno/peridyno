#include "VtkVisualModule.h"

#include <vtkActor.h>
#include <vtkVolume.h>
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

	if (m_volume != NULL)
	{
		m_volume->Delete();
		m_volume = NULL;
	}

}

void VtkVisualModule::setColor(float r, float g, float b, float a )
{
	if (m_actor != NULL)
		m_actor->GetProperty()->SetColor(r, g, b);
}

vtkActor* VtkVisualModule::getActor()
{
	return m_actor;
}

vtkVolume* VtkVisualModule::getVolume()
{
	return m_volume;
}