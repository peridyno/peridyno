/**
 * Copyright 2017-2021 Jian SHI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <Module/VisualModule.h>

#include <vtkTimeStamp.h>

class vtkActor;
class vtkVolume;

namespace dyno
{
	class VtkVisualModule : public VisualModule
	{	
	public:
		VtkVisualModule();
		virtual ~VtkVisualModule();

		void setColor(float r, float g, float b, float a = 1.f);

		vtkActor*	getActor();
		vtkVolume*	getVolume();


		virtual void display() final;
		virtual void updateRenderingContext() final;
		
		bool isDirty(bool update = true);

	protected:

		vtkActor*	m_actor = NULL;
		vtkVolume*  m_volume = NULL;

		// timestamp for data sync
		vtkTimeStamp m_sceneTime;
		vtkTimeStamp m_updateTime;
	};
};