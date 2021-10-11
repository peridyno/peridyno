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

#include <Rendering.h>

#include <vtkExternalOpenGLRenderer.h>
#include <vtkExternalOpenGLRenderWindow.h>
#include <vtkExternalOpenGLCamera.h>
#include <vtkExternalLight.h>

#include <vtkActor.h>
#include <vtkCubeSource.h>
#include <vtkPlaneSource.h>

#include <vtkSequencePass.h>
#include <vtkDefaultPass.h>
#include <vtkLightsPass.h>
#include <vtkShadowMapPass.h>
#include <vtkShadowMapBakerPass.h>
#include <vtkRenderPassCollection.h>
#include <vtkCameraPass.h>
#include <vtkOverlayPass.h>
#include <vtkOpaquePass.h>
#include <vtkVolumetricPass.h>
#include <vtkTranslucentPass.h>

#include "VtkVisualModule.h"

namespace dyno
{
	class VtkRenderEngine : public RenderEngine
	{
	public:
		VtkRenderEngine();

		virtual void initialize(int width, int height) override;
		virtual void draw(dyno::SceneGraph* scene) override; 
		virtual void resize(int w, int h) override;

		virtual std::string name() override;

	private:
		void setScene(dyno::SceneGraph* scene);
		void setCamera();

	private:
		bool m_useOffScreen = true;

		vtkNew<vtkExternalOpenGLRenderer>		mVtkRenderer;
		vtkNew<vtkExternalOpenGLRenderWindow>	mVtkWindow;
		vtkNew<vtkExternalOpenGLCamera>			mVtkCamera;

		vtkNew<vtkLight>						mVtkLight;

		std::vector<VtkVisualModule*>		mVisualModules;

		// ground plane
		vtkNew<vtkPlaneSource>	mPlane;
		vtkNew<vtkPlaneSource>	mPlaneWireFrame;
		vtkNew<vtkActor>		mPlaneActor;
		vtkNew<vtkActor>		mPlaneWireFrameActor;

		// bounding box
		vtkNew<vtkCubeSource>	mSceneCube;
		vtkNew<vtkActor>		mBoxActor;

		// render passes
		struct
		{
			vtkNew<vtkRenderPassCollection> passes;
			vtkNew<vtkSequencePass>			seq;

			vtkNew<vtkShadowMapPass>		shadow;
			vtkNew<vtkShadowMapBakerPass>	baker;
			vtkNew<vtkLightsPass>			light;
			vtkNew<vtkTranslucentPass>		translucent;
			vtkNew<vtkVolumetricPass>		volume;
			vtkNew<vtkOverlayPass>			overlay;
			vtkNew<vtkOpaquePass>			opaque;
			
			vtkNew<vtkCameraPass>			cameraPass;
		} mRenderPasses;


		dyno::SceneGraph* m_scene = NULL;

		friend struct GatherVisualModuleAction;
	};
};