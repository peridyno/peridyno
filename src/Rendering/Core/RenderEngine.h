/**
 * Copyright 2017-2023 Jian SHI
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

#include <memory>
#include <string>
#include <vector>
#include <glm/glm.hpp>

namespace dyno
{
	class SceneGraph;
	class Node;
	class Camera;

	struct RenderParams
	{		

		// illumination settings
		struct Light
		{
			// ambient light
			glm::vec3	ambientColor = glm::vec3(0.05f);
			float		ambientScale = 1.f;

			// directional light
			glm::vec3	mainLightColor = glm::vec3(1.f);
			float		mainLightScale = 5.f;
			glm::vec3	mainLightDirection = glm::vec3(0.4f, 0.6f, 0.8f);
			float		mainLightShadow = 1.f;	// 0 - disable shadow; otherwise enable shadow

			// camera light
			glm::vec3	cameraLightColor = glm::vec3(0.1f);
			float		cameraLightScale = 1.f;
		} light;

		 
		// Backcolor gray scale
		glm::vec3	bgColor0 = glm::vec3(0.2f);
		glm::vec3	bgColor1 = glm::vec3(0.8f);

		// some render options...
		bool  showGround = true;
		float planeScale = 3.f;
		float rulerScale = 1.f;

		bool showSceneBounds = false;
	};

	// data structure for mouse selection
	struct Selection {

		struct Item {
			std::shared_ptr<Node> node = 0;
			int   instance = -1;
			int	  primitive = -1;
		};

		int x = -1;
		int y = -1;
		int w = -1;
		int h = -1;
		std::vector<Item> items;
	};

	// RenderEngine interface
	class RenderEngine
	{
	public:
		virtual void initialize() = 0;
		virtual void terminate() = 0;

		virtual void draw(SceneGraph* scene, Camera* camera, const RenderParams& rparams) = 0;


		virtual Selection select(int x, int y, int w, int h) = 0;

		virtual std::string name() const = 0;
	};
};

