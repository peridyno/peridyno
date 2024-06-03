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

#include "RenderParams.h"

namespace dyno
{
	class SceneGraph;
	class Node;
	class Camera;
	struct RenderParams;

	enum EEnvStyle
	{
		Standard = 0,
		Studio = 1
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

		virtual void draw(SceneGraph* scene, const RenderParams& rparams) = 0;


		virtual Selection select(int x, int y, int w, int h) = 0;

		virtual std::string name() const = 0;

		virtual void setDefaultEnvmap() {};

		void setUseEnvmapBackground(bool flag) { bDrawEnvmap = flag; }
		void setEnvmapScale(float scale) { enmapScale = scale; }

		virtual void setEnvStyle(EEnvStyle style) { envStyle = style; }

	public:

		// Backcolor gray scale
		glm::vec3 bgColor0 = glm::vec3(0.2f);
		glm::vec3 bgColor1 = glm::vec3(0.8f);

		// some render options...
		bool  showGround = true;
		float planeScale = 3.f;
		float rulerScale = 1.f;
		glm::vec4 planeColor = { 0.3, 0.3, 0.3, 0.5 };
		glm::vec4 rulerColor = { 0.0, 0.0, 0.0, 0.5 };

		bool  bDrawEnvmap = false;
		float enmapScale = 0.0f;

		bool  showSceneBounds = false;

		int envStyle = 0;
	};
};

