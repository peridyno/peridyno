/**
 * Copyright 2024 Xiawoei He
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

#include "GlfwGUI/GlfwApp.h"

#if (defined QT_GUI_SUPPORTED)
	#include "QtGUI/QtApp.h"
#endif

#if (defined WT_GUI_SUPPORTED)
	#include "WtGUI/WtApp.h"
#endif

namespace dyno
{
	enum GUIType
	{
		GUI_GLFW = 0,
		GUI_QT = 1,
		GUI_WT = 2
	};

	/**
	 * @brief This class provides a unified representation for all three GUIs, including the GlfwGUI, QtGUI and WtGUI
	 */
	class UbiApp : public AppBase
	{
	public:
		UbiApp(GUIType type = GUIType::GUI_GLFW);
		~UbiApp();

		void initialize(int width, int height, bool usePlugin = false) override;

		void mainLoop() override;

	private:
		AppBase* mApp = nullptr;

		GUIType mType = GUIType::GUI_GLFW;
	};
}
