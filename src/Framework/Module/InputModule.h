/**
 * Copyright 2022 Xiaowei He
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
#include "Module.h"
#include "Camera.h"

#include "Primitive/Primitive3D.h"

namespace dyno
{
	enum PButtonType
	{
		BT_UNKOWN = -1,
		BT_LEFT = 0,
		BT_RIGHT = 1,
		BT_MIDDLE = 2,
	};

	enum PActionType
	{
		AT_UNKOWN = -1,
		AT_RELEASE = 0,
		AT_PRESS = 1,
		AT_REPEAT = 2
	};

	enum PKeyboardType
	{
		PKEY_UNKNOWN = -1,
		PKEY_SPACE = 32,
		PKEY_APOSTROPHE = 39,
		PKEY_COMMA = 44,
		PKEY_MINUS = 45,
		PKEY_PERIOD = 46,
		PKEY_SLASH = 47,
		PKEY_0 = 48,
		PKEY_1 = 49,
		PKEY_2 = 50,
		PKEY_3 = 51,
		PKEY_4 = 52,
		PKEY_5 = 53,
		PKEY_6 = 54,
		PKEY_7 = 55,
		PKEY_8 = 56,
		PKEY_9 = 57,
		PKEY_SEMICOLON = 59,
		PKEY_EQUAL = 61,
		PKEY_A = 65,
		PKEY_B = 66,
		PKEY_C = 67,
		PKEY_D = 68,
		PKEY_E = 69,
		PKEY_F = 70,
		PKEY_G = 71,
		PKEY_H = 72,
		PKEY_I = 73,
		PKEY_J = 74,
		PKEY_K = 75,
		PKEY_L = 76,
		PKEY_M = 77,
		PKEY_N = 78,
		PKEY_O = 79,
		PKEY_P = 80,
		PKEY_Q = 81,
		PKEY_R = 82,
		PKEY_S = 83,
		PKEY_T = 84,
		PKEY_U = 85,
		PKEY_V = 86,
		PKEY_W = 87,
		PKEY_X = 88,
		PKEY_Y = 89,
		PKEY_Z = 90,
		PKEY_LEFT_BRACKET = 91,
		PKEY_BACKSLASH = 92,
		PKEY_RIGHT_BRACKET = 93,
		PKEY_GRAVE_ACCENT = 96,
		PKEY_WORLD_1 = 161,
		PKEY_WORLD_2 = 162,
		PKEY_ESCAPE = 256,
		PKEY_ENTER = 257,
		PKEY_TAB = 258,
		PKEY_BACKSPACE = 259,
		PKEY_INSERT = 260,
		PKEY_DELETE = 261,
		PKEY_RIGHT = 262,
		PKEY_LEFT = 263,
		PKEY_DOWN = 264,
		PKEY_UP = 265,
		PKEY_PAGE_UP = 266,
		PKEY_PAGE_DOWN = 267,
		PKEY_HOME = 268,
		PKEY_END = 269,
		PKEY_CAPS_LOCK = 280,
		PKEY_SCROLL_LOCK = 281,
		PKEY_NUM_LOCK = 282,
		PKEY_PRINT_SCREEN = 283,
		PKEY_PAUSE = 284,
		PKEY_F1 = 290,
		PKEY_F2 = 291,
		PKEY_F3 = 292,
		PKEY_F4 = 293,
		PKEY_F5 = 294,
		PKEY_F6 = 295,
		PKEY_F7 = 296,
		PKEY_F8 = 297,
		PKEY_F9 = 298,
		PKEY_F10 = 299,
		PKEY_F11 = 300,
		PKEY_F12 = 301,
		PKEY_F13 = 302,
		PKEY_F14 = 303,
		PKEY_F15 = 304,
		PKEY_F16 = 305,
		PKEY_F17 = 306,
		PKEY_F18 = 307,
		PKEY_F19 = 308,
		PKEY_F20 = 309,
		PKEY_F21 = 310,
		PKEY_F22 = 311,
		PKEY_F23 = 312,
		PKEY_F24 = 313,
		PKEY_F25 = 314,
		PKEY_KP_0 = 320,
		PKEY_KP_1 = 321,
		PKEY_KP_2 = 322,
		PKEY_KP_3 = 323,
		PKEY_KP_4 = 324,
		PKEY_KP_5 = 325,
		PKEY_KP_6 = 326,
		PKEY_KP_7 = 327,
		PKEY_KP_8 = 328,
		PKEY_KP_9 = 329,
		PKEY_KP_DECIMAL = 330,
		PKEY_KP_DIVIDE = 331,
		PKEY_KP_MULTIPLY = 332,
		PKEY_KP_SUBTRACT = 333,
		PKEY_KP_ADD = 334,
		PKEY_KP_ENTER = 335,
		PKEY_KP_EQUAL = 336,
		PKEY_LEFT_SHIFT = 340,
		PKEY_LEFT_CONTROL = 341,
		PKEY_LEFT_ALT = 342,
		PKEY_LEFT_SUPER = 343,
		PKEY_RIGHT_SHIFT = 344,
		PKEY_RIGHT_CONTROL = 345,
		PKEY_RIGHT_ALT = 346,
		PKEY_RIGHT_SUPER = 347
	};

	enum PModifierBits
	{
		MB_NO_MODIFIER = 0x0000,
		MB_SHIFT = 0x0001,		//If the Shift keys were held down.
		MB_CONTROL = 0x0002,	//If the Control keys were held down.
		MB_ALT = 0x0004,		//If the Alt keys were held down.
		MB_SUPER = 0x0008,		//If the Super keys were held down.
		MB_CAPS_LOCK = 0x0010,	//If the Caps Lock key is enabled.
		MB_NUM_LOCK = 0x0020	//If the Num Lock key is enabled.
	};

	struct PKeyboardEvent
	{
		bool shiftKeyPressed() { return (mods & PModifierBits::MB_SHIFT) != 0; }
		bool controlKeyPressed() { return (mods & PModifierBits::MB_CONTROL) != 0; }
		bool altKeyPressed() { return (mods & PModifierBits::MB_ALT) != 0; }
		bool superKeyPressed() { return (mods & PModifierBits::MB_SUPER) != 0; }
		bool capsLockEnabled() { return (mods & PModifierBits::MB_CAPS_LOCK) != 0; }
		bool numLockEnabled() { return (mods & PModifierBits::MB_NUM_LOCK) != 0; }

		PKeyboardType key =  PKEY_UNKNOWN;
		PActionType action = AT_UNKOWN;
		PModifierBits mods = MB_NO_MODIFIER;
	};

	struct PMouseEvent
	{
		bool operator==(const PMouseEvent& event) {
			return buttonType == event.buttonType && actionType == event.actionType && mods == event.mods;
		};

		bool operator!=(const PMouseEvent& event) {
			return buttonType != event.buttonType || actionType == event.actionType || mods == event.mods;
		};

		bool shiftKeyPressed() { return (mods & PModifierBits::MB_SHIFT) != 0; }
		bool controlKeyPressed() { return (mods & PModifierBits::MB_CONTROL) != 0; }
		bool altKeyPressed() { return (mods & PModifierBits::MB_ALT) != 0; }
		bool superKeyPressed() { return (mods & PModifierBits::MB_SUPER) != 0; }
		bool capsLockEnabled() { return(mods & PModifierBits::MB_CAPS_LOCK) != 0; }
		bool numLockEnabled() { return (mods & PModifierBits::MB_NUM_LOCK) != 0; }

		PButtonType buttonType;

		PActionType actionType;

		PModifierBits mods = PModifierBits::MB_NO_MODIFIER;

		TRay3D<float> ray;

		std::shared_ptr<Camera> camera;

		float x;
		float y;
	};

	class InputModule : public Module
	{
	public:
		InputModule();
		virtual ~InputModule();

		std::string getModuleType() final { return "InputModule"; }
	};
}
