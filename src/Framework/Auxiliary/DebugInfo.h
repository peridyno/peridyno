/**
 * Copyright 2017-2021 Xiaowei He
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

namespace dyno 
{
	class DebugInfo : public Module
	{
	public:
		DebugInfo() {};
		~DebugInfo() override {};

		virtual void print() = 0;

		std::string getModuleType() override { return "Auxiliary"; }

		DEF_VAR(std::string, Prefix, "", "Prefix");

	private:
		void updateImpl() final;
	};

	class PrintInt : public DebugInfo
	{
		DECLARE_CLASS(PrintInt);
	public:
		PrintInt() {};

		DEF_VAR_IN(int, Int, "Input value");

		void print() override;
	};

	IMPLEMENT_CLASS(PrintInt);

	class PrintUnsigned : public DebugInfo
	{
		DECLARE_CLASS(PrintUnsigned);
	public:
		PrintUnsigned() {};

		DEF_VAR_IN(uint, Unsigned, "Input value");

		void print() override;
	};

	IMPLEMENT_CLASS(PrintUnsigned);

	class PrintFloat : public DebugInfo
	{
		DECLARE_CLASS(PrintFloat);
	public:
		PrintFloat() {};
		~PrintFloat() {};

		DEF_VAR_IN(float, Float, "Input value");

		void print() override;
	};

	IMPLEMENT_CLASS(PrintFloat);
}
