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
#include "Node.h"
#include "FilePath.h"

#include "Topology/PointSet.h"

namespace dyno
{
	template<typename TDataType>
	class InputFieldTest : public Node
	{
		DECLARE_TCLASS(InputFieldTest, TDataType)
	public:
		typedef typename TDataType::Coord Coord;

		InputFieldTest();

	public:
		void updateStates() override;

		DEF_VAR(Real, Variable, Real(0.1), "Test a variable range");

		DEF_VAR(FilePath, FileName, std::string(""), "");

		DEF_INSTANCE_IN(PointSet<TDataType>, PointSet, "");
	};

	IMPLEMENT_TCLASS(InputFieldTest, TDataType)
};
