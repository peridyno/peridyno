/**
 * Copyright 2021 Xiaowei He
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
	class CollidableObject : public Module
	{
	public:
		enum CType {
			SPHERE_TYPE = 0,
			TRIANGLE_TYPE,
			TETRAHEDRON_TYPE,
			POINTSET_TYPE,
			GEOMETRIC_PRIMITIVE_TYPE,
			SIGNED_DISTANCE_TYPE,
			UNDFINED
		};

	public:
		CollidableObject(CType ctype);
		virtual ~CollidableObject();

		CType getType() { return m_type; }

		//should be called before collision is started
		virtual void updateCollidableObject() = 0;

		//should be called after the collision is finished
		virtual void updateMechanicalState() = 0;
		
		std::string getModuleType() override { return "CollidableObject"; }
	private:
		CType m_type;
	};
}
