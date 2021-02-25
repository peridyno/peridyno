/**
 * Copyright 2021 Xiawoei He
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
#include <iostream>
#include "Framework/Field.h"
#include "Framework/Object.h"

namespace dyno {
/**
*  \brief Base class for modules
*
*  This class contains all functionality shared by every module in Peridyno.
*  It defines how to retrieve information about an class (name, type, data fields).
*
*/

typedef std::string FieldID;

class Base : public Object
{
public:
	typedef std::vector<Field*> FieldVector;
	typedef std::map<FieldID, Field*> FieldMap;

	Base() : Object() {};
	~Base() override {};

	/**
	 * @brief Add a field to Base
	 * FieldID will be set to the name of Field by default
	 */
	bool addField(Field* data);
	/**
	 * @brief Add a field to Base
	 * 
	 * @param Field name
	 * @param Field pointer
	 */
	bool addField(FieldID name, Field* data);
	bool addFieldAlias(FieldID name, Field* data);
	bool addFieldAlias(FieldID name, Field* data, FieldMap& fieldAlias);

	/**
	 * @brief Find a field by its pointer
	 * 
	 * @param data Field pointer
	 */
	bool findField(Field* data);
	/**
	 * @brief Find a field by its name
	 * 
	 * @param name Field name
	 */
	bool findFieldAlias(const FieldID name);
	/**
	 * @brief Find a field in fieldAlias by its name
	 * This function is typically called by other functions
	 * 
	 * @param name Field name
	 * @param fieldAlias All fields the searching is taken on
	 */
	bool findFieldAlias(const FieldID name, FieldMap& fieldAlias);

	/**
	 * @brief Remove a field by its pointer
	 * 
	 */
	bool removeField(Field* data);
	/**
	 * @brief Remove a field by its name
	 * 
	 */
	bool removeFieldAlias(const FieldID name);
	bool removeFieldAlias(const FieldID name, FieldMap& fieldAlias);

	/**
	 * @brief Return a field by its name
	 * 
	 */
	Field*	getField(const FieldID name);

	std::vector<Field*>& getAllFields();

	/**
	 * @brief Attach a field to Base
	 * 
	 * @param field Field pointer
	 * @param name Field name
	 * @param desc Field description
	 * @param autoDestroy The field will be destroyed by Base if true, otherwise, the field should be explicitly destroyed by its creator.
	 * 
	 * @return Return false if the name conflicts with exists fields' names
	 */
	virtual bool attachField(Field* field, std::string name, std::string desc, bool autoDestroy = true);

	template<typename T>
	T* getField(FieldID name)
	{
		FieldMap::iterator iter = m_fieldAlias.find(name);
		if (iter != m_fieldAlias.end())
		{
			return dynamic_cast<T*>(iter->second);
		}
		return nullptr;
	}

	/**
	 * @brief Check the completeness of all required fields
	 */
	bool isAllFieldsReady();

	std::vector<FieldID>	getFieldAlias(Field* data);
	int				getFieldAliasCount(Field* data);


	inline void setBlockCoord(float x, float y) { block_x = x; block_y = y; }

	inline float bx() { return block_x; }
	inline float by() { return block_y; }

private:
	float block_x = 0.0f;
	float block_y = 0.0f;

	FieldVector m_field;
	FieldMap m_fieldAlias;
};

}
