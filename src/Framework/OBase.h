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
#include "FBase.h"
#include "Object.h"

namespace dyno {
	/**
	*  \brief Base class for modules
	*
	*  This class contains all functionality shared by every module in Peridyno.
	*  It defines how to retrieve information about an class (name, type, data fields).
	*
	*/

	typedef std::string FieldID;

	class OBase : public Object
	{
	public:
		typedef std::vector<FBase*> FieldVector;
		typedef std::map<FieldID, FBase*> FieldMap;

		OBase() : Object() {};
		~OBase() override;

		/**
		 * @brief Return the caption
		 * 
		 * @return The default value is the class name, overload this virtual function to return a user-defined caption name
		 */
		virtual std::string caption();

		/**
		 * @brief Add a field to Base
		 * FieldID will be set to the name of Field by default
		 */
		bool addField(FBase* data);
		/**
		 * @brief Add a field to Base
		 *
		 * @param Field name
		 * @param Field pointer
		 */
		bool addField(FieldID name, FBase* data);
		bool addFieldAlias(FieldID name, FBase* data);
		bool addFieldAlias(FieldID name, FBase* data, FieldMap& fieldAlias);

		/**
		 * @brief Find a field by its pointer
		 *
		 * @param data Field pointer
		 */
		bool findField(FBase* data);
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
		bool removeField(FBase* data);
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
		FBase* getField(const FieldID name);

		std::vector<FBase*>& getAllFields();

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
		virtual bool attachField(FBase* field, std::string name, std::string desc, bool autoDestroy = true);

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

		std::vector<FieldID>	getFieldAlias(FBase* data);
		int				getFieldAliasCount(FBase* data);


		inline void setBlockCoord(float x, float y) { block_x = x; block_y = y; }

		inline float bx() { return block_x; }
		inline float by() { return block_y; }


		bool findInputField(FBase* field);
		bool addInputField(FBase* field);
		bool removeInputField(FBase* field);

		std::vector<FBase*>& getInputFields() { return fields_input; }

		bool findOutputField(FBase* field);
		bool addOutputField(FBase* field);
		bool addToOutput(FBase* field);
		bool removeOutputField(FBase* field);
		bool removeFromOutput(FBase* field);

		std::vector<FBase*>& getOutputFields() { return fields_output; }

		bool findParameter(FBase* field);
		bool addParameter(FBase* field);
		bool removeParameter(FBase* field);

		std::vector<FBase*>& getParameters() { return fields_param; }

	private:
		float block_x = 0.0f;
		float block_y = 0.0f;

		FieldVector m_field;
		FieldMap m_fieldAlias;

	protected:
		std::vector<FBase*> fields_input;
		std::vector<FBase*> fields_output;
		std::vector<FBase*> fields_param;
	};

}
