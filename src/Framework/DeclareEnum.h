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
#include <map>
#include <string>

#include "Field.h"

namespace dyno
{

	typedef std::map<int, std::string> EnumMap;
	//typedef std::map<std::string, enum_map>	enum_map_list;

	#define DECLARE_ENUM(enum_type,...)					\
		enum enum_type{										\
			__VA_ARGS__										\
		};													\
	const std::string full_name_##enum_type = #__VA_ARGS__;

	bool parse_enum_string(const std::string& enum_str, EnumMap& enumKeyValueList);


	class PEnum
	{
	public:
		PEnum()
		{
			m_enum_name = "";
			m_enum_value = -1;
		}

		PEnum(std::string enum_name, int val, const std::string enum_str)
		{
			parse_enum_string(enum_str, m_enum_map);
			m_enum_value = val;
			m_enum_name = enum_name;
		}

		inline bool operator== (const int val) const
		{
			return m_enum_value == val;
		}

		inline bool operator!= (const int val) const
		{
			return m_enum_value != val;
		}

		int currentKey() { return m_enum_value; }

		std::string currentString() {
			return m_enum_map[m_enum_value];
		}

		void setCurrentKey(int index);

		EnumMap& enumMap() { return m_enum_map; }

	private:
		std::string m_enum_name;
		int m_enum_value;

		EnumMap m_enum_map;
	};

	/*!
	*	\class	FVar<PEnum>
	*	\brief	Specialization for the field type of PEnum.
	*/
	template<>
	class FVar<PEnum> : public FBase
	{
	public:
		typedef PEnum				VarType;
		typedef PEnum				DataType;
		typedef FVar<PEnum>			FieldType;

		DEFINE_FIELD_FUNC(FieldType, DataType, FVar);

		FVar(VarType value, std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent)
			: FBase(name, description, fieldType, parent) 
		{
			this->setValue(value);
		}

		~FVar() override {};

		uint size() override { return 1; }

		void setValue(VarType val) {
			std::shared_ptr<VarType>& data = this->getDataPtr();
			if (data == nullptr)
			{
				data = std::make_shared<VarType>(val);
			}
			else
			{
				*data = val;
			}

			this->update();

			this->tick();
		}

		VarType getValue() {
			std::shared_ptr<VarType>& data = this->constDataPtr();
			return *data;
		}

		int currentKey() {
			std::shared_ptr<VarType>& data = this->constDataPtr();
			return data->currentKey();
		}

		void setCurrentKey(int index) 
		{
			std::shared_ptr<VarType>& data = this->getDataPtr();
			
			data->setCurrentKey(index);

			///Call other functions
			this->update();

			//Notify this field is updated
			this->tick();
		}

		std::string serialize()
		{
			if (isEmpty())
				return "";

			int key = this->constDataPtr()->currentKey();

			std::stringstream ss;
			ss << key;

			return ss.str();
		}

		bool deserialize(const std::string& str)
		{
			if (str.empty())
				return false;

			int key = std::stoi(str);

			this->getDataPtr()->setCurrentKey(key);

			return true;
		}

		bool isEmpty() override {
			return this->constDataPtr() == nullptr;
		}
	};


#define DEF_ENUM(enum_type, enum_name, enum_value, desc)				\
private:									\
	FVar<PEnum> var_##enum_name = FVar<PEnum>(PEnum(#enum_type, enum_value, full_name_##enum_type), std::string(#enum_name), desc, FieldTypeEnum::Param, this);			\
public:										\
	inline FVar<PEnum>* var##enum_name() {return &var_##enum_name;}

}