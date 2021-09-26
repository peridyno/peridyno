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

typedef std::map<int, std::string> enum_map;
//typedef std::map<std::string, enum_map>	enum_map_list;

#define DECLARE_ENUM(enum_type,...)					\
enum enum_type{										\
	__VA_ARGS__										\
};													\
const std::string full_name_##enum_type = #__VA_ARGS__;

bool parse_enum_string(const std::string& enum_str, enum_map& enumKeyValueList);


class PEnum
{
public:
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

private:
	std::string m_enum_name;
	int m_enum_value;

	enum_map m_enum_map;
};

#define DEF_ENUM(enum_type, enum_name, enum_value, desc)				\
private:									\
	FVar<PEnum> var_##enum_name = FVar<PEnum>(PEnum(#enum_type, enum_value, full_name_##enum_type), std::string(#enum_name), desc, FieldTypeEnum::Param, this);			\
public:										\
	inline FVar<PEnum>* var##enum_name() {return &var_##enum_name;}