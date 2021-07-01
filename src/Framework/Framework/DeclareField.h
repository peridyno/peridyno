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

namespace dyno
{

#define DEF_VAR(T, name, value, desc) \
private:									\
	VarField<T> var_##name = VarField<T>(T(value), std::string(#name), desc, FieldTypeEnum::Param, this);			\
public:										\
	inline VarField<T>* var##name() {return &var_##name;}

#define DEF_VAR_IN(T, name, desc) \
private:									\
	VarField<T> in_##name = VarField<T>(std::string(#name), desc, FieldTypeEnum::In, this);			\
public:										\
	inline VarField<T>* in##name() {return &in_##name;}

#define DEF_VAR_OUT(T, name, desc) \
private:									\
	VarField<T> out_##name = VarField<T>(std::string(#name), desc, FieldTypeEnum::Out, this);			\
public:									\
	inline VarField<T>* out##name() {return &out_##name;}

/**
*	Macro definition for input/output of type instance
*/
#define DEF_INSTANCE_IN(T, name, desc) \
private:									\
	InstanceField<T> in_##name = InstanceField<T>(std::string(#name), desc, FieldTypeEnum::In, this);			\
public:										\
	inline InstanceField<T>* in##name() {return &in_##name;}

#define DEF_INSTANCE_OUT(T, name, desc) \
private:									\
	InstanceField<T> out_##name = InstanceField<T>(std::string(#name), desc, FieldTypeEnum::Out, this);			\
public:									\
	inline InstanceField<T>* out##name() {return &out_##name;}

#define DEF_INSTANCE_STATE(T, name, desc) \
private:									\
	InstanceField<T> state_##name = InstanceField<T>(std::string(#name), desc, FieldTypeEnum::Current, this);			\
public:									\
	inline InstanceField<T>* state##name() {return &state_##name;}


/**
*	Macro definition for input/output of type Array
*/
#define DEF_ARRAY_IN(T, name, device, desc) \
private:									\
	ArrayField<T, device> in_##name = ArrayField<T, device>(std::string(#name), desc, FieldTypeEnum::In, this);	\
public:									\
	inline ArrayField<T, device>* in##name() {return &in_##name;}

#define DEF_ARRAY_OUT(T, name, device, desc) \
private:									\
	ArrayField<T, device> out_##name = ArrayField<T, device>(std::string(#name), desc, FieldTypeEnum::Out, this);	\
public:									\
	inline ArrayField<T, device>* out##name() {return &out_##name;}

#define DEF_ARRAY_IO(T, name, device, desc) \
private:									\
	ArrayField<T, device> io_##name = ArrayField<T, device>(std::string(#name), desc, FieldTypeEnum::IO, this);	\
public:									\
	inline ArrayField<T, device>* in##name() {return &io_##name;}		\
	inline ArrayField<T, device>* out##name() {return &io_##name;}


/**
* Macro definition for input/output of type ArrayList
*/
#define DEF_ARRAYLIST_IN(T, name, device, desc)			\
private:												\
	ArrayListField<T, device> in_##name = ArrayListField<T, device>(std::string(#name), desc, FieldTypeEnum::In, this);	\
public:													\
	inline ArrayListField<T, device>* in##name() {return &in_##name;}

#define DEF_ARRAYLIST_OUT(T, name, device, desc)		\
private:												\
	ArrayListField<T, device> out_##name = ArrayListField<T, device>(std::string(#name), desc, FieldTypeEnum::Out, this);	\
public:													\
	inline ArrayListField<T, device>* out##name() {return &out_##name;}

#define DEF_ARRAYLIST_IO(T, name, device, desc)		\
private:												\
	ArrayListField<T, device> io_##name = ArrayListField<T, device>(std::string(#name), desc, FieldTypeEnum::IO, this);	\
public:													\
	inline ArrayListField<T, device>* in##name() {return &io_##name;}	\
	inline ArrayListField<T, device>* out##name() {return &io_##name;}
}