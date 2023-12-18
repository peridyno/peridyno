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
	FVar<T> var_##name = FVar<T>(T(value), std::string(#name), desc, FieldTypeEnum::Param, this);			\
public:										\
	inline FVar<T>* var##name() {return &var_##name;}

#define DEF_VAR_IN(T, name, desc) \
private:									\
	FVar<T> in_##name = FVar<T>(std::string(#name), desc, FieldTypeEnum::In, this);			\
public:										\
	inline FVar<T>* in##name() {return &in_##name;}

#define DEF_VAR_OUT(T, name, desc) \
private:									\
	FVar<T> out_##name = FVar<T>(std::string(#name), desc, FieldTypeEnum::Out, this);			\
public:									\
	inline FVar<T>* out##name() {return &out_##name;}

#define DEF_VAR_STATE(T, name, value, desc) \
private:									\
	FVar<T> state_##name = FVar<T>(T(value), std::string(#name), desc, FieldTypeEnum::State, this);			\
public:										\
	inline FVar<T>* state##name() {return &state_##name;}

/**
*	Macro definition for input/output of type instance
*/
#define DEF_INSTANCE_IN(T, name, desc) \
private:									\
	FInstance<T> in_##name = FInstance<T>(std::string(#name), desc, FieldTypeEnum::In, this);			\
public:										\
	inline FInstance<T>* in##name() {return &in_##name;}

#define DEF_INSTANCE_OUT(T, name, desc) \
private:									\
	FInstance<T> out_##name = FInstance<T>(std::string(#name), desc, FieldTypeEnum::Out, this);			\
public:									\
	inline FInstance<T>* out##name() {return &out_##name;}

#define DEF_INSTANCE_IO(T, name, desc) \
private:									\
	FInstance<T> io_##name = FInstance<T>(std::string(#name), desc, FieldTypeEnum::IO, this);			\
public:									\
	inline FInstance<T>* io##name() {return &io_##name;}

/**
 * @brief Macro definitions for instance state
 *
 */
#define DEF_INSTANCE_STATE(T, name, desc)		\
private:									\
	FInstance<T> state_##name = FInstance<T>(std::string(#name), desc, FieldTypeEnum::State, this);	\
public:									\
	inline FInstance<T>* state##name() {return &state_##name;}

/**
 * @brief Macro definitions for an array of instance
 */
#define DEF_INSTANCES_STATE(T, name, desc)		\
private:									\
	FArray<std::shared_ptr<T>, DeviceType::CPU> state_##name = FArray<std::shared_ptr<T>, DeviceType::CPU>(std::string(#name)+std::string("(s)"), desc, FieldTypeEnum::State, this);	\
public:									\
	inline FArray<std::shared_ptr<T>, DeviceType::CPU>* state##name##s() {return &state_##name;}

#define DEF_INSTANCES_IN(T, name, desc)		\
private:									\
	FArray<std::shared_ptr<T>, DeviceType::CPU> in_##name = FArray<std::shared_ptr<T>, DeviceType::CPU>(std::string(#name)+std::string("(s)"), desc, FieldTypeEnum::In, this);	\
public:									\
	inline FArray<std::shared_ptr<T>, DeviceType::CPU>* in##name##s() {return &in_##name;}

#define DEF_INSTANCES_OUT(T, name, desc)		\
private:									\
	FArray<std::shared_ptr<T>, DeviceType::CPU> out_##name = FArray<std::shared_ptr<T>, DeviceType::CPU>(std::string(#name)+std::string("(s)"), desc, FieldTypeEnum::Out, this);	\
public:									\
	inline FArray<std::shared_ptr<T>, DeviceType::CPU>* out##namee##s() {return &out_##name;}


/**
*	Macro definition for input/output of type Array
*/
#define DEF_ARRAY_IN(T, name, device, desc) \
private:									\
	FArray<T, device> in_##name = FArray<T, device>(std::string(#name), desc, FieldTypeEnum::In, this);	\
public:									\
	inline FArray<T, device>* in##name() {return &in_##name;}

#define DEF_ARRAY_OUT(T, name, device, desc) \
private:									\
	FArray<T, device> out_##name = FArray<T, device>(std::string(#name), desc, FieldTypeEnum::Out, this);	\
public:									\
	inline FArray<T, device>* out##name() {return &out_##name;}

#define DEF_ARRAY_IO(T, name, device, desc) \
private:									\
	FArray<T, device> io_##name = FArray<T, device>(std::string(#name), desc, FieldTypeEnum::IO, this);	\
public:									\
	inline FArray<T, device>* in##name() {return &io_##name;}		\
	inline FArray<T, device>* out##name() {return &io_##name;}


/**
* Macro definition for input/output of type Array2D
*/
#define DEF_ARRAY2D_IN(T, name, device, desc)			\
private:												\
	FArray2D<T, device> in_##name = FArray2D<T, device>(std::string(#name), desc, FieldTypeEnum::In, this);	\
public:													\
	inline FArray2D<T, device>* in##name() {return &in_##name;}

#define DEF_ARRAY2D_OUT(T, name, device, desc)		\
private:												\
	FArray2D<T, device> out_##name = FArray2D<T, device>(std::string(#name), desc, FieldTypeEnum::Out, this);	\
public:													\
	inline FArray2D<T, device>* out##name() {return &out_##name;}

#define DEF_ARRAY2D_IO(T, name, device, desc)		\
private:												\
	FArray2D<T, device> io_##name = FArray2D<T, device>(std::string(#name), desc, FieldTypeEnum::IO, this);	\
public:													\
	inline FArray2D<T, device>* in##name() {return &io_##name;}	\
	inline FArray2D<T, device>* out##name() {return &io_##name;}


/**
* Macro definition for input/output of type Array3D
*/
#define DEF_ARRAY3D_IN(T, name, device, desc)			\
private:												\
	FArray3D<T, device> in_##name = FArray3D<T, device>(std::string(#name), desc, FieldTypeEnum::In, this);	\
public:													\
	inline FArray3D<T, device>* in##name() {return &in_##name;}

#define DEF_ARRAY3D_OUT(T, name, device, desc)		\
private:												\
	FArray3D<T, device> out_##name = FArray3D<T, device>(std::string(#name), desc, FieldTypeEnum::Out, this);	\
public:													\
	inline FArray3D<T, device>* out##name() {return &out_##name;}

#define DEF_ARRAY3D_IO(T, name, device, desc)		\
private:												\
	FArray3D<T, device> io_##name = FArray3D<T, device>(std::string(#name), desc, FieldTypeEnum::IO, this);	\
public:													\
	inline FArray3D<T, device>* in##name() {return &io_##name;}	\
	inline FArray3D<T, device>* out##name() {return &io_##name;}


/**
* Macro definition for input/output of type ArrayList
*/
#define DEF_ARRAYLIST_IN(T, name, device, desc)			\
private:												\
	FArrayList<T, device> in_##name = FArrayList<T, device>(std::string(#name), desc, FieldTypeEnum::In, this);	\
public:													\
	inline FArrayList<T, device>* in##name() {return &in_##name;}

#define DEF_ARRAYLIST_OUT(T, name, device, desc)		\
private:												\
	FArrayList<T, device> out_##name = FArrayList<T, device>(std::string(#name), desc, FieldTypeEnum::Out, this);	\
public:													\
	inline FArrayList<T, device>* out##name() {return &out_##name;}

#define DEF_ARRAYLIST_IO(T, name, device, desc)		\
private:												\
	FArrayList<T, device> io_##name = FArrayList<T, device>(std::string(#name), desc, FieldTypeEnum::IO, this);	\
public:													\
	inline FArrayList<T, device>* in##name() {return &io_##name;}	\
	inline FArrayList<T, device>* out##name() {return &io_##name;}
}