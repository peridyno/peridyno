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
/**
 * @brief Macro definitions for ArrayField
 * 
 */
#define DEF_ARRAY_STATE(T, name, device, desc) \
private:									\
	FArray<T, device> state_##name = FArray<T, device>(std::string(#name), desc, FieldTypeEnum::State, this);	\
public:									\
	inline FArray<T, device>* state##name() {return &state_##name;}


#define DEF_ARRAY2D_STATE(T, name, device, desc) \
private:									\
	FArray2D<T, device> state_##name = FArray2D<T, device>(std::string(#name), desc, FieldTypeEnum::State, this);	\
public:									\
	inline FArray2D<T, device>* state##name() {return &state_##name;}

#define DEF_ARRAY3D_STATE(T, name, device, desc) \
private:									\
	FArray3D<T, device> state_##name = FArray3D<T, device>(std::string(#name), desc, FieldTypeEnum::State, this);	\
public:									\
	inline FArray3D<T, device>* state##name() {return &state_##name;}


#define DEF_ARRAYLIST_STATE(T, name, device, desc) \
private:									\
	FArrayList<T, device> state_##name = FArrayList<T, device>(std::string(#name), desc, FieldTypeEnum::State, this);	\
public:									\
	inline FArrayList<T, device>* state##name() {return &state_##name;}

 /**
  * @brief Macro definitions for instance
  */
#define DEF_INSTANCE_STATE(T, name, desc)		\
private:									\
	FInstance<T> state_##name = FInstance<T>(std::string(#name), desc, FieldTypeEnum::State, this);	\
public:									\
	inline FInstance<T>* state##name() {return &state_##name;}

/**
 * @brief Macro definitions for node ports
 * 
 */
#define DEF_NODE_PORT(T, name, desc)				\
private:									\
	SingleNodePort<T> single_##name = SingleNodePort<T>(std::string(#name), desc, this);					\
public:																										\
	inline T* get##name() {	return single_##name.getDerivedNode(); }						\
																			\
	SingleNodePort<T>* import##name(){ return &single_##name; }
 

#define DEF_NODE_PORTS(T, name, desc)				\
private:									\
	MultipleNodePort<T> multiple_##name = MultipleNodePort<T>(std::string(#name)+std::string("(s)"), desc, this);					\
public:									\
	inline MultipleNodePort<T>* import##name##s() { return &multiple_##name; }			\
	inline std::vector<T*>& get##name##s(){return multiple_##name.getDerivedNodes();}				\
														\
	bool add##name(std::shared_ptr<T> c){				\
		multiple_##name.addDerivedNode(c.get());				\
		return true;									\
	}													\
														\
	bool remove##name(std::shared_ptr<T> c) {			\
		multiple_##name.removeDerivedNode(c.get());			\
		return true;									\
	}
}