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
 * @brief Macro definitions for FVar
 * 
 */
#define DEF_CURRENT_VAR(name, T, value, desc) \
private:									\
	FVar<T> current_##name = FVar<T>(T(value), std::string(#name), desc, FieldTypeEnum::Current, this);			\
public:										\
	inline FVar<T>* current##name() {return &current_##name;}

#define DEF_NEXT_VAR(name, T, value, desc) \
private:									\
	FVar<T> next_##name = FVar<T>(T(value), std::string(#name), desc, FieldTypeEnum::Next, this);			\
public:										\
	inline FVar<T>* next##name() {return &next_##name;}

#define DEF_EMPTY_CURRENT_VAR(name, T, desc) \
private:									\
	FVar<T> current_##name = FVar<T>(std::string(#name), desc, FieldTypeEnum::Current, this);			\
public:										\
	inline FVar<T>* current##name() {return &current_##name;}

#define DEF_EMPTY_NEXT_VAR(name, T, desc) \
private:									\
	FVar<T> next_##name = FVar<T>(std::string(#name), desc, FieldTypeEnum::Next, this);			\
public:									\
	inline FVar<T>* next##name() {return &next_##name;}


/**
 * @brief Macro definitions for ArrayField
 * 
 */
#define DEF_EMPTY_CURRENT_ARRAY(name, T, device, desc) \
private:									\
	FArray<T, device> current_##name = FArray<T, device>(std::string(#name), desc, FieldTypeEnum::Current, this);	\
public:									\
	inline FArray<T, device>* current##name() {return &current_##name;}

#define DEF_EMPTY_NEXT_ARRAY(name, T, device, desc) \
private:									\
	FArray<T, device> next_##name = FArray<T, device>(std::string(#name), desc, FieldTypeEnum::Next, this);	\
public:									\
	inline FArray<T, device>* next##name() {return &next_##name;}

#define DEF_ARRAY_STATE(T, name, device, desc) \
private:									\
	FArray<T, device> current_##name = FArray<T, device>(std::string(#name), desc, FieldTypeEnum::Current, this);	\
public:									\
	inline FArray<T, device>* state##name() {return &current_##name;}



#define DEF_EMPTY_CURRENT_ARRAYLIST(T, name, device, desc) \
private:									\
	FArrayList<T, device> current_##name = FArrayList<T, device>(std::string(#name), desc, FieldTypeEnum::Current, this);	\
public:									\
	inline FArrayList<T, device>* current##name() {return &current_##name;}

#define DEF_EMPTY_NEXT_ARRAYLIST(T, name, device, desc) \
private:									\
	FArrayList<T, device> next_##name = FArrayList<T, device>(std::string(#name), desc, FieldTypeEnum::Next, this);	\
public:									\
	inline FArrayList<T, device>* next##name() {return &next_##name;}

/**
 * @brief Macro definitions for neighbor list
 * 
 */
#define DEF_EMPTY_CURRENT_NEIGHBOR_LIST(name, T, desc)		\
private:									\
	NeighborField<T> current_##name = NeighborField<T>(std::string(#name), desc, FieldTypeEnum::Current, this, 0, 0);	\
public:									\
	inline NeighborField<T>* current##name() {return &current_##name;}

#define DEF_EMPTY_NEXT_NEIGHBOR_LIST(name, T, desc)		\
private:									\
	NeighborField<T> next_##name = NeighborField<T>(std::string(#name), desc, FieldTypeEnum::Next, this, 0, 0);	\
public:									\
	inline NeighborField<T>* next##name() {return &next_##name;}

 /**
  * @brief Macro definitions for instance
  *
  */
#define DEF_INSTANCE_STATE(T, name, desc)		\
private:									\
	FInstance<T> current_##name = FInstance<T>(std::string(#name), desc, FieldTypeEnum::Current, this);	\
public:									\
	inline FInstance<T>* current##name() {return &current_##name;}

/**
 * @brief Macro definitions for node ports
 * 
 */
#define DEF_NODE_PORT(T, name, desc)				\
private:									\
	SingleNodePort<T> single_##name = SingleNodePort<T>(std::string(#name), desc, this);					\
public:																										\
	inline std::shared_ptr<T> get##name() {	return single_##name.getDerivedNode(); }						\
																			\
	void set##name(std::shared_ptr<T> c) {									\
		single_##name.setDerivedNode(c);									\
	}

#define DEF_NODE_PORTS(name, T, desc)				\
private:									\
	MultipleNodePort<T> multiple_##name = MultipleNodePort<T>(std::string(#name)+std::string("(s)"), desc, this);					\
public:									\
	inline MultipleNodePort<T>* inport##name##s() { return &multiple_##name; }			\
	inline std::vector<std::shared_ptr<T>>& get##name##s(){return multiple_##name.getDerivedNodes();}				\
														\
	bool add##name(std::shared_ptr<T> c){				\
		multiple_##name.addDerivedNode(c);				\
		return true;									\
	}													\
														\
	bool remove##name(std::shared_ptr<T> c) {			\
		multiple_##name.removeDerivedNode(c);			\
		return true;									\
	}
}