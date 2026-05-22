/**
 * Copyright 2025 Xiaowei He
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

#include <list>

#include "Field.h"
#include "Tuple.h"

namespace dyno {

	class FList : public FBase
	{
	public:
		FList() : FBase() {}
		FList(std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent)
			: FBase(name, description, fieldType, parent) {}

		~FList() override {};

		const std::string getClassName() override { return "FList"; }

		virtual std::list<std::unique_ptr<FBase>>::iterator begin() = 0;

		virtual std::list<std::unique_ptr<FBase>>::iterator end() = 0;

		virtual void pushBack() = 0;
		virtual void erase(FBase* f) = 0;
		virtual void clear() = 0;
	};

	/*!
	*	\class 	Variable of List
	*	\brief	Variables of build-in data types.
	*/
	template<typename T>
	class TFList : public FList
	{
	public:
		typedef T										VarType;
		typedef std::list<std::unique_ptr<FBase>>		DataType;
		typedef TFList<T>								FieldType;

		TFList() : FList() {}
		TFList(std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent)
			: FList(name, description, fieldType, parent) {}

		~TFList() override {};

		const std::string getTemplateName() override { return std::string(typeid(VarType).name()); }

		uint size() override { 
			auto data_ptr = this->constDataPtr();
			return data_ptr == nullptr ? 0 : data_ptr->size();
		}

		inline std::string serialize() override { return "Unknown"; }
		inline bool deserialize(const std::string& str) override { return false; }

		typename DataType::iterator begin() override {
			auto data_ptr = this->constDataPtr();
			return data_ptr->begin();
		}

		typename DataType::iterator end() override {
			auto data_ptr = this->constDataPtr();
			return data_ptr->end();
		}

		/**
		 * Push back a default value
		 */
		void pushBack() override;
		void erase(FBase* f) override;
		void clear() override;

		T getElement(std::list<std::unique_ptr<FBase>>::iterator it);

		void pushBack(T val);

		bool isEmpty() override {
			auto data_ptr = this->constDataPtr();
			return data_ptr == nullptr || data_ptr->size() == 0;
		}

		bool quote(FieldType* dst)
		{
			if (this->getFieldType() != FieldTypeEnum::Param || dst->getFieldType() != FieldTypeEnum::Param)
				return false;

			this->connectField(dst);
			return true;
		}

		void assign(const std::list<T>& lst);
		void assign(TFList<T>* lst);

	private:
		std::shared_ptr<DataType>& constDataPtr()
		{
			FBase* topField = this->getTopField();
			FieldType* derived = dynamic_cast<FieldType*>(topField);
			return derived->mDataPtr;
		}

		std::shared_ptr<DataType>& getDataPtr()
		{
			FBase* topField = this->getTopField();
			FieldType* derived = dynamic_cast<FieldType*>(topField);
			return derived->mDataPtr;
		}

		bool connect(FBase* dst) override {
			FieldType* derived = dynamic_cast<FieldType*>(dst);
			if (derived == nullptr) return false;
			return this->quote(derived);
		}

		std::shared_ptr<DataType> mDataPtr = std::make_shared<DataType>();
	};

	template<typename T>
	void TFList<T>::assign(const std::list<T>& lst)
	{
		this->clear();

		for (auto it = lst.begin(); it != lst.end(); it++)
		{
			this->pushBack(*it);
		}
	}

	template<typename T>
	void TFList<T>::assign(TFList<T>* lst)
	{
		this->clear();

		auto data_ptr = lst->constDataPtr();

		for (auto it = data_ptr->begin(); it != data_ptr->end(); it++)
		{
			this->pushBack(lst->getElement(it));
		}
	}

	template<typename T>
	void TFList<T>::clear()
	{
		auto& data_ptr = this->getDataPtr();
		data_ptr->clear();
	}

	template<typename T>
	void TFList<T>::erase(FBase* f)
	{
		auto& data_ptr = this->getDataPtr();
		auto it = std::find_if(data_ptr->begin(), data_ptr->end(),
			[f](const std::unique_ptr<FBase>& ptr) {
				return ptr.get() == f;
			});

		if (it != data_ptr->end()) {
			data_ptr->erase(it);
		}
	}

	template<typename T>
	void TFList<T>::pushBack()
	{
		T val{};
		this->pushBack(val);
	}

	template<typename T>
	T TFList<T>::getElement(std::list<std::unique_ptr<FBase>>::iterator it)
	{
		FBase* base_ptr = (*it).get();
		if constexpr (std::is_base_of<Tuple, T>::value) {
			if (auto* derived_ptr = dynamic_cast<TFTuple<T>*>(base_ptr)) {
				return derived_ptr->getValue();
			}
		}
		else
		{
			if (auto* derived_ptr = dynamic_cast<FVar<T>*>(base_ptr)) {
				return derived_ptr->getValue();
			}
		}
	}

	template<typename T>
	void TFList<T>::pushBack(T val)
	{
		auto& data = this->getDataPtr();

		//if constexpr requires C++ 17
		if constexpr (std::is_base_of<Tuple, T>::value) {
			auto tuplePtr = std::make_unique<TFTuple<T>>();
			tuplePtr->setValue(val);
			data->push_back(std::move(tuplePtr));
		}
		else
		{
			auto derived = std::make_unique<FVar<T>>();
			derived->setValue(val);
			data->push_back(std::move(derived));
		}

		this->update();

		this->tick();
	}


#define DEF_LIST(T, name, desc) \
private:									\
	TFList<T> var_##name = TFList<T>(std::string(#name), desc, FieldTypeEnum::Param, this);			\
public:										\
	inline TFList<T>* var##name() {return &var_##name;}
}

#include "FList.inl"
