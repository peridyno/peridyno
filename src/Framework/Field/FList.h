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

#include "FBase.h"

#include "Array/Array.h"

namespace dyno {
	/*!
	*	\class 	Variable of List
	*	\brief	Variables of build-in data types.
	*/
	template<typename T>
	class FList : public FBase
	{
	public:
		typedef T					VarType;
		typedef std::list<T>		DataType;
		typedef FList<T>			FieldType;

		FList() : FBase("", "") {}
		FList(std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent)
			: FBase(name, description, fieldType, parent) {}

		~FList() override;

		const std::string getTemplateName() override { return std::string(typeid(VarType).name()); }
		const std::string getClassName() override { return "FList"; }

		uint size() override { return mList.size(); }

		inline std::string serialize() override { return "Unknown"; }
		inline bool deserialize(const std::string& str) override { return false; }

		void insert(T val);

		bool isEmpty() override {
			auto& data = this->constData();
			return data.size() == 0;
		}

		const DataType& constData()
		{
			FBase* topField = this->getTopField();
			FieldType* derived = dynamic_cast<FieldType*>(topField);
			return derived->mList;
		}

		bool bind(FieldType* dst)
		{
			if (this->getFieldType() != FieldTypeEnum::Param || dst->getFieldType() != FieldTypeEnum::Param)
				return false;

			this->connectField(dst);
			return true;
		}

	protected:
		bool connect(FBase* dst) override {
			FieldType* derived = dynamic_cast<FieldType*>(dst);
			if (derived == nullptr) return false;
			return this->bind(derived);
		}

	private:
		std::list<T> mList;
	};

	template<typename T>
	void FList<T>::insert(T val)
	{
		std::shared_ptr<T>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<T>(val);
		}
		else
		{
			*data = val;
		}

		this->update();

		this->tick();
	}

	template<typename T>
	FList<T>::~FList()
	{
	};

#define DEF_LIST(T, name, desc) \
private:									\
	FList<T> var_##name = FList<T>(std::string(#name), desc, FieldTypeEnum::Param, this);			\
public:										\
	inline FList<T>* var##name() {return &var_##name;}
}

#include "FList.inl"
