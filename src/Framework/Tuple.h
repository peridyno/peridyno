/**
 * Copyright 2026 Xiaowei He
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
#include "OBase.h"
#
#include "DeclareEnum.h"
#include "DeclareField.h"


namespace dyno {
	/*!
	*	\class	Tuple
	*	\brief	Tuple provides the base class for complex structure that requires reflection.
	*/

	class Tuple : public OBase
	{
	public:
		Tuple();
		~Tuple() override;
	};

	class FTuple : public FBase
	{
	public:
		FTuple() : FBase("", "") {}
		FTuple(std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent)
			: FBase(name, description, fieldType, parent) {}

		~FTuple() override {};

		const std::string getClassName() override { return "FTuple"; }

		uint size() override { return 0; }

		inline std::string serialize() override { return "Unknown"; }
		inline bool deserialize(const std::string& str) override { return false; }

		virtual FBase* get(uint i) = 0;
	};

	template<typename T>
	class TFTuple : public FTuple
	{
	public:
		typedef T				VarType;
		typedef T				DataType;
		typedef TFTuple<T>			FieldType;

		TFTuple() : FBase("", "") { m_data = std::make_shared<DataType>(); }
		TFTuple(std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent)
			: FTuple(name, description, fieldType, parent) {
			m_data = std::make_shared<DataType>();
		}

		~TFTuple() override {};

		const std::string getTemplateName() override { return std::string(typeid(VarType).name()); }

		inline std::string serialize() override { return "Unknown"; }
		inline bool deserialize(const std::string& str) override { return false; }

		bool isEmpty() override {
			return false;
		}

		uint size() override {
			auto ptr = this->getDataPtr();
			if (ptr != nullptr)
			{
				auto& fields = ptr->getAllFields();
				return fields.size();
			}

			return 0; 
		}

		bool connect(FBase* dst) override {
			return false;
		}

		bool quote(FBase* dst) {
			bool valid = false;
			valid = valid || (this->getFieldType() == FieldTypeEnum::Param && dst->getFieldType() == FieldTypeEnum::Param);
			assert(valid == true);

			FieldType* derived = dynamic_cast<FieldType*>(dst);
			if (derived == nullptr) return false;

			return this->connectField(dst);
		}

		std::shared_ptr<DataType>& constDataPtr()
		{
			FBase* topField = this->getTopField();
			FieldType* derived = dynamic_cast<FieldType*>(topField);
			return derived->m_data;
		}

		FBase* get(uint i) override {
			auto ptr = this->getDataPtr();
			if (ptr != nullptr)
			{
				auto& fields = ptr->getAllFields();
				return fields[i];
			}

			return nullptr;
		};

	private:
		std::shared_ptr<DataType>& getDataPtr()
		{
			FBase* topField = this->getTopField();
			FieldType* derived = dynamic_cast<FieldType*>(topField);
			return derived->m_data;
		}

		std::shared_ptr<DataType> m_data = nullptr;
	};

#define DEF_TUPLE(T, name, desc) \
private:									\
	TFTuple<T> var_##name = TFTuple<T>(std::string("var_") + std::string(#name), desc, FieldTypeEnum::Param, this);			\
public:										\
	inline TFTuple<T>* tuple##name() {return &var_##name;}
}