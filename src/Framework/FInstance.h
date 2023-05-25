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
#include <iostream>
#include "FBase.h"

namespace dyno {

	class Object;

	class InstanceBase : public FBase
	{
	public:
		InstanceBase() : FBase() {};
		InstanceBase(std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent)
			: FBase(name, description, fieldType, parent) {}

	public:
		virtual bool canBeConnectedBy(InstanceBase* ins) = 0;
		virtual void setObjectPointer(std::shared_ptr<Object> op) = 0;
		virtual std::shared_ptr<Object> objectPointer() = 0;
		virtual std::shared_ptr<Object> standardObjectPointer() = 0;

		static const std::string className() {
			return std::string("FInstance");
		}
	private:

	};

	/*!
	*	\class	FInstance
	*	\brief	Pointer of objects.
	*/
	template<typename T>
	class FInstance : public InstanceBase
	{
	public:
		typedef T					VarType;
		typedef FInstance<T>		FieldType;

		FInstance() : InstanceBase() {}

		FInstance(std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent)
			: InstanceBase(name, description, fieldType, parent) {}

		const std::string getTemplateName() override { return std::string(typeid(VarType).name()); }
		const std::string getClassName() final { return InstanceBase::className(); }

		std::shared_ptr<T> getDataPtr() {
			InstanceBase* ins = dynamic_cast<InstanceBase*>(this->getTopField());
			std::shared_ptr<T> data = std::static_pointer_cast<T>(ins->objectPointer());

			this->tick();

			return data;
		}

		std::shared_ptr<T> constDataPtr() {
			InstanceBase* ins = dynamic_cast<InstanceBase*>(this->getTopField());
			std::shared_ptr<T> data = std::static_pointer_cast<T>(ins->objectPointer());

			return data;
		}

		void setDataPtr(std::shared_ptr<T> sPtr)
		{
			InstanceBase* ins = dynamic_cast<InstanceBase*>(this->getTopField());
			ins->setObjectPointer(sPtr);

			this->tick();
		}

		std::shared_ptr<T> allocate() {
			if (mData == nullptr) {
				mData = std::make_shared<T>();
			}

			this->tick();

			return mData;
		}

		bool isEmpty() override {
			return this->constDataPtr() == nullptr;
		}

		bool connect(FBase* dst) override {
			InstanceBase* ins = dynamic_cast<InstanceBase*>(dst);
			if (ins == nullptr) {
				return false;
			}

			if (!ins->canBeConnectedBy(this)) {
				return false;
			}

			this->connectField(dst);
			return true;
		}

		T& getData() {
			auto dataPtr = this->getDataPtr();
			assert(dataPtr != nullptr);

			return *dataPtr;
		}

		uint size() override { return 1; }

	public:
		std::shared_ptr<Object> objectPointer() final {
			return std::dynamic_pointer_cast<Object>(mData);
		}

		std::shared_ptr<Object> standardObjectPointer() final {
			return std::make_shared<T>();
		}

		void setObjectPointer(std::shared_ptr<Object> op) final	{
			auto dPtr = std::dynamic_pointer_cast<T>(op);
			assert(dPtr != nullptr);

			mData = dPtr;
		}

		bool canBeConnectedBy(InstanceBase* ins) final {
			std::shared_ptr<Object> dataPtr = ins->isEmpty() ? ins->standardObjectPointer() : ins->objectPointer();
			auto dPtr = std::dynamic_pointer_cast<T>(dataPtr);

			return dPtr == nullptr ? false : true;
		}

	private:
		std::shared_ptr<T> mData = nullptr;
	};
}