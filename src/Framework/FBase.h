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
#include "Platform.h"
#include <typeinfo>
#include <string>
#include <functional>

namespace dyno {
	class OBase;
	class FCallBackFunc;

	enum FieldTypeEnum
	{
		In,
		Out,
		IO,
		Param,
		Current,
		Next
	};

/*!
*	\class	Variable
*	\brief	Interface for all variables.
*/
class FBase
{
public:
	FBase() : m_name("default"), m_description("") {};
	FBase(std::string name, std::string description, FieldTypeEnum type = FieldTypeEnum::Param, OBase* parent = nullptr);
	virtual ~FBase();

	virtual uint getElementCount() = 0;
	virtual const std::string getTemplateName() { return std::string(""); }
	virtual const std::string getClassName() { return std::string("Field"); }

	std::string	getObjectName() { return m_name; }
	std::string	getDescription() { return m_description; }
	virtual DeviceType getDeviceType() { return DeviceType::UNDEFINED; }

	void setObjectName(std::string name) { m_name = name; }
	void setDescription(std::string description) { m_description = description; }

	void setParent(OBase* owner);
	OBase* parent();

	bool isDerived();
	bool isAutoDestroyable();

	void setAutoDestroy(bool autoDestroy);
	void setDerived(bool derived);

	uint sizeOfSinks() { return (uint)mSinks.size(); }
	std::vector<FBase*>& getSinks() { return mSinks; }

	bool isModified();
	void tagModified(bool modifed);

	bool isOptional();
	void tagOptional(bool optional);

	inline float getMin() { return m_min; }
	inline void setMin(float min_val) { m_min = min_val; }

	inline float getMax() { return m_max; }
	inline void setMax(float max_val) { m_max = max_val; }

	FieldTypeEnum getFieldType();

	virtual bool connect(FBase* dst) = 0;
	virtual bool disconnect(FBase* dst);

	FBase* getTopField();
	FBase* getSource();

	FBase* promoteToOuput();
	FBase* promoteToInput();

	virtual bool isEmpty() = 0;
	virtual void update();

	void attach(std::shared_ptr<FCallBackFunc> func);

protected:
	void setSource(FBase* source);

	void addSink(FBase* f);
	bool removeSink(FBase* f);

	bool connectField(FBase* dst);
	bool disconnectField(FBase* dst);

	FieldTypeEnum m_fType = FieldTypeEnum::Param;

private:
	std::string m_name;
	std::string m_description;

	bool m_optional = false;

	bool m_autoDestroyable = true;
	bool m_derived = false;

	float m_min = -FLT_MAX;
	float m_max = FLT_MAX;

	bool m_modified = false;

	OBase* mOwner = nullptr;

	FBase* mSource = nullptr;

	std::vector<FBase*> mSinks;

	std::vector<std::shared_ptr<FCallBackFunc>> mCallbackFunc;
};

#define DEFINE_FIELD_FUNC(DerivedField, Data, FieldName)						\
FieldName() : FBase("", ""){}								\
\
FieldName(std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent)		\
	: FBase(name, description, fieldType, parent){}				\
\
const std::string getTemplateName() override { return std::string(typeid(VarType).name()); }			\
const std::string getClassName() override { return std::string(#FieldName); }					\
\
std::shared_ptr<Data>& getDataPtr()									\
{																	\
	FBase* topField = this->getTopField();						\
	DerivedField* derived = dynamic_cast<DerivedField*>(topField);	\
	return derived->m_data;											\
}																	\
\
std::shared_ptr<Data> allocate()									\
{																	\
	auto& data = this->getDataPtr();								\
	if (data == nullptr) {											\
		data = std::make_shared<Data>();							\
	}																\
	return data;													\
}																	\
\
bool isEmpty() override {											\
return this->getDataPtr() == nullptr;								\
}																	\
\
bool connect(DerivedField* dst)										\
{																	\
	this->connectField(dst);										\
	this->update();													\
	return true;													\
}																	\
bool connect(FBase* dst) override {									\
	DerivedField* derived = dynamic_cast<DerivedField*>(dst);		\
	if (derived == nullptr) return false;							\
	return this->connect(derived);									\
}																	\
Data& getData() {													\
	auto dataPtr = this->getDataPtr();								\
	assert(dataPtr != nullptr);										\
	return *dataPtr;												\
}																	\
private:															\
	std::shared_ptr<Data> m_data = nullptr;							\
public:

}