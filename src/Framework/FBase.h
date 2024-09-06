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
#include "TimeStamp.h"

#include <typeinfo>
#include <string>
#include <functional>
#include <cfloat>

namespace dyno {
	class OBase;
	class FCallBackFunc;

	enum FieldTypeEnum
	{
		In,
		Out,
		IO,
		Param,
		State,
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

	virtual uint size() = 0;
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

	void tick();
	void tack();

	bool isOptional();
	void tagOptional(bool optional);

	inline float getMin() { return m_min; }
	inline void setMin(float min_val) { m_min = min_val; }

	inline float getMax() { return m_max; }
	inline void setMax(float max_val) { m_max = max_val; }

	inline void setRange(float min_val, float max_val) { m_min = min_val; m_max = max_val; }

	FieldTypeEnum getFieldType();

	virtual bool connect(FBase* dst) = 0;
	virtual bool disconnect(FBase* dst);

	virtual std::string serialize() { return ""; }
	virtual bool deserialize(const std::string& str) { return false; }

	FBase* getTopField();
	FBase* getSource();

	/**
	 * @brief Display a state field as an ouput field
	 * 
	 * @return state field
	 */
	FBase* promoteOuput();

	/**
	 * @brief Display a state field as an input field
	 * 
	 * @return state field
	 */
	FBase* promoteInput();


	/**
	 * @brief Hide a state field from outputs
	 * 
	 * @return state field
	 */
	FBase* demoteOuput();

		/**
	 * @brief Hide a state field from inputs
	 * 
	 * @return state field
	 */
	FBase* demoteInput();

	virtual bool isEmpty() = 0;
	virtual void update();

	void attach(std::shared_ptr<FCallBackFunc> func);
	void detach(std::shared_ptr<FCallBackFunc> func);

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

	OBase* mOwner = nullptr;

	FBase* mSource = nullptr;

	std::vector<FBase*> mSinks;

	TimeStamp mTickTime;
	TimeStamp mTackTime;

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
	derived->tick();												\
	return derived->m_data;											\
}																	\
\
std::shared_ptr<Data>& constDataPtr()								\
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
bool connect(DerivedField* dst)										\
{																	\
	this->connectField(dst);										\
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
const Data& constData() {											\
	auto dataPtr = this->constDataPtr();							\
	assert(dataPtr != nullptr);										\
	return *dataPtr;												\
}																	\
private:															\
	std::shared_ptr<Data> m_data = nullptr;							\
public:

}