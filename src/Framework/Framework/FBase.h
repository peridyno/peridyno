#pragma once
#include "Platform.h"
#include <typeinfo>
#include <string>
#include <functional>

namespace dyno {
	class Base;

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
	using CallBackFunc = std::function<void()>;
public:
	FBase() : m_name("default"), m_description("") {};
	FBase(std::string name, std::string description, FieldTypeEnum type = FieldTypeEnum::Param, Base* parent = nullptr);
	virtual ~FBase() {};

	virtual uint getElementCount() = 0;
	virtual const std::string getTemplateName() { return std::string(""); }
	virtual const std::string getClassName() { return std::string("Field"); }

	std::string	getObjectName() { return m_name; }
	std::string	getDescription() { return m_description; }
	virtual DeviceType getDeviceType() { return DeviceType::UNDEFINED; }

	void setObjectName(std::string name) { m_name = name; }
	void setDescription(std::string description) { m_description = description; }

	void setParent(Base* owner);
	Base* parent();

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


	bool connectField(FBase* dst);

	FBase* getTopField();

	virtual bool isEmpty() = 0;
	virtual void update();

	void setCallBackFunc(CallBackFunc func) { callbackFunc = func; }
protected:
	void setSource(FBase* source);
	FBase* getSource();

	void addSink(FBase* f);
	void removeSink(FBase* f);

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

	Base* mOwner = nullptr;

	FBase* mSource = nullptr;

	std::vector<FBase*> mSinks;

	CallBackFunc callbackFunc;
};

#define DEFINE_FIELD_FUNC(DerivedField, Data, FieldName)						\
FieldName() : FBase("", ""){}								\
\
FieldName(std::string name, std::string description, FieldTypeEnum fieldType, Base* parent)		\
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
Data& getData() {													\
	auto dataPtr = this->getDataPtr();								\
	assert(dataPtr != nullptr);										\
	return *dataPtr;												\
}																	\
private:															\
	std::shared_ptr<Data> m_data = nullptr;							\
public:

}