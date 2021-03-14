#pragma once
#include "Typedef.h"
#include "Base.h"
#include "Array/Array.h"
#include "Field.h"
#include "Framework/Log.h"

namespace dyno {

template<typename T, DeviceType deviceType>
class ArrayField : public Field
{
public:
	typedef T VarType;
	typedef Array<T, deviceType> DataType;
	typedef ArrayField<T, deviceType> FieldClass;

	DEFINE_DERIVED_FUNC(FieldClass, DataType);

	ArrayField();
	ArrayField(int num, std::string name, std::string description, EFieldType fieldType, Base* parent);
	~ArrayField() override;

	inline size_t getElementCount() override {
		auto ref = this->getReference();
		return ref == nullptr ? 0 : ref->size();
	}

	void setElementCount(size_t num);

	const std::string getTemplateName() override { return std::string(typeid(T).name()); }
	const std::string getClassName() override { return std::string("ArrayBuffer"); }

	void setValue(std::vector<T>& vals);
	void setValue(GArray<T>& vals);
};

template<typename T, DeviceType deviceType>
ArrayField<T, deviceType>::ArrayField()
	: Field("", "")
{
}

template<typename T, DeviceType deviceType>
ArrayField<T, deviceType>::ArrayField(int num, std::string name, std::string description, EFieldType fieldType, Base* parent)
	: Field(name, description, fieldType, parent)
{
	m_data = num <= 0 ? nullptr : std::make_shared<Array<T, deviceType>>(num);	
}

template<typename T, DeviceType deviceType>
ArrayField<T, deviceType>::~ArrayField()
{
	if (m_data.use_count() == 1)
	{
		m_data->clear();
	}
}

template<typename T, DeviceType deviceType>
void ArrayField<T, deviceType>::setElementCount(size_t num)
{
	Field* topField = this->getTopField();
	ArrayField<T, deviceType>* derived = dynamic_cast<ArrayField<T, deviceType>*>(topField);

	if (derived->m_data == nullptr)
	{
		derived->m_data = std::make_shared<Array<T, deviceType>>(num);
	}
	else
	{
		derived->m_data->resize(num);
	}
}

template<typename T, DeviceType deviceType>
void ArrayField<T, deviceType>::setValue(std::vector<T>& vals)
{
	std::shared_ptr<Array<T, deviceType>>& data = this->getReference();
	if (data == nullptr)
	{
		data = std::make_shared<Array<T, deviceType>>();
	}
	
	data->assign(vals);
}

template<typename T, DeviceType deviceType>
void ArrayField<T, deviceType>::setValue(GArray<T>& vals)
{
	std::shared_ptr<Array<T, deviceType>>& data = this->getReference();
	if (data == nullptr)
	{
		data = std::make_shared<Array<T, deviceType>>();
	}

	data->assign(vals);
}

template<typename T>
using HostArrayField = ArrayField<T, DeviceType::CPU>;

template<typename T>
using DeviceArrayField = ArrayField<T, DeviceType::GPU>;
}