#pragma once
#include "Typedef.h"
#include "Array/Array.h"
#include "Array/MemoryManager.h"
#include "Utility.h"
#include "Field.h"
#include "Base.h"
#include "Framework/Log.h"

namespace dyno {

template<typename T, DeviceType deviceType>
class ArrayField : public Field
{
public:
	typedef T VarType;
	typedef Array<T, deviceType> DataType;

	ArrayField();
	ArrayField(int num);
	ArrayField(std::string name, std::string description, int num = 1);
	ArrayField(int num, std::string name, std::string description, FieldType fieldType, Base* parent);
	~ArrayField() override;

	inline size_t getElementCount() override {
		auto ref = this->getReference();
		return ref == nullptr ? 0 : ref->size();
	}

	void setElementCount(size_t num);
//	void resize(int num);
	const std::string getTemplateName() override { return std::string(typeid(T).name()); }
	const std::string getClassName() override { return std::string("ArrayBuffer"); }
//	DeviceType getDeviceType() override { return deviceType; }

	/**
	 * @brief Get the shared pointer of the contained array, it will return the its own pointer if no Field is driven this object.
	 * Otherwise, it will return the pointer its source contains.
	 * 
	 * @return std::shared_ptr<Array<T, deviceType>> 
	 */
	std::shared_ptr<Array<T, deviceType>> getReference();

	Array<T, deviceType>& getValue() { return *(getReference()); }
	void setValue(std::vector<T>& vals);
	void setValue(GArray<T>& vals);

//	void reset() override { m_data->reset(); }

	inline bool isEmpty() override {
		return getReference() == nullptr;
	}

	bool connect(ArrayField<T, deviceType>* field2);

	ArrayField<T, deviceType>* getSourceArrayField();

private:
	std::shared_ptr<Array<T, deviceType>> m_data = nullptr;
};

template<typename T, DeviceType deviceType>
ArrayField<T, deviceType>::ArrayField()
	: Field("", "")
	, m_data(nullptr)
{
//	m_data = std::make_shared<Array<T, deviceType>>();
}

// template<typename T, DeviceType deviceType>
// ArrayBuffer<T, deviceType>::ArrayBuffer(int num)
// 	: Field()
// 	, m_data(NULL)
// {
// 	m_data = new Array<T, deviceType>(num);
// }


template<typename T, DeviceType deviceType>
ArrayField<T, deviceType>::ArrayField(int num)
	: Field("", "")
{
	m_data = num <= 0 ? nullptr : std::make_shared<Array<T, deviceType>>(num);
}


template<typename T, DeviceType deviceType>
ArrayField<T, deviceType>::ArrayField(std::string name, std::string description, int num)
	: Field(name, description)
{
	m_data = num <= 0 ? nullptr : std::make_shared<Array<T, deviceType>>(num);
}

template<typename T, DeviceType deviceType>
ArrayField<T, deviceType>::ArrayField(int num, std::string name, std::string description, FieldType fieldType, Base* parent)
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
	auto arr = this->getSourceArrayField();
	if (arr == nullptr)
	{
		if (m_data != nullptr)
			m_data->resize(num);
		else
			m_data = num <= 0 ? nullptr : std::make_shared<Array<T, deviceType>>(num);
	}
	else
	{
		//if (arr->m_data != nullptr && arr->m_data->size() == num)
		//	return;
		if(arr->m_data != nullptr)
			arr->m_data->resize(num);
		else
			arr->m_data = num <= 0 ? nullptr : std::make_shared<Array<T, deviceType>>(num);
		
	}
}

template<typename T, DeviceType deviceType>
bool ArrayField<T, deviceType>::connect(ArrayField<T, deviceType>* field2)
{
	auto f = field2->fieldPtr();
	this->connectPtr(f);

	return true;
}

template<typename T, DeviceType deviceType>
void ArrayField<T, deviceType>::setValue(std::vector<T>& vals)
{
	std::shared_ptr<Array<T, deviceType>> data = getReference();
	if (data == nullptr)
	{
		m_data = std::make_shared<Array<T, deviceType>>();
		m_data->resize(vals.size());
		Function1Pt::copy(*m_data, vals);
		return;
	}
	else
	{
		if (vals.size() != data->size())
		{
			Log::sendMessage(Log::Error, "The input array size is not equal to Field " + this->getObjectName());
		}
		else
		{
			Function1Pt::copy(*data, vals);
		}
	}
}

template<typename T, DeviceType deviceType>
void ArrayField<T, deviceType>::setValue(GArray<T>& vals)
{
	std::shared_ptr<Array<T, deviceType>> data = getReference();
	if (data == nullptr)
	{
		m_data = std::make_shared<Array<T, deviceType>>();
		m_data->resize(vals.size());
		Function1Pt::copy(*m_data, vals);
		return;
	}
	else
	{
		if (vals.size() != data->size())
		{
			m_data->resize(vals.size());
		}
		
		Function1Pt::copy(*data, vals);
	}
}

template<typename T, DeviceType deviceType>
std::shared_ptr<Array<T, deviceType>> ArrayField<T, deviceType>::getReference()
{
	Field* source = getSource();
	if (source == nullptr)
	{
		return m_data;
	}
	else
	{
		ArrayField<T, deviceType>* var = dynamic_cast<ArrayField<T, deviceType>*>(source);
		if (var != nullptr)
		{
			return var->getReference();
		}
		else
		{
			return nullptr;
		}
	}
}

template<typename T, DeviceType deviceType>
ArrayField<T, deviceType>* ArrayField<T, deviceType>::getSourceArrayField()
{
	Field* source = getSource();
	if (source == nullptr)
	{
		return this;
	}
	else
	{
		ArrayField<T, deviceType>* var = dynamic_cast<ArrayField<T, deviceType>*>(source);
		if (var == nullptr)
			return nullptr;
		else
			return var->getSourceArrayField();
	}
}

template<typename T>
using HostArrayField = ArrayField<T, DeviceType::CPU>;

template<typename T>
using DeviceArrayField = ArrayField<T, DeviceType::GPU>;
}