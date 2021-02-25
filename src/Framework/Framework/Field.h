#pragma once
#include "Platform.h"
#include <typeinfo>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "Typedef.h"

namespace dyno {
	class Base;

	enum FieldType
	{
		In,
		Out,
		Param,
		Current,
		Next
	};

/*!
*	\class	Variable
*	\brief	Interface for all variables.
*/
class Field
{
public:
	Field() : m_name("default"), m_description("") {};
	Field(std::string name, std::string description, FieldType type = FieldType::Param, Base* parent = nullptr);
	virtual ~Field() {};

	virtual size_t getElementCount() { return 0; }
	virtual const std::string getTemplateName() { return std::string(""); }
	virtual const std::string getClassName() { return std::string("Field"); }

	FieldType getFieldType();

	std::string	getObjectName() { return m_name; }
	std::string	getDescription() { return m_description; }
	virtual DeviceType getDeviceType() { return DeviceType::UNDEFINED; }

	void setObjectName(std::string name) { m_name = name; }
	void setDescription(std::string description) { m_description = description; }

	void setParent(Base* owner);
	Base* getParent();

	bool isDerived();
	bool isAutoDestroyable();

	virtual bool isEmpty() = 0;

	void setAutoDestroy(bool autoDestroy);
	void setDerived(bool derived);

	bool connectPtr(Field* field2);

	Field* fieldPtr();

	std::vector<Field*>& getSinkFields() { return m_field_sink; }

	bool isModified();
	void tagModified(bool modifed);

	bool isOptional();
	void tagOptional(bool optional);

	inline float getMin() { return m_min; }
	inline void setMin(float min_val) { m_min = min_val; }

	inline float getMax() { return m_max; }
	inline void setMax(float max_val) { m_max = max_val; }

protected:
	void setSource(Field* source);
	Field* getSource();

	void addSink(Field* f);
	void removeSink(Field* f);

	FieldType m_fType = FieldType::Param;

private:
	std::string m_name;
	std::string m_description;

	bool m_optional = false;

	bool m_autoDestroyable = true;
	bool m_derived = false;
	Field* m_source = nullptr;
	Base* m_owner = nullptr;

	float m_min = -FLT_MAX;
	float m_max = FLT_MAX;

	bool m_modified = false;

	std::vector<Field*> m_field_sink;
};

}