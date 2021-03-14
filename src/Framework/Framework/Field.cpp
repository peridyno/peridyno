#include "Field.h"
#include <algorithm>

#include "Module.h"

namespace dyno
{
	void Field::setParent(Base* owner)
	{
		m_owner = owner;
	}

	Base* Field::getParent()
	{
		return m_owner;
	}

	void Field::setSource(Field* source)
	{
		m_derived = source == nullptr ? false : true;
		m_source = source;
	}

	Field* Field::getSource()
	{
		return m_source;
	}

	void Field::addSink(Field* f)
	{
		auto it = std::find(m_field_sink.begin(), m_field_sink.end(), f);

		if (it == m_field_sink.end())
		{
			m_field_sink.push_back(f);

//			f->setDerived(true);
			f->setSource(this);
		}
	}

	void Field::removeSink(Field* f)
	{
		auto it = std::find(m_field_sink.begin(), m_field_sink.end(), f);
		
		if (it != m_field_sink.end())
		{
			m_field_sink.erase(it);

//			f->setDerived(false);
			f->setSource(nullptr);
		}
	}

	bool Field::isDerived()
	{
		return m_derived;
	}

	bool Field::isAutoDestroyable()
	{
		return m_autoDestroyable;
	}

	void Field::setAutoDestroy(bool autoDestroy)
	{
		m_autoDestroyable = autoDestroy;
	}

	void Field::setDerived(bool derived)
	{
		m_derived = derived;
	}

	bool Field::connectField(Field* dst)
	{
		if (dst->getSource() != nullptr && dst->getSource() != this)
		{
			dst->getSource()->removeSink(dst);
		}

		this->addSink(dst);

		return true;
	}

	void Field::update()
	{
		if (!this->isEmpty() && callbackFunc != nullptr)
		{
			callbackFunc();
		}

		auto& sinks = this->getSinkFields();

		for each (auto var in sinks)
		{
			if (var != nullptr)
			{
				var->update();
			}
		}
	}

	Field* Field::fieldPtr()
	{
		return this;
	}

	bool Field::isModified()
	{
		return m_modified;
	}

	void Field::tagModified(bool modifed)
	{
		m_modified = modifed;
	}

	bool Field::isOptional()
	{
		return m_optional;
	}

	void Field::tagOptional(bool optional)
	{
		m_optional = optional;
	}

	Field::Field(std::string name, std::string description, FieldType type, Base* parent)
	{
		m_name = name; m_description = description;
		m_fType = type;
		if (parent != nullptr)
		{
			parent->attachField(this, name, description, false);
		}
	}

	FieldType Field::getFieldType()
	{
		return m_fType;
	}

}

