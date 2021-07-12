#include "FBase.h"
#include <algorithm>

#include "Module.h"

namespace dyno
{
	void FBase::setParent(OBase* owner)
	{
		mOwner = owner;
	}

	OBase* FBase::parent()
	{
		return mOwner;
	}

	void FBase::setSource(FBase* source)
	{
		m_derived = source == nullptr ? false : true;
		mSource = source;
	}

	FBase* FBase::getSource()
	{
		return mSource;
	}

	void FBase::addSink(FBase* f)
	{
		auto it = std::find(mSinks.begin(), mSinks.end(), f);

		if (it == mSinks.end())
		{
			mSinks.push_back(f);

//			f->setDerived(true);
			f->setSource(this);
		}
	}

	void FBase::removeSink(FBase* f)
	{
		auto it = std::find(mSinks.begin(), mSinks.end(), f);
		
		if (it != mSinks.end())
		{
			mSinks.erase(it);

//			f->setDerived(false);
			f->setSource(nullptr);
		}
	}

	bool FBase::isDerived()
	{
		return m_derived;
	}

	bool FBase::isAutoDestroyable()
	{
		return m_autoDestroyable;
	}

	void FBase::setAutoDestroy(bool autoDestroy)
	{
		m_autoDestroyable = autoDestroy;
	}

	void FBase::setDerived(bool derived)
	{
		m_derived = derived;
	}

	bool FBase::connectField(FBase* dst)
	{
		if (dst->getSource() != nullptr && dst->getSource() != this) {
			dst->getSource()->removeSink(dst);
		}

		this->addSink(dst);

		return true;
	}

	bool FBase::disconnectField(FBase* dst)
	{
		if (dst->getSource() == this) {
			dst->getSource()->removeSink(dst);
		}

		return true;
	}

	FBase* FBase::getTopField()
	{
		return mSource == nullptr ? this : mSource->getTopField();
	}

	void FBase::update()
	{
		if (!this->isEmpty() && callbackFunc != nullptr)
		{
			callbackFunc();
		}

		auto& sinks = this->getSinks();

		for each (auto var in sinks)
		{
			if (var != nullptr)
			{
				var->update();
			}
		}
	}

	bool FBase::isModified()
	{
		return m_modified;
	}

	void FBase::tagModified(bool modifed)
	{
		m_modified = modifed;
	}

	bool FBase::isOptional()
	{
		return m_optional;
	}

	void FBase::tagOptional(bool optional)
	{
		m_optional = optional;
	}

	FBase::FBase(std::string name, std::string description, FieldTypeEnum type, OBase* parent)
	{
		m_name = name; m_description = description;
		m_fType = type;
		if (parent != nullptr)
		{
			parent->attachField(this, name, description, false);
		}
	}

	FieldTypeEnum FBase::getFieldType()
	{
		return m_fType;
	}

}

