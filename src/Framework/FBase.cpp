#include "FBase.h"
#include <algorithm>

#include "Node.h"
#include "Module.h"

#include "FCallbackFunc.h"

namespace dyno
{
	std::string FormatConnectionInfo(FBase* fin, FBase* fout, bool connecting, bool succeeded)
	{
		OBase* pIn = fin != nullptr ? fin->parent() : nullptr;
		OBase* pOut = fout != nullptr ? fout->parent() : nullptr;

		std::string capIn = pIn != nullptr ? pIn->caption() : "";
		std::string capOut = pOut != nullptr ? pOut->caption() : "";

		std::string nameIn = pIn != nullptr ? pIn->getName() : "";
		std::string nameOut = pOut != nullptr ? pOut->getName() : "";

		if (connecting) 
		{
			std::string message1 = capIn + ":" + nameIn + " is connected to " + capOut + ":" + nameOut;
			std::string message2 = capIn + ":" + nameIn + " cannot be connected to " + capOut + ":" + nameOut;
			return succeeded ? message1 : message2;
		}
		else
		{
			std::string message1 = capIn + ":" + nameIn + " is disconnected from " + capOut + ":" + nameOut;
			std::string message2 = capIn + ":" + nameIn + " cannot be disconnected from " + capOut + ":" + nameOut;
			return succeeded ? message1 : message2;
		}
	}

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

	FBase* FBase::promoteOuput()
	{
		if (m_fType != FieldTypeEnum::State && mOwner == nullptr)
			return nullptr;

		if (!mOwner->findOutputField(this)) {
			mOwner->addToOutput(this);
		}

		return this;
	}

	FBase* FBase::promoteInput()
	{
		if (mOwner == nullptr)
			return nullptr;

		if (!mOwner->findInputField(this)) {
			mOwner->addInputField(this);
		}

		return this;
	}

	FBase* FBase::demoteOuput()
	{
		if (m_fType != FieldTypeEnum::State && mOwner == nullptr)
			return nullptr;

		if (mOwner->findOutputField(this)) {
			mOwner->removeFromOutput(this);
		}

		return this;
	}

	FBase* FBase::demoteInput()
	{
		if (mOwner == nullptr)
			return nullptr;

		if (mOwner->findInputField(this)) {
			mOwner->removeInputField(this);
		}

		return this;
	}

	void FBase::addSink(FBase* f)
	{
		auto it = std::find(mSinks.begin(), mSinks.end(), f);

		if (it == mSinks.end())
		{
			mSinks.push_back(f);

//			f->setDerived(true);
			f->setSource(this);

			Log::sendMessage(Log::Info, FormatConnectionInfo(this, f, true, true));

			return;
		}

		Log::sendMessage(Log::Info, FormatConnectionInfo(this, f, true, false));
	}

	bool FBase::removeSink(FBase* f)
	{
		auto it = std::find(mSinks.begin(), mSinks.end(), f);
		
		if (it != mSinks.end())
		{
			mSinks.erase(it);

//			f->setDerived(false);
			f->setSource(nullptr);

			Log::sendMessage(Log::Info, FormatConnectionInfo(this, f, false, true));

			return true;
		}

		Log::sendMessage(Log::Info, FormatConnectionInfo(this, f, false, false));

		return false;
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

		//If state is connected to an input field of the other node, the field should be exported
		Node* node = dynamic_cast<Node*>(dst->parent());
		if (m_fType == FieldTypeEnum::State && node !=nullptr)
		{
			this->promoteOuput();
		}
		
		// fprintf(stderr,"%s ----> %s\n",this->m_name.c_str(), dst->m_name.c_str());
		this->addSink(dst);

		this->update();

		return true;
	}

	bool FBase::disconnectField(FBase* dst)
	{
		return this->removeSink(dst);
	}

	bool FBase::disconnect(FBase* dst)
	{
		return this->disconnectField(dst);
	}

	FBase* FBase::getTopField()
	{
		return mSource == nullptr ? this : mSource->getTopField();
	}

	void FBase::update()
	{
		if (!this->isEmpty())
		{
			for(auto func : mCallbackFunc)
			{
				func->update();
			}
		}

		auto& sinks = this->getSinks();

		for(auto var : sinks)
		{
			if (var != nullptr)
			{
				var->update();
			}
		}
	}

	void FBase::attach(std::shared_ptr<FCallBackFunc> func)
	{
		//Add the current field as one of the input to the callback function
		func->addInput(this);

		mCallbackFunc.push_back(func);
	}

	void FBase::detach(std::shared_ptr<FCallBackFunc> func)
	{
		if (func == nullptr || mCallbackFunc.size() <= 0)
			return;

		auto it = std::find(mCallbackFunc.begin(), mCallbackFunc.end(), func);

		if (it != mCallbackFunc.end())
		{
			mCallbackFunc.erase(it);
		}
	}

	bool FBase::isModified()
	{
		FBase* topField = this->getTopField();

		return mTackTime < topField->mTickTime;
	}

	void FBase::tick()
	{
		FBase* topField = this->getTopField();

		topField->mTickTime.mark();
	}

	void FBase::tack()
	{
		this->mTackTime.mark();
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

	FBase::~FBase()
	{
		//Before deallocating data, fields should be disconnected first
		FBase* src = this->getSource();
		if (src != nullptr) {
			src->disconnectField(this);
		}

		while (!mSinks.empty()) {
			auto sink = mSinks.back();
			sink->setSource(nullptr);

			mSinks.pop_back();
		}

		mCallbackFunc.clear();
	}

	FieldTypeEnum FBase::getFieldType()
	{
		return m_fType;
	}

}

