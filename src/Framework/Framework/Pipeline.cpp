#include "Pipeline.h"

namespace dyno
{
IMPLEMENT_CLASS(Pipeline)

Pipeline::Pipeline()
{
}

Pipeline::~Pipeline()
{
}

Pipeline::Iterator Pipeline::entry()
{
	Iterator iter;
	iter.module = start_module;

	return iter;
}

Pipeline::Iterator Pipeline::finished()
{
	Iterator iter;
	iter.module = end_module;

	return iter;
}

unsigned int Pipeline::size()
{
	return num;
}

void Pipeline::push_back(std::weak_ptr<Module> m)
{
	if (start_module.lock() == nullptr)
	{
		start_module = m;
		current_module = m;
	}
	else
	{
		current_module.lock()->setNext(m);
		current_module = m;
	}

	num++;
}

void Pipeline::addModule(Module* m)
{
	ObjectId id = m->objectId();
	moduleMap[id] = m;

	mModuleUpdated = true;
}

void Pipeline::preprocess()
{

}

void Pipeline::updateImpl()
{
	for each (auto m in moduleList)
	{
		m->update();
	}
}

bool Pipeline::requireUpdate()
{
	return mModuleUpdated;
}

ModuleIterator::ModuleIterator()
{

}


ModuleIterator::ModuleIterator(const ModuleIterator &iterator)
{
	module = iterator.module;
}

ModuleIterator& ModuleIterator::operator++()
{
	module = module.lock()->next();

	return *this;
}

ModuleIterator& ModuleIterator::operator++(int)
{
	return operator++();
}

Module* ModuleIterator::operator->()
{
	return module.lock().get();
}

Module* ModuleIterator::get()
{
	return module.lock().get();
}

std::shared_ptr<Module> ModuleIterator::operator*()
{
	auto m = module.lock();
	return m;
}

bool ModuleIterator::operator!=(const ModuleIterator &iterator) const
{
	return module.lock() != iterator.module.lock();
}

bool ModuleIterator::operator==(const ModuleIterator &iterator) const
{
	return module.lock() == iterator.module.lock();
}

ModuleIterator& ModuleIterator::operator=(const ModuleIterator &iterator)
{
	module = iterator.module;
	return *this;
}

ModuleIterator::~ModuleIterator()
{

}

}