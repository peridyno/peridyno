#include "Pipeline.h"
#include "Node.h"

#include <queue>
#include <set>

namespace dyno
{
Pipeline::Pipeline(Node* node)
	: Module()
{
	assert(node != nullptr);
	mNode = node;
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

void Pipeline::pushModule(std::shared_ptr<Module> m)
{
	mNode->addModule(m);

	ObjectId id = m->objectId();
	mModuleMap[id] = m.get();

	mModuleUpdated = true;
}

void Pipeline::pushPersistentModule(std::shared_ptr<Module> m)
{
	mNode->addModule(m);
	mPersistentModule.push_back(m.get());

	mModuleUpdated = true;
}

void Pipeline::preprocess()
{
	if (mModuleUpdated)
	{
		reconstructPipeline();
	}
}

void Pipeline::updateImpl()
{
	for each (auto m in mModuleList)
	{
		m->update();
	}
}

bool Pipeline::requireUpdate()
{
	return true;
}

void Pipeline::reconstructPipeline()
{
	mModuleList.clear();

	std::queue<Module*> moduleQueue;
	std::set<ObjectId> moduleSet;

	auto retrieveModules = [&](std::vector<FBase *>& fields) {
		for each (auto f in fields) {
			auto& sinks = f->getSinks();
			for each (auto sink in sinks)
			{
				Module* module = dynamic_cast<Module*>(sink->parent());
				if (module != nullptr)
				{
					ObjectId oId = module->objectId();

					if (moduleSet.find(oId) == moduleSet.end() && mModuleMap.count(oId) > 0)
					{
						moduleSet.insert(oId);
						moduleQueue.push(module);
					}
				}
			}
		}
	};

	auto& fields = mNode->getAllFields();
	retrieveModules(fields);

	while (!moduleQueue.empty())
	{
		Module* m = moduleQueue.front();

		mModuleList.push_back(m);

		auto& outFields = m->getOutputFields();
		retrieveModules(outFields);

		moduleQueue.pop();
	}

	for each (auto m in mPersistentModule)
	{
		mModuleList.push_back(m);
	}
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