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
		mModuleList.clear();
		mPersistentModule.clear();
		mModuleMap.clear();
	}

	uint Pipeline::sizeOfDynamicModules()
	{
		return (uint)mModuleList.size();
	}

	uint Pipeline::sizeOfPersistentModules()
	{
		return (uint)mPersistentModule.size();
	}

	void Pipeline::pushModule(std::shared_ptr<Module> m)
	{
		ObjectId id = m->objectId();
		if (mModuleMap.find(id) != mModuleMap.end())
			return;

		mModuleUpdated = true;
		mModuleMap[id] = m.get();

		mNode->addModule(m);
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
			mModuleUpdated = false;
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

		moduleSet.clear();
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