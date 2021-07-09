#include "ModuleGroup.h"
#include "Framework/Node.h"

#include <queue>
#include <set>

namespace dyno
{
	GroupModule::GroupModule()
		: Module()
	{
	}

	GroupModule::~GroupModule()
	{
	}

	void GroupModule::pushModule(std::shared_ptr<Module> m)
	{
		ObjectId id = m->objectId();
		if (mModuleMap.find(id) != mModuleMap.end())
			return;

		mModuleMap[id] = m;
		mModuleUpdated = true;
	}

	void GroupModule::preprocess()
	{
		if (mModuleUpdated) {
			reconstructPipeline();
			mModuleUpdated = false;
		}
	}

	void GroupModule::updateImpl()
	{
		for each (auto m in mModuleList) {
			m->update();
		}
	}

	void GroupModule::reconstructPipeline()
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

		auto& fields = this->getInputFields();
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
			mModuleList.push_back(m.get());
		}

		moduleSet.clear();
	}
}