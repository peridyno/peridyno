#include "GroupModule.h"
#include "Node.h"
#include "DirectedAcyclicGraph.h"

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
		ObjectId baseId = Object::baseId();

		mModuleList.clear();

		std::queue<Module*> moduleQueue;
		std::set<ObjectId> moduleSet;

		DirectedAcyclicGraph graph;

		auto retrieveModules = [&](ObjectId id, std::vector<FBase *>& fields) {
			for each (auto f in fields) {
				auto& sinks = f->getSinks();
				for each (auto sink in sinks)
				{
					Module* module = dynamic_cast<Module*>(sink->parent());
					if (module != nullptr)
					{
						ObjectId oId = module->objectId();
						graph.addEdge(id, oId);

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
		retrieveModules(baseId, fields);

		for each (auto m in mPersistentModule)
		{
			moduleQueue.push(m.get());
		}

// 		auto retrieveModules = [&](std::vector<FBase *>& fields) {
// 		};

		while (!moduleQueue.empty())
		{
			Module* m = moduleQueue.front();

			auto& outFields = m->getOutputFields();
			retrieveModules(m->objectId(), outFields);

			moduleQueue.pop();
		}

		auto& ids = graph.topologicalSort();

		for each (auto id in ids)
		{
			if (mModuleMap.count(id) > 0)
			{
				mModuleList.push_back(mModuleMap[id].get());
			}
		}

		moduleSet.clear();
	}
}