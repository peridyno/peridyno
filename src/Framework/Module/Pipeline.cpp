#include "Pipeline.h"
#include "Node.h"
#include "DirectedAcyclicGraph.h"

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
	
	void Pipeline::clear()
	{
// 		for (auto const &pair : mModuleMap)
// 		{
// 			mNode->deleteModule(std::shared_ptr<Module>(pair.second));
// 		}

		mModuleList.clear();
		mPersistentModule.clear();
		mModuleMap.clear();
		
		mModuleUpdated = true;
	}

	void Pipeline::pushPersistentModule(std::shared_ptr<Module> m)
	{
		mNode->addModule(m);
		mPersistentModule.push_back(m.get());

		mModuleUpdated = true;
	}

	void Pipeline::enable()
	{
		mUpdateEnabled = true;
	}

	void Pipeline::disable()
	{
		mUpdateEnabled = false;
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
		if (mUpdateEnabled)
		{
			for each (auto m in mModuleList)
			{
				m->update();
			}
		}
	}

	bool Pipeline::requireUpdate()
	{
		return true;
	}

	void Pipeline::reconstructPipeline()
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

		auto& fields = mNode->getAllFields();
		retrieveModules(baseId, fields);

		for each (auto m in mPersistentModule)
		{
			moduleQueue.push(m);
		}

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
				mModuleList.push_back(mModuleMap[id]);
			}
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