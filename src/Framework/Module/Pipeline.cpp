#include "Pipeline.h"
#include "Node.h"
#include "SceneGraph.h"
#include "DirectedAcyclicGraph.h"

#include "Timer.h"

#include <sstream>
#include <iomanip>
#include <queue>
#include <set>

namespace dyno
{
	Pipeline::Pipeline(Node* node)
		: Module()
	{
		assert(node != nullptr);
		this->setParentNode(node);
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
		mModuleMap[id] = m;
		
		this->getParentNode()->addModule(m);
	}
	
	void Pipeline::popModule(std::shared_ptr<Module> m)
	{
		ObjectId id = m->objectId();

		mModuleMap.erase(id);
		this->getParentNode()->deleteModule(m);

		mModuleUpdated = true;
	}

	void Pipeline::clear()
	{
		//TODO: fix the memeory leak
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
		this->getParentNode()->addModule(m);
		mPersistentModule.push_back(m);

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

	void Pipeline::updateExecutionQueue()
	{
		reconstructPipeline();
	}

	void Pipeline::forceUpdate()
	{
		mModuleUpdated = true;

		this->update();
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
		if (mUpdateEnabled)
		{
#ifdef CUDA_BACKEND
			GTimer timer;
#else
			CTimer timer;
#endif // CUDA_BACKEND
			for(auto m : mModuleList)
			{
				if (this->printDebugInfo()) {
					timer.start();
				}

				//update the module
				m->update();

				if (this->printDebugInfo()) {
					timer.stop();

					std::stringstream name;
					std::stringstream ss;
					name << std::setw(40) << m->getClassInfo()->getClassName();
					ss << std::setprecision(10) << timer.getElapsedTime();

					std::string info = "\t Module: " + name.str() + ": \t " + ss.str() + "ms";
					Log::sendMessage(Log::Info, info);
				}
			}
		}
	}

	bool Pipeline::requireUpdate()
	{
		return true;
	}

	bool Pipeline::printDebugInfo()
	{
		return true;
	}

	FBase* Pipeline::promoteOutputToNode(FBase* base)
	{
		if (this->getParentNode() != nullptr && base->getFieldType() != FieldTypeEnum::Out)
			return nullptr;

		this->getParentNode()->addOutputField(base);

		return base;
	}

	void Pipeline::demoteOutputFromNode(FBase* base)
	{
		if (this->getParentNode() != nullptr && base->getFieldType() != FieldTypeEnum::Out)
			return;

		this->getParentNode()->removeOutputField(base);
	}

	void Pipeline::reconstructPipeline()
	{
		ObjectId baseId = Object::baseId();

		mModuleList.clear();

		std::queue<Module*> moduleQueue;
		std::set<ObjectId> moduleSet;

// 		std::map<ObjectId, bool> visited;
// 
// 		visited[baseId] = false;
// 		for (auto& mItor : mModuleMap) {
// 			visited[mItor.first] = false;
// 		}

		DirectedAcyclicGraph graph;

		auto retrieveModules = [&](ObjectId id, std::vector<FBase *>& fields) {
			for(auto f : fields) {
				auto& sinks = f->getSinks();
				for(auto sink : sinks)
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

		auto& fields = this->getParentNode()->getAllFields();
		retrieveModules(baseId, fields);

// 		for each (auto m in mPersistentModule)
// 		{
// 			moduleQueue.push(m.get());
// 		}

		auto flushQueue = [&]()
		{
			while (!moduleQueue.empty())
			{
				Module* m = moduleQueue.front();

				auto& outFields = m->getOutputFields();
				retrieveModules(m->objectId(), outFields);

				moduleQueue.pop();
			}
		};

		flushQueue();

// 		while (!moduleQueue.empty())
// 		{
// 			Module* m = moduleQueue.front();
// 
// 			auto& outFields = m->getOutputFields();
// 			retrieveModules(m->objectId(), outFields);
// 
// 			moduleQueue.pop();
// 		}

// 		while (!moduleQueue.empty())
// 		{
// 			Module* m = moduleQueue.front();
// 
// 			auto& outFields = m->getOutputFields();
// 			retrieveModules(m->objectId(), outFields);
// 
// 			moduleQueue.pop();
// 		}

		for (auto m : mModuleMap) {
			ObjectId oId = m.second->objectId();
			if (moduleSet.find(oId) == moduleSet.end())
			{
				moduleSet.insert(oId);
				moduleQueue.push(m.second.get());

				flushQueue();
			}
		}

		auto& ids = graph.topologicalSort();

		for(auto id : ids)
		{
			if (mModuleMap.count(id) > 0)
			{
				mModuleList.push_back(mModuleMap[id]);
			}
		}

		moduleSet.clear();

		mModuleUpdated = false;
	}
}