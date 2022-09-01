#include "SceneGraph.h"
//#include "Action/ActDraw.h"
#include "Action/ActReset.h"
#include "Action/ActNodeInfo.h"
#include "Action/ActPostProcessing.h"
#include "SceneLoaderFactory.h"

#include "Timer.h"

#include <sstream>
#include <iomanip>

namespace dyno
{
	class AssignFrameNumberAct : public Action
	{
	public:
		AssignFrameNumberAct(uint num) {
			mFrameNumber = num;
		};

		void process(Node* node) override {
			if (node == NULL)
				return;

			node->stateFrameNumber()->setValue(mFrameNumber);
		}

		uint mFrameNumber;
	};

	SceneGraph& SceneGraph::getInstance()
	{
		static SceneGraph m_instance;
		return m_instance;
	}

	bool SceneGraph::isIntervalAdaptive()
	{
		return mAdvativeInterval;
	}

	void SceneGraph::setAdaptiveInterval(bool adaptive)
	{
		mAdvativeInterval = adaptive;
	}

	void SceneGraph::setGravity(Vec3f g)
	{
		mGravity = g;
	}

	Vec3f SceneGraph::getGravity()
	{
		return mGravity;
	}

	SceneGraph::~SceneGraph()
	{
		mNodeMap.clear();
		mNodeQueue.clear();
	}

	bool SceneGraph::initialize()
	{
		if (mInitialized)
		{
			return true;
		}
 		//TODO: check initialization
// 		if (mRoot == nullptr)
// 		{
// 			return false;
// 		}

		class InitAct : public Action
		{
		public:
			void process(Node* node) override {
				node->initialize();

				auto& list = node->getModuleList();
				std::list<std::shared_ptr<Module>>::iterator iter = list.begin();
				for (; iter != list.end(); iter++)
				{
					(*iter)->initialize();
				}
				
				node->graphicsPipeline()->update();
			}
		};

		this->traverseForward<InitAct>();
		mInitialized = true;

		return mInitialized;
	}

	void SceneGraph::invalid()
	{
		mInitialized = false;
	}

	void SceneGraph::advance(float dt)
	{
		class AdvanceAct : public Action
		{
		public:
			AdvanceAct(float dt, float t, bool bTiming = false) {
				mDt = dt; 
				mElapsedTime = t;
				mTiming = bTiming;
			};

			void start(Node* node) override {
				if (node == NULL)
					return;

				node->stateTimeStep()->setValue(mDt);
				node->stateElapsedTime()->setValue(mElapsedTime);
			}

			void process(Node* node) override {
				if (node == NULL)
				{
					Log::sendMessage(Log::Error, "Node is invalid!");
					return;
				}

				GTimer timer;
				if (mTiming) {
					timer.start();
				}

				if (node->isActive())
				{
					auto customModules = node->getCustomModuleList();
					for (std::list<std::shared_ptr<CustomModule>>::iterator iter = customModules.begin(); iter != customModules.end(); iter++)
					{
						(*iter)->update();
					}

					node->update();
				}

				if (mTiming) {
					timer.stop();

					std::stringstream name;
					std::stringstream ss;
					name << std::setw(40) << node->getClassInfo()->getClassName();
					ss << std::setprecision(10) << timer.getEclipsedTime();
					
					std::string info = "Node: \t" + name.str() + ": \t " + ss.str() + "ms \n";
					Log::sendMessage(Log::Info, info);
				}
			}

			float mDt;
			float mElapsedTime;

			bool mTiming = false;
		};	

		this->traverseForward<AdvanceAct>(dt, mElapsedTime, mNodeTiming);

		mElapsedTime += dt;
	}

	void SceneGraph::takeOneFrame()
	{
		mSync.lock();

		std::cout << "****************    Frame " << mFrameNumber << " Started    ****************" << std::endl;

// 		if (mRoot == nullptr)
// 		{
// 			return;
// 		}

		float t = 0.0f;
		float dt = 0.0f;

		class QueryTimeStep : public Action
		{
		public:
			void process(Node* node) override {
				dt = min(node->getDt(), dt);
			}

			float dt;
		} timeStep;

		timeStep.dt = 1.0f / mFrameRate;

		this->traverseForward(&timeStep);
		dt = timeStep.dt;

		if (mAdvativeInterval)
		{
			this->advance(dt);
		}
		else
		{
			float interval = 1.0f / mFrameRate;
			while (t + dt < interval)
			{
				this->advance(dt);

				t += dt;
				timeStep.dt = 1.0f / mFrameRate;
				this->traverseForward(&timeStep);
				dt = timeStep.dt;
			}

			this->advance(interval - t);
		}

// 		class UpdateGrpahicsContextAct : public Action
// 		{
// 		public:
// 			void process(Node* node) override {
// 				node->graphicsPipeline()->update();
// 			}
// 		};
// 
// 		m_root->traverseTopDown<UpdateGrpahicsContextAct>();

		this->traverseForward<PostProcessing>();

		this->traverseForward<AssignFrameNumberAct>(mFrameNumber);

		std::cout << "----------------    Frame " << mFrameNumber << " Ended      ----------------" << std::endl << std::endl;

		mFrameNumber++;

		mSync.unlock();
	}

	void SceneGraph::updateGraphicsContext()
	{
		class UpdateGrpahicsContextAct : public Action
		{
		public:
			void process(Node* node) override {
				node->graphicsPipeline()->update();
			}
		};

		if (mSync.try_lock())
		{
			this->traverseForward<UpdateGrpahicsContextAct>();

			mSync.unlock();
		}
		
	}

	void SceneGraph::run()
	{

	}

	void SceneGraph::reset()
	{
		mSync.lock();

		this->traverseForward<ResetAct>();

		this->mFrameNumber = 0;

		mSync.unlock();
	}

	void SceneGraph::reset(std::shared_ptr<Node> node)
	{
		mSync.lock();

		this->traverseForward<ResetAct>(node);

		mSync.unlock();
	}

	void SceneGraph::printNodeInfo(bool enabled)
	{
		mNodeTiming = enabled;
	}

	void SceneGraph::printModuleInfo(bool enabled)
	{
		mModuleTiming = enabled;
	}

	bool SceneGraph::load(std::string name)
	{
		SceneLoader* loader = SceneLoaderFactory::getInstance().getEntryByFileName(name);
		if (loader)
		{
			//mRoot = loader->load(name);
			return true;
		}

		return false;
	}

	Vec3f SceneGraph::getLowerBound()
	{
		return mLowerBound;
	}

	Vec3f SceneGraph::getUpperBound()
	{
		return mUpperBound;
	}

	void SceneGraph::setLowerBound(Vec3f lowerBound)
	{
		mLowerBound = lowerBound;
	}

	void SceneGraph::setUpperBound(Vec3f upperBound)
	{
		mUpperBound = upperBound;
	}

	void SceneGraph::markQueueUpdateRequired()
	{
		mQueueUpdateRequired = true;
	}

	void SceneGraph::onMouseEvent(PMouseEvent event)
	{
		class MouseEventAct : public Action
		{
		public:
			MouseEventAct(PMouseEvent event) { mMouseEvent = event; }
			~MouseEventAct() override {}

		private:
			void process(Node* node) override
			{
				if (!node->isVisible())
					return;

				for (auto iter : node->animationPipeline()->activeModules())
				{
					auto m = dynamic_cast<MouseInputModule*>(iter.get());
					if (m)
					{
						m->enqueueEvent(mMouseEvent);
					}
				}

				for (auto iter : node->graphicsPipeline()->activeModules())
				{
					auto m = dynamic_cast<MouseInputModule*>(iter.get());
					if (m)
					{
						m->enqueueEvent(mMouseEvent);
					}
				}
			}

			PMouseEvent mMouseEvent;
		};

		MouseEventAct eventAct(event);

		this->traverseForward(&eventAct);
	}

	void SceneGraph::onKeyboardEvent(PKeyboardEvent event)
	{

	}

	//Used to traverse the whole scene graph
	void DFS(Node* node, NodeList& nodeQueue, std::map<ObjectId, bool>& visited) {

		visited[node->objectId()] = true;

		auto imports = node->getImportNodes();
		for (auto port : imports) {
			auto& inNodes = port->getNodes();
			for (auto inNode :  inNodes) {
				if (inNode != nullptr && !visited[inNode->objectId()]) {
					DFS(inNode, nodeQueue, visited);
				}
			}
		}

		auto inFields = node->getInputFields();
		for each (auto f in inFields) {
			auto* src = f->getSource();
			if (src != nullptr)	{
				auto* inNode = dynamic_cast<Node*>(src->parent());
				if (inNode != nullptr && !visited[inNode->objectId()]) {
					DFS(inNode, nodeQueue, visited);
				}
			}
		}

		nodeQueue.push_back(node);

		auto exports = node->getExportNodes();
		for (auto port : exports) {
			auto exNode = port->getParent();
			if (exNode != nullptr && !visited[exNode->objectId()]) {
				DFS(exNode, nodeQueue, visited);
			}
		}

		auto outFields = node->getOutputFields();
		for each (auto f in outFields) {
			auto& sinks = f->getSinks();
			for each (auto sink in sinks) {
				if (sink != nullptr) {
					auto exNode = dynamic_cast<Node*>(sink->parent());
					if (exNode != nullptr && !visited[exNode->objectId()]) {
						DFS(exNode, nodeQueue, visited);
					}
				}
			}
		}
	};

	//Used to traverse the scene graph from a specific node
	void BFS(Node* node, NodeList& nodeQueue, std::map<ObjectId, bool>& visited) {

		visited[node->objectId()] = true;

		nodeQueue.push_back(node);

		auto exports = node->getExportNodes();
		for (auto port : exports) {
			auto exNode = port->getParent();
			if (exNode != nullptr && !visited[node->objectId()]) {
				BFS(exNode, nodeQueue, visited);
			}
		}

		auto outFields = node->getOutputFields();
		for each (auto f in outFields) {
			auto& sinks = f->getSinks();
			for each (auto sink in sinks) {
				if (sink != nullptr) {
					auto exNode = dynamic_cast<Node*>(sink->parent());
					if (exNode != nullptr && !visited[exNode->objectId()]) {
						BFS(exNode, nodeQueue, visited);
					}
				}
			}
		}
	};

// 	void SceneGraph::retriveExecutionQueue(std::list<Node*>& nQueue)
// 	{
// 		nQueue.clear();
// 
// 		std::map<ObjectId, bool> visited;
// 		for (auto& nm : mNodeMap) {
// 			visited[nm.first] = false;
// 		}
// 
// 		for (auto& n : mNodeMap) {
// 			if (!visited[n.first]) {
// 
// 				Node* node = n.second.get();
// 
// 				DFS(node, nQueue, visited);
// 			}
// 		}
// 
// 		visited.clear();
// 	}

	void SceneGraph::updateExecutionQueue()
	{
		if (!mQueueUpdateRequired)
			return;

		mNodeQueue.clear();

		std::map<ObjectId, bool> visited;
		for (auto& nm : mNodeMap) {
			visited[nm.first] = false;
		}

		for (auto& n : mNodeMap) {
			if (!visited[n.first])	{

				Node* node = n.second.get();

				DFS(node, mNodeQueue, visited);
			}
		}

		visited.clear();

		mQueueUpdateRequired = false;
	}

	void SceneGraph::traverseBackward(Action* act)
	{
		updateExecutionQueue();

		for (auto rit = mNodeQueue.rbegin(); rit != mNodeQueue.rend(); ++rit)
		{
			Node* node = *rit;

			act->start(node);
			act->process(node);
			act->end(node);
		}
	}

	void SceneGraph::traverseForward(Action* act)
	{
		updateExecutionQueue();

		for (auto it = mNodeQueue.begin(); it != mNodeQueue.end(); ++it)
		{
			Node* node = *it;

			act->start(node);
			act->process(node);
			act->end(node);
		}
	}

	void SceneGraph::traverseForward(std::shared_ptr<Node> node, Action* act)
	{
		std::map<ObjectId, bool> visited;
		for (auto& nm : mNodeMap) {
			visited[nm.first] = false;
		}

		NodeList list;
		BFS(node.get(), list, visited);

		for (auto it = list.begin(); it != list.end(); ++it)
		{
			Node* node = *it;

			act->start(node);
			act->process(node);
			act->end(node);
		}

		list.clear();
		visited.clear();
	}

	void SceneGraph::deleteNode(std::shared_ptr<Node> node)
	{
		if (node == nullptr ||
			mNodeMap.find(node->objectId()) == mNodeMap.end())
			return;

		mNodeMap.erase(node->objectId());
		mQueueUpdateRequired = true;
	}

	void DownwardDFS(Node* node, std::map<ObjectId, bool>& visited) {

		visited[node->objectId()] = true;
		node->update();

		auto exports = node->getExportNodes();
		for (auto port : exports) {
			auto exNode = port->getParent();
			if (exNode != nullptr && !visited[node->objectId()]) {
				DownwardDFS(exNode, visited);
			}
		}
	};

	void SceneGraph::propagateNode(std::shared_ptr<Node> node)
	{
		std::map<ObjectId, bool> visited;
		for (auto it = mNodeQueue.begin(); it != mNodeQueue.end(); ++it)
		{
			visited[(*it)->objectId()] = false;
		}

		DownwardDFS(node.get(), visited);

		visited.clear();
	}

}