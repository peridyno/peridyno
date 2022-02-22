#include "SceneGraph.h"
//#include "Action/ActDraw.h"
#include "Action/ActReset.h"
#include "Action/ActNodeInfo.h"
#include "Action/ActPostProcessing.h"
#include "SceneLoaderFactory.h"

namespace dyno
{
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
			AdvanceAct(float dt, float t) {
				mDt = dt; 
				mElapsedTime = t;
			};

			void start(Node* node) override {
				if (node == NULL)
					return;

				node->varTimeStep()->setValue(mDt);
				node->varElapsedTime()->setValue(mElapsedTime);
			}

			void process(Node* node) override {
				if (node == NULL)
				{
					Log::sendMessage(Log::Error, "Node is invalid!");
					return;
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
			}

			float mDt;
			float mElapsedTime;
		};	

		this->traverseForward<AdvanceAct>(dt, mElapsedTime);

		mElapsedTime += dt;
	}

	void SceneGraph::takeOneFrame()
	{
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
				dt = std::min(node->getDt(), dt);
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

		std::cout << "----------------    Frame " << mFrameNumber << " Ended      ----------------" << std::endl << std::endl;

		mFrameNumber++;
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

		this->traverseForward<UpdateGrpahicsContextAct>();
	}

	void SceneGraph::run()
	{

	}

	void SceneGraph::reset()
	{
// 		if (mRoot == nullptr)
// 		{
// 			return;
// 		}

		this->traverseForward<ResetAct>();

		//m_root->traverseBottomUp();
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
					auto m = dynamic_cast<InputMouseModule*>(iter);
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
			if (exNode != nullptr && !visited[node->objectId()]) {
				DFS(exNode, nodeQueue, visited);
			}
		}

		auto outFields = node->getOutputFields();
		for each (auto f in outFields) {
			auto& sinks = f->getSinks();
			for each (auto sink in sinks) {
				if (sink != nullptr) {
					auto exNode = dynamic_cast<Node*>(sink->parent());
					if (exNode != nullptr && !visited[node->objectId()]) {
						DFS(exNode, nodeQueue, visited);
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