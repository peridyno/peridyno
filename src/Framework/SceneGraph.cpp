#include "SceneGraph.h"
//#include "Action/ActDraw.h"
#include "Action/ActReset.h"
#include "Action/ActNodeInfo.h"
#include "Action/ActPostProcessing.h"

#include "Module/VisualModule.h"

#include "Module/MouseInputModule.h"
#include "Module/KeyboardInputModule.h"

#include "DirectedAcyclicGraph.h"

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

	void SceneGraph::setAsynchronousSimulation(bool b)
	{
		mAsynchronousSimulation = b;
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

				CTimer timer;

				if (mTiming) {
					timer.start();
				}

				node->update();

				if (mTiming) {
					timer.stop();

					std::stringstream name;
					std::stringstream ss;
					name << std::setw(40) << node->getClassInfo()->getClassName();
					ss << std::setprecision(10) << timer.getElapsedTime();

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

	void SceneGraph::advanceInAsync()
	{
		class AsyncAdvanceAct : public Action
		{
		public:
			AsyncAdvanceAct(bool bTiming = false) {
				mTiming = bTiming;
			};

			void start(Node* node) override {
				if (node == NULL)
					return;

				auto dt = node->getDt();
				node->stateTimeStep()->setValue(dt);
			}

			void process(Node* node) override {
				if (node == NULL)
				{
					Log::sendMessage(Log::Error, "Node is invalid!");
					return;
				}

				CTimer timer;

				if (mTiming) {
					timer.start();
				}

				node->update();

				if (mTiming) {
					timer.stop();

					std::stringstream name;
					std::stringstream ss;
					name << std::setw(40) << node->getClassInfo()->getClassName();
					ss << std::setprecision(10) << timer.getElapsedTime();

					std::string info = "Node: \t" + name.str() + ": \t " + ss.str() + "ms \n";
					Log::sendMessage(Log::Info, info);
				}
			}

			void end(Node* node) override {
				if (node == NULL)
					return;

				auto dt = node->stateTimeStep()->getValue();
				auto time = node->stateElapsedTime()->getValue();
				node->stateElapsedTime()->setValue(time + dt);
			}

			bool mTiming = false;
		};

		this->traverseForward<AsyncAdvanceAct>(mNodeTiming);
	}

	void SceneGraph::takeOneFrame()
	{
		std::cout << "****************    Frame " << mFrameNumber << " Started    ****************" << std::endl;
		
		CTimer timer;
		timer.start();

		if (mAsynchronousSimulation)
		{
			this->advanceInAsync();
		}
		else
		{
			float t = 0.0f;
			float dt = 0.0f;

			class QueryTimeStep : public Action
			{
			public:
				void process(Node* node) override {
					if (node != nullptr && node->isActive())
						dt = node->getDt() < dt ? node->getDt() : dt;
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
		}

		this->traverseForward<PostProcessing>();

		this->traverseForward<AssignFrameNumberAct>(mFrameNumber);

		timer.stop();

		std::cout << "----------------    Frame " << mFrameNumber << " Ended! ( " << timer.getElapsedTime() << " ms in Total)  ----------------" << std::endl << std::endl;

		mFrameNumber++;

		mWorkMode = RUNNING_MODE;
	}

	void SceneGraph::updateGraphicsContext()
	{
		class UpdateGrpahicsContextAct : public Action
		{
		public:
			void process(Node* node) override {
				if (node->isVisible())
				{
					node->updateGraphicsContext();
				}
			}
		};

		this->traverseForward<UpdateGrpahicsContextAct>();
	}


	void SceneGraph::run()
	{

	}

	NBoundingBox SceneGraph::boundingBox()
	{
		NBoundingBox box;
		for (auto it = this->begin(); it != this->end(); it++)
		{
			box.join(it->boundingBox());
		}

		return box;
	}

	void SceneGraph::reset()
	{
		class ResetNodeAct : public Action
		{
		public:
			void process(Node* node) override {
				if (node == NULL) {
					Log::sendMessage(Log::Error, "Node is invalid!");
					return;
				}

				node->reset();
			}
		};

		this->traverseForward<ResetNodeAct>();

		mElapsedTime = 0.0f;
		mFrameNumber = 0;

		mWorkMode = EDIT_MODE;
	}

	void SceneGraph::reset(std::shared_ptr<Node> node)
	{
		this->traverseForward<ResetAct>(node);
	}

	void SceneGraph::printNodeInfo(bool enabled)
	{
		mNodeTiming = enabled;
	}

	void SceneGraph::printSimulationInfo(bool enabled)
	{
		mSimulationTiming = enabled;
	}

	void SceneGraph::printRenderingInfo(bool enabled)
	{
		mRenderingTiming = enabled;
	}

	void SceneGraph::printValidationInfo(bool enabled)
	{
		mValidationInfo = enabled;
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

	void SceneGraph::onMouseEvent(PMouseEvent event, std::shared_ptr<Node> node)
	{
		if (node == nullptr || !node->isVisible())
			return;

		for (auto iter : node->animationPipeline()->activeModules())
		{
			auto m = dynamic_cast<MouseInputModule*>(iter.get());
			if (m)
			{
				m->enqueueEvent(event);
			}
		}

		for (auto iter : node->graphicsPipeline()->activeModules())
		{
			auto m = dynamic_cast<MouseInputModule*>(iter.get());
			if (m)
			{
				m->enqueueEvent(event);
			}
		}
	}

	void SceneGraph::onKeyboardEvent(PKeyboardEvent event)
	{
		class KeyboardEventAct : public Action
		{
		public:
			KeyboardEventAct(PKeyboardEvent event) { mKeyboardEvent = event; }
			~KeyboardEventAct() override {}

		private:
			void process(Node* node) override
			{
				if (!node->isVisible())
					return;

				for (auto iter : node->animationPipeline()->activeModules())
				{
					auto m = dynamic_cast<KeyboardInputModule*>(iter.get());
					if (m)
					{
						m->enqueueEvent(mKeyboardEvent);
					}
				}

				for (auto iter : node->graphicsPipeline()->activeModules())
				{
					auto m = dynamic_cast<KeyboardInputModule*>(iter.get());
					if (m)
					{
						m->enqueueEvent(mKeyboardEvent);
					}
				}
			}

			PKeyboardEvent mKeyboardEvent;
		};

		KeyboardEventAct eventAct(event);

		this->traverseForward(&eventAct);
	}

// 	//Used to traverse the whole scene graph
// 	void DFS(Node* node, NodeList& nodeQueue, std::map<ObjectId, bool>& visited) {
// 
// 		visited[node->objectId()] = true;
// 
// 		auto imports = node->getImportNodes();
// 		for (auto port : imports) {
// 			auto& inNodes = port->getNodes();
// 			for (auto inNode : inNodes) {
// 				if (inNode != nullptr && !visited[inNode->objectId()]) {
// 					DFS(inNode, nodeQueue, visited);
// 				}
// 			}
// 		}
// 
// 		auto inFields = node->getInputFields();
// 		for (auto f : inFields) {
// 			auto* src = f->getSource();
// 			if (src != nullptr) {
// 				auto* inNode = dynamic_cast<Node*>(src->parent());
// 				if (inNode != nullptr && !visited[inNode->objectId()]) {
// 					DFS(inNode, nodeQueue, visited);
// 				}
// 			}
// 		}
// 
// 		nodeQueue.push_back(node);
// 
// 		auto exports = node->getExportNodes();
// 		for (auto port : exports) {
// 			auto exNode = port->getParent();
// 			if (exNode != nullptr && !visited[exNode->objectId()]) {
// 				DFS(exNode, nodeQueue, visited);
// 			}
// 		}
// 
// 		auto outFields = node->getOutputFields();
// 		for (auto f : outFields) {
// 			auto& sinks = f->getSinks();
// 			for (auto sink : sinks) {
// 				if (sink != nullptr) {
// 					auto exNode = dynamic_cast<Node*>(sink->parent());
// 					if (exNode != nullptr && !visited[exNode->objectId()]) {
// 						DFS(exNode, nodeQueue, visited);
// 					}
// 				}
// 			}
// 		}
// 	};

	//Used to traverse the scene graph from a specific node
	void BFS(Node* node, NodeList& list, std::map<ObjectId, bool>& visited) {

		visited[node->objectId()] = true;

		std::queue<Node*> queue;
		queue.push(node);

		while (!queue.empty())
		{
			auto fn = queue.front();
			queue.pop();

			list.push_back(fn);

			auto exports = fn->getExportNodes();
			for (auto port : exports) {
				auto next = port->getParent();
				if (next != nullptr && !visited[next->objectId()]) {
					queue.push(next);
				}
			}

			auto outFields = fn->getOutputFields();
			for (auto f : outFields) {
				auto& sinks = f->getSinks();
				for (auto sink : sinks) {
					if (sink != nullptr) {
						auto next = dynamic_cast<Node*>(sink->parent());
						if (next != nullptr && !visited[next->objectId()]) {
							queue.push(next);
						}
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

		DirectedAcyclicGraph dag;

		auto dummyId = Object::baseId();

		for (auto& n : mNodeMap) {
			Node* node = n.second.get();

			auto outId = node->objectId();

			bool hasAncestor = false;

			//Input nodes
			auto imports = node->getImportNodes();
			for (auto port : imports) {
				auto& inNodes = port->getNodes();
				for (auto inNode : inNodes) {
					if (inNode != nullptr) {
						dag.addEdge(inNode->objectId(), outId);

						hasAncestor = true;
					}
				}
			}

			//Input fields
			auto inFields = node->getInputFields();
			for (auto f : inFields) {
				auto* src = f->getSource();
				if (src != nullptr) {
					auto* inNode = dynamic_cast<Node*>(src->parent());
					if (inNode != nullptr) {
						dag.addEdge(inNode->objectId(), outId);

						hasAncestor = true;
					}
				}
			}

			//In case of an isolated node, add an virtual edge to connect a dummy node and the node
			if (!hasAncestor)
			{
				dag.addEdge(dummyId, outId);
			}
		}

		auto& sortedIds = dag.topologicalSort();

		for (auto id : sortedIds)
		{
			if (id != dummyId)
				mNodeQueue.push_back(mNodeMap[id].get());
		}

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

	//Used to traverse the scene graph from a specific node
	void BFSWithAutoSync(Node* node, NodeList& list, std::map<ObjectId, bool>& visited) {

		visited[node->objectId()] = true;

		std::queue<Node*> queue;
		queue.push(node);

		while (!queue.empty())
		{
			auto fn = queue.front();
			queue.pop();

			list.push_back(fn);

			auto exports = fn->getExportNodes();
			for (auto port : exports) {
				auto next = port->getParent();
				if (next != nullptr && next->isAutoSync() && !visited[next->objectId()]) {
					queue.push(next);
				}
			}

			auto outFields = fn->getOutputFields();
			for (auto f : outFields) {
				auto& sinks = f->getSinks();
				for (auto sink : sinks) {
					if (sink != nullptr) {
						auto next = dynamic_cast<Node*>(sink->parent());
						if (next != nullptr && next->isAutoSync() && !visited[next->objectId()]) {
							queue.push(next);
						}
					}
				}
			}
		}
	};

	void SceneGraph::traverseForwardWithAutoSync(std::shared_ptr<Node> node, Action* act)
	{
		std::map<ObjectId, bool> visited;
		for (auto& nm : mNodeMap) {
			visited[nm.first] = false;
		}

		NodeList list;
		BFSWithAutoSync(node.get(), list, visited);

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

	void SceneGraph::propagateNode(std::shared_ptr<Node> node)
	{
		std::map<ObjectId, bool> visited;
		for (auto it = mNodeQueue.begin(); it != mNodeQueue.end(); ++it)
		{
			visited[(*it)->objectId()] = false;
		}

		//DownwardDFS(node.get(), visited);

		visited.clear();
	}

}