/**
 * Copyright 2021 Xiaowei He
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "OBase.h"
#include "Node.h"
#include "NodeIterator.h"

#include "Module/InputModule.h"

#include <mutex>

namespace dyno 
{
	typedef std::list<Node*> NodeList;

	typedef std::map<ObjectId, std::shared_ptr<Node>> NodeMap;

	class SceneGraph : public OBase
	{
	public:
		typedef NodeIterator Iterator;

		~SceneGraph();

// 		void setRootNode(std::shared_ptr<Node> root) { mRoot = root; }
// 		std::shared_ptr<Node> getRootNode() { return mRoot; }

		virtual bool initialize();
		bool isInitialized() { return mInitialized; }
		bool isEmpty() {
			return mNodeMap.size() == 0;
		}

		void invalid();

		virtual void advance(float dt);
		virtual void takeOneFrame();
		virtual void updateGraphicsContext();
		virtual void run();

		NBoundingBox boundingBox();

		void reset();

		void reset(std::shared_ptr<Node> node);

		void printNodeInfo(bool enabled);
		void printModuleInfo(bool enabled);

		bool isNodeInfoPrintable() { return mNodeTiming; }
		bool isModuleInfoPrintable() { return mModuleTiming; }

		virtual bool load(std::string name);

		virtual void invoke(unsigned char type, unsigned char key, int x, int y) {};

		template<class TNode, class ...Args>
		std::shared_ptr<TNode> createNewScene(Args&& ... args)
		{
			std::shared_ptr<TNode> root = TypeInfo::New<TNode>(std::forward<Args>(args)...);
			//mRoot = root;

			addNode(root);

			return root;
		}

		template<class TNode, class ...Args>
		std::shared_ptr<TNode> addNode(Args&& ... args)
		{
			std::shared_ptr<TNode> node = TypeInfo::New<TNode>(std::forward<Args>(args)...);

			return addNode(node);
		}

		template<class TNode>
		std::shared_ptr<TNode> addNode(std::shared_ptr<TNode> tNode)
		{
			if (tNode == nullptr ||
				mNodeMap.find(tNode->objectId()) != mNodeMap.end())
				return nullptr;
				
			mNodeMap[tNode->objectId()] = tNode;
			mQueueUpdateRequired = true;

			tNode->setSceneGraph(this);

			return tNode;
		}

		void deleteNode(std::shared_ptr<Node> node);

		void propagateNode(std::shared_ptr<Node> node);

	public:
		static SceneGraph& getInstance();

		inline void setTotalTime(float t) { mMaxTime = t; }
		inline float getTotalTime() { return mMaxTime; }

		inline void setFrameRate(float frameRate) { mFrameRate = frameRate; }
		inline float getFrameRate() { return mFrameRate; }
		inline float getTimeCostPerFrame() { return mFrameCost; }
		inline float getFrameInterval() { return 1.0f / mFrameRate; }

		inline int getFrameNumber() { return mFrameNumber; }
		inline void setFrameNumber(int n) { mFrameNumber = n; }
		
		bool isIntervalAdaptive();
		void setAdaptiveInterval(bool adaptive);

		void setGravity(Vec3f g);
		Vec3f getGravity();

		Vec3f getLowerBound();
		Vec3f getUpperBound();

		void setLowerBound(Vec3f lowerBound);
		void setUpperBound(Vec3f upperBound);

		inline Iterator begin() { 

			updateExecutionQueue();

			return NodeIterator(mNodeQueue, mNodeMap);
		}

		inline Iterator end() { return NodeIterator(); }

		/**
		 * @brief An interface to tell SceneGraph to update the execuation queue
		 */
		void markQueueUpdateRequired();

	public:
		void onMouseEvent(PMouseEvent event);

		void onKeyboardEvent(PKeyboardEvent event);

		/**
		 * @brief Depth-first tree traversal
		 *
		 * @param act 	Operation on the node
		 */
		void traverseBackward(Action* act);

		template<class Act, class ... Args>
		void traverseBackward(Args&& ... args) {
			Act action(std::forward<Args>(args)...);
			traverseBackward(&action);
		}

		/**
		 * @brief Breadth-first tree traversal
		 *
		 * @param act 	Operation on the node
		 */
		void traverseForward(Action* act);

		template<class Act, class ... Args>
		void traverseForward(Args&& ... args) {
			Act action(std::forward<Args>(args)...);
			traverseForward(&action);
		}

		/**
		 * @brief Breadth-first tree traversal starting from a specific node
		 *
		 * @param node  Root node
		 * @param act 	Operation on the node
		 */
		void traverseForward(std::shared_ptr<Node> node, Action* act);

		template<class Act, class ... Args>
		void traverseForward(std::shared_ptr<Node> node, Args&& ... args) {
			Act action(std::forward<Args>(args)...);
			traverseForward(node, &action);
		}

		/**
		 * @brief Breadth-first tree traversal starting from a specific node, only those whose mAutoSync turned-on will be visited.
		 *
		 * @param node  Root node
		 * @param act 	Operation on the node
		 */
		void traverseForwardWithAutoSync(std::shared_ptr<Node> node, Action* act);

		template<class Act, class ... Args>
		void traverseForwardWithAutoSync(std::shared_ptr<Node> node, Args&& ... args) {
			Act action(std::forward<Args>(args)...);
			traverseForwardWithAutoSync(node, &action);
		}

	protected:
		//void retriveExecutionQueue(std::list<Node*>& nQueue);

		void updateExecutionQueue();

	public:
		SceneGraph()
			: mElapsedTime(0)
			, mMaxTime(0)
			, mFrameRate(25)
			, mFrameNumber(0)
			, mFrameCost(0)
			, mInitialized(false)
			, mLowerBound(-1, -1, -1)
			, mUpperBound(1, 1, 1)
		{
			//mRoot = std::make_shared<Node>();
			mGravity = Vec3f(0.0f, -9.8f, 0.0f);
		};

		/**
		* To avoid erroneous operations
		*/
		SceneGraph(const SceneGraph&) = delete;
		SceneGraph& operator=(const SceneGraph&) = delete;

	private:
		bool mInitialized;
		bool mAdvativeInterval = true;

		float mElapsedTime;
		float mMaxTime;
		float mFrameRate;
		float mFrameCost;

		int mFrameNumber;

		Vec3f mGravity;

		Vec3f mLowerBound;
		Vec3f mUpperBound;

	private:
		//std::shared_ptr<Node> mRoot = nullptr;

		bool mQueueUpdateRequired = false;

		NodeMap mNodeMap;

		NodeList mNodeQueue;

		bool mNodeTiming = false;
		bool mModuleTiming = false;

		/**
		 * A  lock to guarantee consistency across threads
		 */
		std::mutex mSync;

	};

}