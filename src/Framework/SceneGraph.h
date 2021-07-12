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
#include "Base.h"
#include "Node.h"
#include "NodeIterator.h"

namespace dyno 
{
	class SceneGraph : public Base
	{
	public:
		typedef NodeIterator Iterator;

		~SceneGraph() {};

		void setRootNode(std::shared_ptr<Node> root) { m_root = root; }
		std::shared_ptr<Node> getRootNode() { return m_root; }

		virtual bool initialize();
		bool isInitialized() { return mInitialized; }
		void invalid();

		virtual void advance(float dt);
		virtual void takeOneFrame();
		virtual void run();

		void reset();

		virtual bool load(std::string name);

		virtual void invoke(unsigned char type, unsigned char key, int x, int y) {};

		template<class TNode, class ...Args>
		std::shared_ptr<TNode> createNewScene(Args&& ... args)
		{
			std::shared_ptr<TNode> root = TypeInfo::New<TNode>(std::forward<Args>(args)...);
			m_root = root;
			return root;
		}

	public:
		static SceneGraph& getInstance();

		inline void setTotalTime(float t) { mMaxTime = t; }
		inline float getTotalTime() { return mMaxTime; }

		inline void setFrameRate(float frameRate) { mFrameRate = frameRate; }
		inline float getFrameRate() { return mFrameRate; }
		inline float getTimeCostPerFrame() { return mFrameCost; }
		inline float getFrameInterval() { return 1.0f / mFrameRate; }
		inline int getFrameNumber() { return mFrameNumber; }

		bool isIntervalAdaptive();
		void setAdaptiveInterval(bool adaptive);

		void setGravity(Vec3f g);
		Vec3f getGravity();

		Vec3f getLowerBound();
		Vec3f getUpperBound();

		void setLowerBound(Vec3f lowerBound);
		void setUpperBound(Vec3f upperBound);

		inline Iterator begin() { return NodeIterator(m_root); }
		inline Iterator end() { return NodeIterator(nullptr); }

	public:
		SceneGraph()
			: mElapsedTime(0)
			, mMaxTime(0)
			, mFrameRate(25)
			, mFrameNumber(0)
			, mFrameCost(0)
			, mInitialized(false)
			, mLowerBound(0, 0, 0)
			, mUpperBound(1, 1, 1)
		{
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
		std::shared_ptr<Node> m_root = nullptr;
	};

}