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
#include <atomic>
#include <mutex>

#include <map>

#include "SceneGraph.h"

namespace dyno 
{
	class NodeAction
	{
	public:
		NodeAction(std::string caption, std::string icon, std::function<std::shared_ptr<Node>()> func = nullptr) {
			mCaption = caption;
			mIcon = icon;

			mCallback = func;
		}

		std::string icon() { return mIcon; }
		std::string caption() { return mCaption; }

		std::function<std::shared_ptr<Node>()> action() { return mCallback; }

	private:
		std::string mCaption;
		std::string mIcon;

		std::function<std::shared_ptr<Node>()> mCallback;
	};

	class NodeGroup
	{
	public:
		NodeGroup(std::string caption, std::string icon) {
			mCaption = caption;
			mIcon = icon;
		}

		void addAction(std::string caption, std::string icon, std::function<std::shared_ptr<Node>()> act);

		std::vector<std::shared_ptr<NodeAction>>& actions() {
			return mActions;
		}

		std::string icon() { return mIcon; }
		std::string caption() { return mCaption; }

	private:
		std::string mCaption;
		std::string mIcon;

		std::vector<std::shared_ptr<NodeAction>> mActions;
	};

	class NodeFactory
	{
	public:
		static NodeFactory* instance();

		std::shared_ptr<NodeGroup> addGroup(std::string groupName, std::string caption, std::string icon);

		std::map<std::string, std::shared_ptr<NodeGroup>>& nodeGroups() {
			return mGroups;
		}

	private:
		NodeFactory() = default;
		~NodeFactory() = default;
		NodeFactory(const NodeFactory&) = delete;
		NodeFactory& operator=(const NodeFactory&) = delete;

	private:
		static std::atomic<NodeFactory*> pInstance;
		static std::mutex mMutex;

		std::map<std::string, std::shared_ptr<NodeGroup>> mGroups;
	};

}