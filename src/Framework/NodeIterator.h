/**
 * Copyright 2017-2021 Xiaowei He
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
#include <stack>
#include <list>
#include <map>
#include <memory>

#include "Object.h"

namespace dyno
{

	class Node;

	class NodeIterator
	{
	public:
		NodeIterator();
		NodeIterator(std::list<Node*>& nList, std::map<ObjectId, std::shared_ptr<Node>>& nMap);

		~NodeIterator();
		
		bool operator== (const NodeIterator &iterator) const;
		bool operator!= (const NodeIterator &iterator) const;

		NodeIterator& operator++();
		NodeIterator& operator++(int);

		std::shared_ptr<Node> operator->() const;

		std::shared_ptr<Node> get() const;

	protected:
		std::shared_ptr<Node> node_current = nullptr;

		std::stack<std::shared_ptr<Node>> node_stack;

		std::list<std::shared_ptr<Node>> mNodeList;

		friend class Node;
	};

}

