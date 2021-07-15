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
#include <string>
#include <vector>
#include <memory>

namespace dyno {

	class Node;


	enum NodePortType
	{
		Single,
		Multiple,
		Unknown
	};

	/*!
	*	\class	NodePort
	*	\brief	Input ports for Node.
	*/
	class NodePort
	{
	public:
		NodePort(std::string name, std::string description, Node* parent = nullptr);
		virtual ~NodePort() { m_nodes.clear(); };

		virtual std::string getPortName() { return m_name; };

		NodePortType getPortType();

		void setPortType(NodePortType portType);

		virtual std::vector<std::shared_ptr<Node>>& getNodes() { return m_nodes; }

		virtual bool addNode(std::shared_ptr<Node> node) = 0;

		virtual bool removeNode(std::shared_ptr<Node> node) = 0;

		virtual bool isKindOf(std::shared_ptr<Node> node) = 0;

		inline Node* getParent() { return m_parent; }

		virtual void clear();

	protected:
		bool addNodeToParent(std::shared_ptr<Node> node);

		bool removeNodeFromParent(std::shared_ptr<Node> node);

		std::vector<std::shared_ptr<Node>> m_nodes;

	private:

		Node* m_parent = nullptr;

		std::string m_name;
		std::string m_description;
		NodePortType m_portType;
	};


	template<typename T>
	class SingleNodePort : NodePort
	{
	public:
		SingleNodePort(std::string name, std::string description, Node* parent = nullptr)
			: NodePort(name, description, parent)
		{
			this->setPortType(NodePortType::Single);
			this->getNodes().resize(1);
		};
		~SingleNodePort() override { m_nodes[0] = nullptr; }

		bool addNode(std::shared_ptr<Node> node) override
		{ 
			auto d_node = std::dynamic_pointer_cast<T>(node);
			if (d_node != nullptr)
			{
				if (m_nodes[0] != node)
				{
					if (m_nodes[0] != nullptr)
					{
						this->removeNodeFromParent(m_nodes[0]);
					}
				
					this->addNodeToParent(node);
					m_nodes[0] = node;

					return true;
				}
			}

			return false;
		}

		bool removeNode(std::shared_ptr<Node> node) override
		{
			m_nodes[0] = nullptr;

			return true;
		}

		bool isKindOf(std::shared_ptr<Node> node) override
		{
			return nullptr != std::dynamic_pointer_cast<T>(node);
		}

		std::vector<std::shared_ptr<Node>>& getNodes() override
		{
			if (m_nodes.size() != 1)
				m_nodes.resize(1);

			m_nodes[0] = std::dynamic_pointer_cast<Node>(m_derived_node);

			return m_nodes;
		}

		inline std::shared_ptr<T> getDerivedNode()
		{
			return m_derived_node;
		}

		bool setDerivedNode(std::shared_ptr<T> d_node) {
			if (d_node != nullptr)
			{
				if (m_derived_node != nullptr) {
					this->removeNodeFromParent(std::dynamic_pointer_cast<Node>(m_derived_node));
					m_derived_node = nullptr;
				}

				this->addNodeToParent(std::dynamic_pointer_cast<Node>(d_node));
				m_derived_node = d_node;
				return true;
			}
			else
			{
				if (m_derived_node != nullptr)
				{
					this->removeNodeFromParent(std::dynamic_pointer_cast<Node>(m_derived_node));
					m_derived_node = nullptr;
				}
			}

			return false;
		}

	private:
		std::shared_ptr<T> m_derived_node;
	};


	template<typename T>
	class MultipleNodePort : NodePort
	{
	public:
		MultipleNodePort(std::string name, std::string description, Node* parent = nullptr)
			: NodePort(name, description, parent) 
		{
			this->setPortType(NodePortType::Multiple);
		};

		~MultipleNodePort() { m_derived_nodes.clear(); }

		void clear() override
		{
			m_derived_nodes.clear();

			NodePort::clear();
		}

		bool addNode(std::shared_ptr<Node> node) override {
			auto d_node = std::dynamic_pointer_cast<T>(node);
			if (d_node != nullptr)
			{
				auto it = find(m_derived_nodes.begin(), m_derived_nodes.end(), d_node);

				if (it == m_derived_nodes.end())
				{
					m_derived_nodes.push_back(d_node);

					this->addNodeToParent(node);

					return true;
				}
			}

			return false;
		}

		bool addDerivedNode(std::shared_ptr<T> d_node) {
			if (d_node != nullptr)
			{
				auto it = find(m_derived_nodes.begin(), m_derived_nodes.end(), d_node);

				if (it == m_derived_nodes.end())
				{
					m_derived_nodes.push_back(d_node);

					this->addNodeToParent(std::dynamic_pointer_cast<Node>(d_node));

					return true;
				}
			}

			return false;
		}

		bool removeNode(std::shared_ptr<Node> node)  override
		{
			auto d_node = std::dynamic_pointer_cast<T>(node);

			if (d_node != nullptr)
			{
				auto it = find(m_derived_nodes.begin(), m_derived_nodes.end(), d_node);

				if (it != m_derived_nodes.end())
				{
					m_derived_nodes.erase(it);
					this->removeNodeFromParent(node);

					return true;
				}
			}
		
			return false;
		}

		bool removeDerivedNode(std::shared_ptr<T> d_node)
		{
			if (d_node != nullptr)
			{
				auto it = find(m_derived_nodes.begin(), m_derived_nodes.end(), d_node);

				if (it != m_derived_nodes.end())
				{
					m_derived_nodes.erase(it);
					this->removeNodeFromParent(std::dynamic_pointer_cast<Node>(d_node));

					return true;
				}
			}

			return false;
		}

		bool isKindOf(std::shared_ptr<Node> node) override
		{
			return nullptr != std::dynamic_pointer_cast<T>(node);
		}

		std::vector<std::shared_ptr<Node>>& getNodes() override
		{
			m_nodes.resize(m_derived_nodes.size());
			for (int i = 0; i < m_nodes.size(); i++)
			{
				m_nodes[i] = std::dynamic_pointer_cast<Node>(m_derived_nodes[i]);
			}
			return m_nodes;
		}

		inline std::vector<std::shared_ptr<T>>& getDerivedNodes()
		{
			return m_derived_nodes;
		}
	private:
		std::vector<std::shared_ptr<T>> m_derived_nodes;
	};
}