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

		virtual std::vector<Node*>& getNodes() { return m_nodes; }


		virtual bool isKindOf(Node* node) = 0;

		inline Node* getParent() { return m_parent; }

		virtual void clear();

	protected:
		virtual bool addNode(Node* node) = 0;

		virtual bool removeNode(Node* node) = 0;

		std::vector<Node*> m_nodes;

	private:

		Node* m_parent = nullptr;

		std::string m_name;
		std::string m_description;
		NodePortType m_portType;

		friend class Node;
	};


	template<typename T>
	class SingleNodePort : public NodePort
	{
	public:
		SingleNodePort(std::string name, std::string description, Node* parent = nullptr)
			: NodePort(name, description, parent)
		{
			this->setPortType(NodePortType::Single);
			this->getNodes().resize(1);
		};
		~SingleNodePort() override { m_nodes[0] = nullptr; }

		bool isKindOf(Node* node) override
		{
			return nullptr != dynamic_cast<T*>(node);
		}

		std::vector<Node*>& getNodes() override
		{
			if (m_nodes.size() != 1)
				m_nodes.resize(1);

			m_nodes[0] = dynamic_cast<Node*>(m_derived_node);

			return m_nodes;
		}

		inline T* getDerivedNode()
		{
			return m_derived_node;
		}

		bool setDerivedNode(T* d_node) {
			if (d_node != nullptr)
			{
				if (m_derived_node != nullptr) {
					m_derived_node = nullptr;
				}

				m_derived_node = d_node;
				m_nodes[0] = d_node;

				return true;
			}
			else
			{
				if (m_derived_node != nullptr)
				{
					//this->removeNodeFromParent(dynamic_cast<Node*>(m_derived_node));
					m_derived_node = nullptr;
					m_nodes[0] = nullptr;
				}
			}

			return false;
		}

	protected:
		bool addNode(Node* node) override
		{
			auto d_node = dynamic_cast<T*>(node);
			if (d_node != nullptr)
			{
				return setDerivedNode(d_node);
			}

			return false;
		}

		bool removeNode(Node* node) override
		{
			m_nodes[0] = nullptr;
			m_derived_node = nullptr;

			return true;
		}

	private:
		T* m_derived_node = nullptr;
	};


	template<typename T>
	class MultipleNodePort : public NodePort
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

		bool addDerivedNode(T* d_node) {
			if (d_node != nullptr)
			{
				auto it = find(m_derived_nodes.begin(), m_derived_nodes.end(), d_node);

				if (it == m_derived_nodes.end())
				{
					m_derived_nodes.push_back(d_node);

					//this->addNodeToParent(dynamic_cast<Node*>(d_node));

					return true;
				}
			}

			return false;
		}

		bool removeDerivedNode(T* d_node)
		{
			if (d_node != nullptr)
			{
				auto it = find(m_derived_nodes.begin(), m_derived_nodes.end(), d_node);

				if (it != m_derived_nodes.end())
				{
					m_derived_nodes.erase(it);
					//this->removeNodeFromParent(dynamic_cast<Node*>(d_node));

					return true;
				}
			}

			return false;
		}

		bool isKindOf(Node* node) override
		{
			return nullptr != dynamic_cast<T*>(node);
		}

		std::vector<Node*>& getNodes() override
		{
			m_nodes.clear();
			m_nodes.resize(m_derived_nodes.size());
			for (int i = 0; i < m_nodes.size(); i++)
			{
				m_nodes[i] = dynamic_cast<Node*>(m_derived_nodes[i]);
			}
			return m_nodes;
		}

		inline std::vector<T*>& getDerivedNodes()
		{
			return m_derived_nodes;
		}

	protected:
		bool addNode(Node* node) override {
			auto d_node = dynamic_cast<T*>(node);
			if (d_node != nullptr)
			{
				addDerivedNode(d_node);
			}

			return false;
		}

		bool removeNode(Node* node)  override
		{
			auto d_node = dynamic_cast<T*>(node);

			if (d_node != nullptr)
			{
				return removeDerivedNode(d_node);
			}

			return false;
		}

	private:
		std::vector<T*> m_derived_nodes;
	};
}