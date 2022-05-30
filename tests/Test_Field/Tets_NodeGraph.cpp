#include "gtest/gtest.h"

#include "Node.h"

using namespace dyno;

class NodeA : public Node {
public:
	NodeA() {};
	~NodeA() override {};
};

class NodeB : public Node {
public:
	NodeB() {};
	~NodeB() override {};

	DEF_NODE_PORT(NodeA, Ancestor1, "");
};

class NodeC : public NodeB {
public:
	NodeC() {};
	~NodeC() override {};
};

class NodeD : public Node {
public:
	NodeD() {};
	~NodeD() override {};

	DEF_NODE_PORT(NodeA, Ancestor1, "");

	DEF_NODE_PORTS(Ancestor2, NodeB, "");
};

/**
 * @brief Test a node graph as follows
 *	 		A	 
 *		  /   \
 *		  \	   B   C
 *		   \  /   /
 *			 D
 */
TEST(NodeGraph, construct)
{
	std::shared_ptr<NodeA> na = std::make_shared<NodeA>();
	std::shared_ptr<NodeB> nb = std::make_shared<NodeB>();
	std::shared_ptr<NodeC> nc = std::make_shared<NodeC>();
	std::shared_ptr<NodeD> nd = std::make_shared<NodeD>();

// 	nd->setAncestor1(na);
// 	nd->addAncestor2(nb);
// 	nd->addAncestor2(nc);

	EXPECT_EQ(nd->sizeOfNodePorts() == 2, true);
	EXPECT_EQ(nd->getAncestor1() != nullptr, true);
	EXPECT_EQ(na->sizeofDescendants() == 1, true);

//	nb->setAncestor1(na);

	EXPECT_EQ(nd->sizeOfAncestors() == 3, true);
	EXPECT_EQ(nb->sizeOfAncestors() == 1, true);
	EXPECT_EQ(na->sizeofDescendants() == 2, true);

//	nb->setAncestor1(nullptr);
	EXPECT_EQ(na->sizeofDescendants() == 1, true);
	EXPECT_EQ(nb->sizeOfAncestors() == 0, true);
}
