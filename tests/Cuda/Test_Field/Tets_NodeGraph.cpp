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

	DEF_NODE_PORTS(NodeB, Ancestor2, "");
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

 	na->connect(nd->importAncestor1());
	nb->connect(nd->importAncestor2s());
	nc->connect(nd->importAncestor2s());

 	EXPECT_EQ(nd->sizeOfNodePorts() == 2, true);
	EXPECT_EQ(nd->sizeOfImportNodes() == 3, true);
	EXPECT_EQ(nd->sizeOfExportNodes() == 0, true);
	EXPECT_EQ(na->sizeOfExportNodes() == 1, true);

	na->connect(nb->importAncestor1());

 	EXPECT_EQ(na->sizeOfExportNodes() == 2, true);
 	EXPECT_EQ(nb->sizeOfImportNodes() == 1, true);
}
