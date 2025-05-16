#include "WtInteraction.h"

WtInteraction::WtInteraction(WtNode& node, WtConnection& connection, WtFlowScene& scene, connectionPointData inPoint, connectionPointData outPoint, std::shared_ptr<Node> inNode, std::shared_ptr<Node> outNode)
	: _node(&node)
	, _connection(&connection)
	, _scene(&scene)
	, _inPoint(inPoint)
	, _outPoint(outPoint)
	, _inNode(inNode)
	, _outNode(outNode)

{}

WtInteraction::WtInteraction(WtNode& node, WtConnection& connection, connectionPointData inPoint, connectionPointData outPoint, std::shared_ptr<dyno::Module> inModule, std::shared_ptr<dyno::Module> outModule)
	: _node(&node)
	, _connection(&connection)
	, _inPoint(inPoint)
	, _outPoint(outPoint)
	, _inModule(inModule)
	, _outModule(outModule)

{
}

bool WtInteraction::canConnect(PortIndex& portIndex, TypeConverter& converter)
{
	// 1) QtConnection requires a port
	PortType requiredPort = connectionRequiredPort();

	if (requiredPort == PortType::None)
		return false;

	// 1.5) Forbid connecting the node to itself
	WtNode* nodeStart = _connection->getNode(oppositePort(requiredPort));

	if (nodeStart == _node)
		return false;

	// 2) connection point is on top of the node port
	portIndex = _inPoint.portIndex;

	if (portIndex == INVALID_PORT)
		return false;

	// 3) WtNode port is vacant
	if (!isNodePortAccessible(requiredPort, portIndex))
		return false;

	WtNode* nodeExp = requiredPort == PortType::In ? nodeStart : _node;
	WtNode* nodeInp = requiredPort == PortType::In ? _node : nodeStart;

	PortIndex portIndexExp = requiredPort == PortType::In ? _connection->getPortIndex(oppositePort(requiredPort)) : portIndex;
	PortIndex portIndexInp = requiredPort == PortType::In ? portIndex : _connection->getPortIndex(oppositePort(requiredPort));

	auto dataOut = nodeExp->nodeDataModel()->outData(portIndexExp);

	if (!nodeInp->nodeDataModel()->tryInData(portIndexInp, dataOut))
		return false;

	// 4) WtConnection type equals node port type, or there is a registered type conversion that can translate between the two
	auto connectionDataType = _connection->dataType(oppositePort(requiredPort));

	auto const& modelTarget = _node->nodeDataModel();
	NodeDataType candidateNodeDataType = modelTarget->dataType(requiredPort, portIndex);

	if (connectionDataType.id != candidateNodeDataType.id)
	{
		if (requiredPort == PortType::In)
		{
			converter = _scene->registry().getTypeConverter(connectionDataType, candidateNodeDataType);
		}
		else if (requiredPort == PortType::Out)
		{
			converter = _scene->registry().getTypeConverter(candidateNodeDataType, connectionDataType);
		}

		return (converter != nullptr);
	}

	return true;
}

bool WtInteraction::tryConnect()
{
	// 1) Check conditions from 'canConnect'
	PortIndex portIndex = INVALID_PORT;

	TypeConverter converter;

	if (!canConnect(portIndex, converter))
	{
		return false;
	}

	// 1.5) If the connection is possible but a type conversion is needed,
	//      assign a convertor to connection
	if (converter)
	{
		_connection->setTypeConverter(converter);
	}

	// 2) Assign node to required port in QtConnection
	PortType requiredPort = connectionRequiredPort();
	_node->nodeState().setConnection(requiredPort, portIndex, *_connection);

	// 3) Assign QtConnection to empty port in NodeState
	// The port is not longer required after this function
	_connection->setNodeToPort(*_node, requiredPort, portIndex);

	// 4) Poke model to intiate data transfer
	auto outNode = _connection->getNode(PortType::Out);
	if (outNode)
	{
		//PortIndex outPortIndex = _connection->getPortIndex(PortType::Out);
		//outNode->onDataUpdated(outPortIndex);
		setInData(_inPoint.portIndex);
	}

	return true;
}

PortType WtInteraction::connectionRequiredPort() const
{
	auto const& state = _connection->connectionState();

	return state.requiredPort();
}

bool WtInteraction::isNodePortAccessible(PortType portType, PortIndex portIndex) const
{
	WtNodeState const& nodeState = _node->nodeState();

	auto const& entries = nodeState.getEntries(portType);

	if (entries[portIndex].empty()) return true;

	if (portType == PortType::Out)
	{
		const auto outPolicy = _node->nodeDataModel()->portOutConnectionPolicy(portIndex);
		return outPolicy == WtNodeDataModel::ConnectionPolicy::Many;
	}
	else
	{
		const auto inPolicy = _node->nodeDataModel()->portInConnectionPolicy(portIndex);
		return inPolicy == WtNodeDataModel::ConnectionPolicy::Many;
	}
}


void WtInteraction::setInData(PortIndex portIndex)
{
	if (_inPoint.portShape == PortShape::Diamond || _inPoint.portShape == PortShape::Bullet)
	{
		_outNode->connect(_inNode->getImportNodes()[portIndex]);
	}
	else if (_inPoint.portShape == PortShape::Point)
	{
		auto outFieldNum = 0;
		auto outPoints = _connection->getNode(PortType::Out)->flowNodeData().getPointsData();
		for (auto point : outPoints)
		{
			if (point.portShape == PortShape::Point)
			{
				outFieldNum = point.portIndex;
				break;
			}
		}

		dyno::FBase* field;

		if (_outNode == nullptr && _outModule != nullptr)
		{
			field = _outModule->getOutputFields()[_outPoint.portIndex - outFieldNum];
		}
		else
		{
			field = _outNode->getOutputFields()[_outPoint.portIndex - outFieldNum];
		}


		if (field != NULL)
		{
			auto node_data = _node->flowNodeData();

			auto points = node_data.getPointsData();

			int fieldNum = 0;

			for (auto point : points)
			{
				if (point.portType == PortType::In)
				{
					if (point.portShape == PortShape::Bullet || point.portShape == PortShape::Diamond)
					{
						fieldNum++;
					}
				}
			}

			dyno::FBase* inField;

			if (_inNode == nullptr && _inModule != nullptr)
			{
				inField = _inModule->getInputFields()[_inPoint.portIndex - fieldNum];
			}
			else
			{
				inField = _inNode->getInputFields()[_inPoint.portIndex - fieldNum];
			}

			if(inField != nullptr)
				field->connect(inField);
		}
	}
}
