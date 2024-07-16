#pragma once

enum class NodeValidationState
{
	Valid,
	Warning,
	Error
};

class WtNodeDataModel
{
public:
	WtNodeDataModel();

	virtual ~WtNodeDataModel() = default;
};