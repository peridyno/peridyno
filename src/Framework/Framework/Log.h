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
#include <list>
#include <fstream>
#include <ctime>

namespace dyno {

	/*!
	 *	\brief	System for logging runtime library errors, warnings, etc.
	 *
	 *	This is the class that catches every (debug) info, warning, error, or user message and
	 *	processes it. Messages can be written to files and/or forwarded to user function for
	 *	processing messages.
	 */
	class Log
	{
	public:
		Log(){}
		~Log();

		/*!
		 *	\brief	Types of logged messages.
		 */
		enum MessageType
		{
			DebugInfo,	//!< Message with some debug information.
			Info,		//!< Information to user.
			Warning,	//!< Warning information.
			Error,		//!< Error information while executing something.
			User		//!< User specific message.
		};

		/*!
		 *	\brief	Logged message of type MessageType with some info.
		 */
		struct Message
		{
			MessageType type;
			std::string text;
			tm* when;
		};

		/*!
		 *	\brief	Open file where to log the messages.
		 */
		static void setOutput(const std::string& filename);

		/*!
		 *	\brief	Get the filename of log.
		 */
		static const std::string& getOutput() { return outputFile; }

		/*!
		 *	\brief	Add a new message to log.
		 *	\param	type	Type of the new message.
		 *	\param	text	Message.
		 *	\remarks Message is directly passes to user receiver if one is set.
		 */
		static void sendMessage(MessageType type, const std::string& text);

		/*!
		 *	\brief	Get the list of all of the logged messages.
		 */
		static std::list<Message>& getMessages() { return messages; }

		/*!
		 *	\brief	Get the last logged message.
		 */
		static const Message& getLastMessage() { return messages.back(); }

		/*!
		 *	\brief	Set user function to receive newly sent messages to logger.
		 */
		static void setUserReceiver(void (*userFunc)(const Message&)) { receiver = userFunc; }

		/*!
		 *	\brief	Set minimum level of message to be logged to file.
		 */
		static void setLevel(MessageType level);

	private:

		static std::string outputFile;
		static std::ofstream outputStream;
		static std::list<Message> messages;
		static MessageType logLevel;
		static void (*receiver)(const Message&);
	};

	// simple debug macro
#ifdef NDEBUG
	#define LogDebug(DESC)
#else
	#define LogDebug(DESC) Log::sendMessage(Log::DebugInfo, DESC)
#endif

}
