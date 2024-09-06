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
* 
* @brief Initial commit
* @author clouddon
* @date 2021-02-25
* 
* @brief Support multi-threading
* @author star
* @date 2023-08-18
* 
*/
#pragma once

#include <mutex>
#include <thread>
#include <string>
#include <fstream>
#include <iostream>
#include <ctime>
#include <queue>
#include <cstdio>
#include <cassert>
#include <cstdarg>

namespace dyno
{
    class Log
    {
    public:
        enum MessageType
        {
            DebugInfo,	//!< Message with some debug information.
            Info,		//!< Information to user.
            Warning,	//!< Warning information.
            Error,		//!< Error information while executing something.
            User,		//!< User specific message.
        };

        struct Message {
            MessageType type;
            std::string text;
            tm* when;
        };

        static Log* instance();


        /*!
         *	\brief	Add a new message to log.
         *	\param	type	Type of the new message.
         *	\param	text	Message.
         *	\remarks Message is directly passes to user receiver if one is set.
         */
        static void sendMessage(MessageType type, const std::string& text);

         /*!
		  *	\brief	Set user function to receive newly sent messages to logger.
		  */
        static void setUserReceiver(void (*userFunc)(const Message&));

        /*!
		 *	\brief	Set minimum level of message to be logged to file.
		 */
        static void setLevel(MessageType level);

        /*!
		 *	\brief	Open file where to log the messages.
		 */
        static void setOutput(const std::string& filename);

        /*!
         *	\brief	Get the filename of log.
         */
        static const std::string& getOutput();

    private:

        Log();
        ~Log();

        void outputThread();

		//Add a new message to log
		void writeMessage(MessageType level, const char* format, ...);

    private:
        bool mRunning;

		std::mutex mtx;
		std::thread mThread;
		std::condition_variable mCondition;

        static std::atomic<Log*> sLogInstance;

        static std::queue<Message> sMessageQueue;
        static MessageType sLogLevel;
		static std::string sOutputFile;
		static std::ofstream sOutputStream;
        static void (*receiver)(const Message&);
    };
}
