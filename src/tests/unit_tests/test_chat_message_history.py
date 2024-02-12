# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import uuid

from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from multiprocessing import Pool


def test_firestore_history_workflow() -> None:
    session_id = uuid.uuid4().hex
    chat_history = FirestoreChatMessageHistory(
        session_id=session_id, collection="HistoryWorkflow"
    )

    chat_history.add_ai_message("AI message")
    chat_history.add_user_message("User message")

    expected_messages = [
        AIMessage(content="AI message"),
        HumanMessage(content="User message"),
    ]

    pytest.case.assertCountEqual(expected_messages, chat_history.messages)

    chat_history.clear()
    chat_history.load_messages()

    assert len(chat_history.messages) == 0

def test_firestore_load_messages() -> None:
  NUM_MESSAGES =  25
  session_id = uuid.uuid4().hex
  chat_history = FirestoreChatMessageHistory(
      session_id=session_id, collection="HistoryLoad")

  expected_messages = []

  for i in range(NUM_MESSAGES):
    ai_m = AIMessage(content=f"AI message: {i}")
    human_m = HumanMessage(content=f"Human message: {i}")

    expected_messages.append(ai_m)
    expected_messages.append(human_m)

    chat_history.add_ai_message(ai_m)
    chat_history.add_user_message(human_m)

  pytest.case.assertCountEqual(expected_messages, chat_history.messages)

  chat_history.clear()
  chat_history.load_messages()

  assert len(chat_history.messages) == 0

def test_firestore_multiple_sessions() -> None:
  collection = "MultipleSession"
  session_1 = uuid.uuid4().hex
  chat_history_1 = FirestoreChatMessageHistory(
      session_id=session_1, collection=collection)
  session_2 = uuid.uuid4().hex
  chat_history_2 = FirestoreChatMessageHistory(
      session_id=session_2, collection=collection)

  chat_history_1.add_ai_message("AI message session 1")
  chat_history_1.add_user_message("Human message session 1")
  chat_history_2.add_ai_message("AI message session 2")
  chat_history_2.add_user_message("Human message session 2")

  expected_message_session_1 = [
      AIMessage(content="AI message session 1"),
      HumanMessage(content="Human message session 1")]
  expected_message_session_2 = [
      AIMessage(content="AI message session 2"),
      HumanMessage(content="Human message session 2")]

  pytest.case.assertCountEqual(expected_message_session_1, chat_history_1.messages)
  pytest.case.assertCountEqual(expected_message_session_2, chat_history_2.messages)

  chat_history_1.clear()
  chat_history_2.clear()

  assert len(chat_history_1.messages) == 0
  assert len(chat_history_2.messages) == 0

