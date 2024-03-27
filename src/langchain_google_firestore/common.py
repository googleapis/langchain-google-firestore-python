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

from typing import Optional

from google.cloud import firestore  # type: ignore
from google.cloud.firestore_v1.services.firestore.transports.base import (  # type: ignore
    DEFAULT_CLIENT_INFO,
)


def client_with_user_agent(
    user_agent: str, client: Optional[firestore.Client] = None
) -> firestore.Client:
    client_info = DEFAULT_CLIENT_INFO
    client_info.user_agent = user_agent
    if not client:
        client = firestore.Client(client_info=client_info)
    client_agent = client._client_info.user_agent
    if not client_agent:
        client._client_info.user_agent = user_agent
    elif user_agent not in client_agent:
        client._client_info.user_agent = " ".join([user_agent, client_agent])
    return client
