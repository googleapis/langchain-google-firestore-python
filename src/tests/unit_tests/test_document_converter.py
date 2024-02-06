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

from langchain_core.documents import Document
from langchain_google_firestore.utility.document_converter import DocumentConverter
from google.cloud.firestore_v1.base_document import DocumentSnapshot
from google.cloud.firestore_v1.document import DocumentReference
from google.cloud.firestore_v1._helpers import GeoPoint

@pytest.mark.parametrize(
        "document_snapshot,langchain_doc", [
            (DocumentSnapshot(
                reference=DocumentReference('foo','bar'),
                data={'field_1':'data_1','field_2':2},
                exists=True,
                read_time=None,
                create_time=None,
                update_time=None),
             Document(
                 page_content="{'field_1': 'data_1', 'field_2': 2}",
                 metadata={'reference':{'path':'foo/bar'}})),
             (DocumentSnapshot(
                reference=DocumentReference('foo','bar'),
                data={'field_1':GeoPoint(1,2),
                      'field_2':['data',2,{'nested':DocumentReference('abc','xyz')}]},
                exists=True,
                read_time=None,
                create_time=None,
                update_time=None),
             Document(
                 page_content="{'field_1': {'latitude': 1, 'longitude': 2}, " +
                 "'field_2': ['data', 2, {'nested': {'path': 'abc/xyz'}}]}",
                 metadata={'reference':{'path':'foo/bar'}})),

        ])
def test_convert_firestore_document_default_fields(document_snapshot, langchain_doc) -> None:
    return_doc = DocumentConverter.convertFirestoreDocument(document_snapshot)

    assert return_doc == langchain_doc

@pytest.mark.parametrize(
    "document_snapshot,langchain_doc,page_content_fields,metadata_fields", [
        (DocumentSnapshot(
          reference=DocumentReference("abc","xyz"),
          data={'data_field':'data','extra_field':1},
          exists=True,
          read_time=None,
          create_time=None,
          update_time=None),
         Document(
             page_content="data",
             metadata={'reference':{'path':'abc/xyz'},'data_field':'data'}),
         ['data_field'],
         ['data_field']),
        (DocumentSnapshot(
          reference=DocumentReference("abc","xyz"),
          data={'field_1':1,'field_2':'val'},
          exists=True,
          read_time=None,
          create_time=None,
          update_time=None),
         Document(
             page_content="val",
             metadata={'reference':{'path':'abc/xyz'},'field_1':1}),
         ['field_2'],
         ['field_1']),
        (DocumentSnapshot(
          reference=DocumentReference("abc","xyz"),
          data={'field_1':'val_1','field_2':'val_2','field_3':'val_3','field_4':'val_4'},
          exists=True,
          read_time=None,
          create_time=None,
          update_time=None),
         Document(
             page_content="{'field_2': 'val_2', 'field_3': 'val_3'}",
             metadata={'reference':{'path':'abc/xyz'},'field_1':'val_1'}),
         ['field_2', 'field_3'],
         ['field_1']),
        (DocumentSnapshot(
          reference=DocumentReference("abc","xyz"),
          data={'field_1':'val_1','field_2':'val_2','field_3':'val_3','field_4':'val_4'},
          exists=True,
          read_time=None,
          create_time=None,
          update_time=None),
         Document(
             page_content="{'field_2': 'val_2', 'field_3': 'val_3'}",
             metadata={'reference':{'path':'abc/xyz'},'field_1':'val_1','field_4':'val_4'}),
         [],
         ['field_1', 'field_4']),
        (DocumentSnapshot(
         reference=DocumentReference("abc","xyz"),
          data={'field_1':'val_1','field_2':'val_2','field_3':'val_3','field_4':'val_4'},
          exists=True,
          read_time=None,
          create_time=None,
          update_time=None),
         Document(
             page_content="{'field_2': 'val_2', 'field_4': 'val_4'}",
             metadata={'reference':{'path':'abc/xyz'},'field_1':'val_1','field_3':'val_3'}),
         ['field_2','field_4'],
         []),
    ])
def test_convert_firestore_document_with_filters(
    document_snapshot,
    langchain_doc,
    page_content_fields,
    metadata_fields) -> None:
  return_doc = DocumentConverter.convertFirestoreDocument(
      document_snapshot,
      page_content_fields,
      metadata_fields)

  assert return_doc == langchain_doc

